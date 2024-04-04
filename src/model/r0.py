import torch

from src.model.matrix_generator import generate_transition_matrix, get_tms_multipliers, get_distr_mul
from src.model.model_base import get_substates


class R0Generator:
    def __init__(self, data, state_data, trans_data, tms_rules):
        self.data = data
        self.device = data.device
        self.state_data = state_data
        self.trans_data = trans_data
        self.tms_rules = tms_rules

        self.inf_states = self.get_infected_states()
        self.n_comp = len(self.inf_states)
        self.n_age = data.n_age
        self.n_states = len(self.inf_states)
        self.params = data.params
        self.i = {self.inf_states[index]: index for index in torch.arange(0, self.n_states)}
        self.s_mtx = self.n_age * self.n_states

        self._get_e()

    def get_infected_states(self):
        states = []
        for state, data in self.state_data.items():
            if data["type"] in ["infected", "infectious"]:
                states += get_substates(data["n_substates"], state)
        return states

    def _idx(self, state: str) -> bool:
        return torch.arange(self.n_age * self.n_states) % self.n_states == self.i[state]

    def get_eig_val(self, susceptibles: torch.Tensor, population: torch.Tensor,
                    contact_mtx: torch.Tensor) -> float:
        # contact matrix needed for effective reproduction number: [c_{j,i} * S_i(t) / N_i(t)]
        cm = contact_mtx / population.reshape((-1, 1))
        cm = cm * susceptibles
        f = self._get_f(cm)
        v_inv = self._get_v()
        ngm_large = v_inv @ f
        ngm = self.e @ ngm_large @ self.e.T if self.n_age > 1 else self.e @ ngm_large @ self.e

        if self.n_age == 1:
            dom_eig_val = torch.abs(ngm)
        else:
            dom_eig_val = torch.sort(torch.abs(torch.linalg.eigvals(ngm)))[0][-1]
        return float(dom_eig_val)

    def _get_v(self) -> torch.Tensor:
        """
        Compute and store the inverse of the transition matrix.

        """

        def isinf_state(state):
            return state in [state for state, data in self.state_data.items() if
                             data["type"] in ["infected", "infectious"]]

        inf_state_dict = {state: data for state, data in self.state_data.items() if isinf_state(state)}
        trans_mtx = generate_transition_matrix(states_dict=inf_state_dict, trans_data=self.trans_data,
                                               parameters=self.params, n_age=self.n_age,
                                               n_comp=self.n_states, c_idx=self.i).to(self.device)

        end_state_dict = {state: f"{state}_{data['n_substates'] - 1}"
                          for state, data in self.state_data.items()}
        basic_trans = [trans for trans in self.trans_data if
                       isinf_state(trans['source']) and isinf_state(trans['target'])]
        idx = self._idx

        for trans in basic_trans:
            source = end_state_dict[trans['source']]
            target = f"{trans['target']}_0"
            param = self.params[trans['param']]
            n_substates = self.state_data[trans['source']]["n_substates"]
            distr = get_distr_mul(distr=trans["distr"], params=self.params)
            trans_mtx[idx(source), idx(target)] = param * distr * n_substates
        return torch.linalg.inv(trans_mtx)

    def _get_f(self, contact_mtx: torch.Tensor) -> torch.Tensor:
        """
       Compute the matrix representing the rate of infection.

       Args:
           contact_mtx (torch.Tensor): The contact matrix.

       Returns:
           torch.Tensor: The matrix representing the rate of infection.
       """
        i = self.i
        s_mtx = self.s_mtx
        n_states = self.n_states
        f = torch.zeros((s_mtx, s_mtx)).to(self.device)

        for tms in self.tms_rules:
            susc_mul, inf_mul = get_tms_multipliers(tms_rule=tms, data=self.data)
            for actor in tms["infection"]["actors-params"]:
                for substate in get_substates(n_substates=self.state_data[actor]["n_substates"],
                                              comp_name=actor):
                    f[i[substate]:s_mtx:n_states, i[f"{tms['target']}_0"]:s_mtx:n_states] =\
                        susc_mul * contact_mtx.T * inf_mul.unsqueeze(0)
        return f

    def _get_e(self):
        block = torch.zeros(self.n_states, ).to(self.device)
        block[0] = 1
        self.e = block
        for i in range(1, self.n_age):
            self.e = torch.block_diag(self.e, block)
