import torch

from src.model.matrix_generator import generate_transition_matrix, get_infectious_states


class R0Generator:
    def __init__(self, data):
        self.data = data
        self.device = data.device
        self.state_data = data.state_data
        self.trans_data = data.trans_data
        self.inf_states = self.get_infected_states()
        self.n_comp = len(self.inf_states)
        self.n_age = data.n_age
        self.params = data.model_params
        self.n_states = len(self.inf_states)
        self.i = {self.inf_states[index]: index for index in torch.arange(0, self.n_states)}
        self.s_mtx = self.n_age * self.n_states

        self._get_e()
        self.inf_state_dict = {state: data for state, data in self.state_data.items()
                               if data["type"] in ["infected", "infectious"]}
        self.inf_inflow_state = [f"{trans['target']}_0" for trans in self.trans_data.values()
                                 if trans["type"] == "infection"][0]

    def get_infected_states(self):
        from src.model.model_base import get_substates
        states = []
        for state, data in self.data.state_data.items():
            if data["type"] in ["infected", "infectious"]:
                states += get_substates(data["n_substates"], state)
        return states

    def _idx(self, state: str) -> bool:
        return torch.arange(self.n_age * self.n_states) % self.n_states == self.i[state]

    def get_eig_val(self, susceptibles: torch.Tensor, population: torch.Tensor,
                    contact_mtx: torch.Tensor = None) -> float:
        # contact matrix needed for effective reproduction number: [c_{j,i} * S_i(t) / N_i(t)]
        cm = contact_mtx / population.reshape((-1, 1))
        cm = cm * susceptibles
        f = self._get_f(cm)
        v_inv = self._get_v()
        ngm_large = v_inv @ f
        ngm = self.e @ ngm_large @ self.e.T

        if len(ngm.shape) == 0:
            dom_eig_val = torch.abs(ngm)
        else:
            dom_eig_val = torch.sort(torch.abs(torch.linalg.eigvals(ngm)))[0][-1]
        return float(dom_eig_val)

    def _get_v(self) -> torch.Tensor:
        """
        Compute and store the inverse of the transition matrix.

        """

        def isinf_state(state):
            return self.state_data[state]['type'] in ['infected', 'infectious']

        trans_mtx = generate_transition_matrix(states_dict=self.inf_state_dict, trans_data=self.data.trans_data,
                                               parameters=self.data.model_params, n_age=self.n_age,
                                               n_comp=self.n_states, c_idx=self.i).to(self.device)
        end_state = {state: f"{state}_{data['n_substates'] - 1}" for state, data in self.data.state_data.items()}
        basic_trans_dict = {trans: data for trans, data in self.data.trans_data.items()
                            if data['type'] == 'basic'
                            and isinf_state(data['source'])
                            and isinf_state(data['target'])}

        for trans, data in basic_trans_dict.items():
            param = self.data.model_params[data['param']]
            trans_mtx[self._idx(end_state[data['source']]), self._idx(f"{data['target']}_0")] = param
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

        infectious_states = get_infectious_states(state_data=self.state_data)

        f = torch.zeros((s_mtx, s_mtx)).to(self.device)
        susc_vec = self.params["susc"].reshape((-1, 1))
        # Rate of infection for every infectious state
        for inf_state in infectious_states:
            f[i[inf_state]:s_mtx:n_states, i[self.inf_inflow_state]:s_mtx:n_states] = torch.mul(susc_vec, contact_mtx.T)
        return f

    def _get_e(self):
        block = torch.zeros(self.n_states, ).to(self.device)
        block[0] = 1
        self.e = block
        for i in range(1, self.n_age):
            self.e = torch.block_diag(self.e, block)
