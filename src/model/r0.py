import torch

from src.model.model import get_n_states
from src.model.r0_base import R0GeneratorBase


def generate_transition_block(transition_param: float, n_states: int) -> torch.Tensor:
    trans_block = torch.zeros((n_states, n_states))
    # Outflow from states (diagonal elements)
    trans_block = trans_block.fill_diagonal_(-transition_param)
    # Inflow to states (elements under the diagonal)
    trans_block[1:, :n_states - 1] = trans_block[1:, :n_states - 1].fill_diagonal_(transition_param)
    return trans_block


def generate_transition_matrix(trans_param_dict, ps, n_age, n_comp, c_idx):
    trans_matrix = torch.zeros((n_age * n_comp, n_age * n_comp))
    for age_group in range(n_age):
        for comp, trans_param in trans_param_dict.items():
            n_states = ps[f'n_{comp}']
            diag_idx = age_group * n_comp + c_idx[f'{comp}_0']
            block_slice = slice(diag_idx, diag_idx + n_states)
            trans_matrix[block_slice, block_slice] = generate_transition_block(trans_param, n_states)
    return trans_matrix


class R0Generator(R0GeneratorBase):
    def __init__(self, param: dict, device, n_age: int):
        from src.model.model import get_n_states
        self.n_e = param["n_e"]
        self.n_i = param["n_i"]
        states = get_n_states(self.n_e, "e") + \
                 get_n_states(self.n_i, "i")
        super().__init__(param=param, states=states, n_age=n_age)

        self.device = device
        self._get_e()

    def _get_v(self) -> None:
        idx = self._idx
        params = self.parameters

        trans_mtx = generate_transition_matrix({"e": self.parameters["alpha"], "i": self.parameters["gamma"]},
                                               self.parameters, self.n_age, self.n_states, self.i)
        trans_mtx[idx('i_0'), idx(f'e_{self.n_e - 1}')] = params["alpha"]
        self.v_inv = torch.linalg.inv(trans_mtx)

    def _get_f(self, contact_mtx: torch.Tensor) -> torch.Tensor:
        i = self.i
        s_mtx = self.s_mtx
        n_states = self.n_states

        f = torch.zeros((s_mtx, s_mtx)).to(self.device)
        susc_vec = self.parameters["susc"].reshape((-1, 1))
        # Rate of infection for every infected state
        for inf_state in get_n_states(self.n_i, "i"):
            f[i["e_0"]:s_mtx:n_states, i[inf_state]:s_mtx:n_states] = torch.mul(contact_mtx, susc_vec)
        return f

    def _get_e(self):
        block = torch.zeros(self.n_states, ).to(self.device)
        block[0] = 1
        self.e = block
        for i in range(1, self.n_age):
            self.e = torch.block_diag(self.e, block)
