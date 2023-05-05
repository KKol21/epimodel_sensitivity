import torch

from src.model.r0_base import R0GeneratorBase
from src.model.model import get_n_states


def generate_transition_block(transition_param: float, n_states: int) -> torch.Tensor:
    trans_block = torch.zeros((n_states, n_states))
    # Outflow from states (diagonal elements)
    trans_block = trans_block.fill_diagonal_(-transition_param)
    # Inflow to states (elements under the diagonal)
    trans_block[1:, :n_states - 1] = trans_block[1:, :n_states - 1].fill_diagonal_(transition_param)
    return trans_block


class R0Generator(R0GeneratorBase):
    def __init__(self, param: dict, n_age: int = 16):
        self.n_e = param["n_e"]
        self.n_i = param["n_i"]
        states = get_n_states(self.n_e, "e") + \
                 get_n_states(self.n_i, "i")
        super().__init__(param=param, states=states, n_age=n_age)

        self._get_e()

    def _get_v(self) -> None:
        idx = self._idx
        v = torch.zeros((self.s_mtx, self.s_mtx))
        params = self.parameters

        e_trans = generate_transition_block(params["alpha"], self.n_e)
        i_trans = generate_transition_block(params["gamma"], self.n_i)
        for age_group in range(self.n_age):
            e_start = age_group * self.n_states
            e_end = e_start + self.n_e
            # Transition between exposed states
            v[e_start:e_end, e_start:e_end] = e_trans

            i_start = e_end
            i_end = i_start + self.n_i
            # Transition between infected states
            v[i_start:i_end, i_start:i_end] = i_trans
        # Transition from last exposed state to first infected to state
        v[idx('i_0'), idx(f'e_{self.n_e - 1}')] = params["alpha"]
        self.v_inv = torch.linalg.inv(v)

    def _get_f(self, contact_mtx: torch.Tensor) -> torch.Tensor:
        i = self.i
        s_mtx = self.s_mtx
        n_states = self.n_states

        f = torch.zeros((s_mtx, s_mtx))
        susc_vec = self.parameters["susc"].reshape((-1, 1))
        # Rate of infection for every infected state
        for inf_state in get_n_states(self.n_i, "i"):
            f[i["e_0"]:s_mtx:n_states, i[inf_state]:s_mtx:n_states] = torch.mul(contact_mtx, susc_vec)
        return f

    def _get_e(self):
        block = torch.zeros(self.n_states, )
        block[0] = 1
        self.e = block
        for i in range(1, self.n_age):
            self.e = torch.block_diag(self.e, block)
