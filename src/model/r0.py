import numpy as np
from scipy.linalg import block_diag
import torch

from src.model.r0_base import R0GeneratorBase


class R0Generator(R0GeneratorBase):
    def __init__(self, param: dict, n_age: int = 16):
        states = ["e", "i"]
        super().__init__(param=param, states=states, n_age=n_age)
        self.n_l = 2
        self.n_a = 3
        self.n_i = 3

        self._get_e()

    def _get_v(self) -> np.array:
        idx = self._idx
        v = np.zeros((self.n_age * self.n_states, self.n_age * self.n_states))

        v[idx("e"), idx("e")] = - self.parameters["alpha"]
        v[idx("i"), idx("e")] = self.parameters["alpha"]
        v[idx("i"), idx("i")] = - self.parameters["gamma"]
        self.v_inv = np.linalg.inv(v)

    def _get_f(self, contact_mtx: np.array) -> np.array:
        i = self.i
        s_mtx = self.s_mtx
        n_states = self.n_states

        f = np.zeros((self.n_age * n_states, self.n_age * n_states))
        susc_vec = self.parameters["susc"].reshape((-1, 1))
        f[i["e"]:s_mtx:n_states, i["i"]:s_mtx:n_states] = torch.mul(contact_mtx.T.to('cpu'), susc_vec.to('cpu'))
        return f

    def _get_e(self):
        block = np.zeros(self.n_states, )
        block[0] = 1
        self.e = block
        for i in range(1, self.n_age):
            self.e = block_diag(self.e, block)
