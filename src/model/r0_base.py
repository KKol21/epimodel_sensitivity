from abc import ABC, abstractmethod

import torch


class R0GeneratorBase(ABC):
    def __init__(self, param: dict, states: list, n_age: int):
        self.states = states
        self.n_age = n_age
        self.parameters = param
        self.n_states = len(self.states)
        self.i = {self.states[index]: index for index in torch.arange(0, self.n_states)}
        self.s_mtx = self.n_age * self.n_states

        self.v_inv = None
        self.e = None
        self.contact_matrix = torch.zeros((n_age, n_age))

    def _idx(self, state: str) -> bool:
        return torch.arange(self.n_age * self.n_states) % self.n_states == self.i[state]

    def get_eig_val(self, susceptibles: torch.Tensor, population: torch.Tensor,
                    contact_mtx: torch.Tensor = None) -> float:
        # contact matrix needed for effective reproduction number: [c_{j,i} * S_i(t) / N_i(t)]
        if contact_mtx is not None:
            self.contact_matrix = contact_mtx
        cm = self.contact_matrix / population.reshape((-1, 1))
        cm = cm * susceptibles
        f = self._get_f(cm)
        self._get_v()
        ngm_large = self.v_inv @ f
        ngm = self.e @ ngm_large @ self.e.T

        dom_eig_val = torch.sort(torch.abs(
                torch.linalg.eigvals(ngm)))[0][-1]
        return float(dom_eig_val)

    @abstractmethod
    def _get_e(self):
        pass

    @abstractmethod
    def _get_v(self):
        pass

    @abstractmethod
    def _get_f(self, contact_matrix: torch.Tensor):
        pass
