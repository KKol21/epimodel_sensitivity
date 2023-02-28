from abc import ABC, abstractmethod
from typing import List

import numpy as np


class R0GeneratorBase(ABC):
    def __init__(self, param: dict, states: list, n_age: int):
        self.states = states
        self.n_age = n_age
        self.parameters = param
        self.n_states = len(self.states)
        self.i = {self.states[index]: index for index in np.arange(0, self.n_states)}
        self.s_mtx = self.n_age * self.n_states

        self.v_inv = None
        self.e = None
        self.contact_matrix = np.zeros((n_age, n_age))

    def _idx(self, state: str) -> bool:
        return np.arange(self.n_age * self.n_states) % self.n_states == self.i[state]

    def get_eig_val(self, susceptibles: np.ndarray, population: np.ndarray,
                    contact_mtx: np.array = None) -> List[np.float]:
        # contact matrix needed for effective reproduction number: [c_{j,i} * S_i(t) / N_i(t)]
        if contact_mtx is not None:
            self.contact_matrix = contact_mtx
        contact_matrix = self.contact_matrix / population.reshape((-1, 1))
        cm_tensor = np.tile(contact_matrix, (susceptibles.shape[0], 1, 1))
        susc_tensor = susceptibles.reshape((susceptibles.shape[0], susceptibles.shape[1], 1))
        contact_matrix_tensor = cm_tensor * susc_tensor
        eig_val_eff = []
        for cm in contact_matrix_tensor:
            f = self._get_f(cm)
            self._get_v()
            ngm_large = f @ self.v_inv
            ngm = self.e @ ngm_large @ self.e.T
            eig_val = np.sort(list(map(lambda x: np.abs(x), np.linalg.eig(ngm)[0])))
            eig_val_eff.append(float(eig_val[-1]))

        return eig_val_eff

    @abstractmethod
    def _get_e(self):
        pass

    @abstractmethod
    def _get_v(self):
        pass

    @abstractmethod
    def _get_f(self, contact_matrix: np.ndarray):
        pass
