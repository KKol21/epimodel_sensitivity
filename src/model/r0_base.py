from abc import ABC, abstractmethod

import torch


class R0GeneratorBase(ABC):
    def __init__(self, data, n_age: int):
        self.data = data
        self.state_data = data.state_data
        self.trans_data = data.trans_data
        self.inf_states = self.get_infected_states()
        self.n_comp = len(self.inf_states)
        self.n_age = n_age
        self.parameters = data.model_parameters
        self.n_states = len(self.inf_states)
        self.i = {self.inf_states[index]: index for index in torch.arange(0, self.n_states)}
        self.s_mtx = self.n_age * self.n_states

        self.v_inv = None
        self.e = None
        self.contact_matrix = torch.zeros((n_age, n_age))
        self.inf_state_dict = {state: data for state, data in self.data.state_data.items()
                               if data["type"] in ["infected", "infectious"]}

    def get_infected_states(self):
        from src.model.model import get_substates
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
