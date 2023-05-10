from abc import ABC, abstractmethod

import torch
from torchdiffeq import odeint


class EpidemicModelBase(ABC):
    def __init__(self, model_data, compartments):
        self.ps = model_data.model_parameters_data
        self.population = model_data.age_data.flatten()
        self.compartments = compartments
        self.n_comp = len(compartments)
        self.c_idx = {comp: idx for idx, comp in enumerate(compartments)}
        self.n_age = self.population.shape[0]
        self.device = model_data.device
        self.size = self.n_age * len(self.compartments)

    def get_initial_values(self):
        iv = torch.zeros(self.size)
        age_group = 3 * self.n_comp
        iv[age_group + self.c_idx['e_0']] = 1
        iv[self.idx('s')] = self.population
        iv[age_group + self.c_idx['s']] -= 1
        return iv

    def get_solution(self, t, cm, daily_vac):
        initial_values = self.get_initial_values()
        return odeint(func=self.get_model(cm, daily_vac), y0=initial_values, t=t, method='euler')

    @abstractmethod
    def get_model(self, cm, daily_vac):
        pass

    def idx(self, state: str) -> bool:
        return torch.arange(self.size) % self.n_comp == self.c_idx[state]


def get_n_states(n_classes, comp_name):
    return [f"{comp_name}_{i}" for i in range(n_classes)]
