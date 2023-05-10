from abc import ABC, abstractmethod

import numpy as np
from scipy.integrate import odeint
import torch


class EpidemicModelBase(ABC):
    def __init__(self, model_data, compartments):
        self.ps = model_data.model_parameters_data
        self.population = model_data.age_data.flatten()
        self.compartments = compartments
        self.n_comp = len(compartments)
        self.c_idx = {comp: idx for idx, comp in enumerate(compartments)}
        self.n_age = self.population.shape[0]
        self.device = model_data.device

    def initialize(self):
        iv = {key: torch.zeros(self.n_age).to(self.device) for key in self.compartments}
        return iv

    def aggregate_by_age(self, solution, comp):
        return solution[-1, self.idx(comp)].sum()

    def get_solution(self, t, parameters, cm):
        initial_values = self.get_initial_values(parameters)
        return np.array(odeint(self.get_model, initial_values, t, args=(parameters, cm)))

    def get_array_from_dict(self, comp_dict):
        return torch.cat([comp_dict[comp] for comp in self.compartments], 0)

    def get_initial_values(self, parameters):
        iv = self.initialize()
        self.update_initial_values(iv=iv, parameters=parameters)
        return self.get_array_from_dict(comp_dict=iv)

    @abstractmethod
    def update_initial_values(self, iv, parameters):
        pass

    @abstractmethod
    def get_model(self, ts, xs, ps, cm):
        pass

    def aggregate_by_age_n_state(self, solution, comp):
        result = 0
        for state in get_n_states(self.ps[f'n_{comp}'], comp):
            result += max(solution[:, self.idx(state)].sum(axis=1))
        return result

    def idx(self, state: str) -> bool:
        return torch.arange(self.n_age * self.n_comp) % self.n_comp == self.c_idx[state]


def get_n_states(n_classes, comp_name):
    return [f"{comp_name}_{i}" for i in range(n_classes)]