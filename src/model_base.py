from abc import ABC, abstractmethod

import numpy as np
from scipy.integrate import odeint


class EpidemicModelBase(ABC):
    def __init__(self, model_data, compartments):
        self.population = model_data.age_data.flatten()
        self.compartments = compartments
        self.c_idx = {comp: idx for idx, comp in enumerate(self.compartments)}
        self.n_age = self.population.shape[0]

    def initialize(self):
        iv = {key: np.zeros(self.n_age) for key in self.compartments}
        iv.update({"i": np.ones(self.n_age)})
        return iv

    def aggregate_by_age(self, solution, idx):
        return np.sum(solution[:, idx * self.n_age:(idx + 1) * self.n_age], axis=1)

    def get_cumulative(self, solution):
        idx = self.c_idx["c"]
        return self.aggregate_by_age(solution, idx)

    def get_deaths(self, solution):
        idx = self.c_idx["d"]
        return self.aggregate_by_age(solution, idx)

    def get_solution(self, t, parameters, cm):
        initial_values = self.get_initial_values()
        return np.array(odeint(self.get_model, initial_values, t, args=(parameters, cm)))

    def get_array_from_dict(self, comp_dict):
        return np.array([comp_dict[comp] for comp in self.compartments]).flatten()

    def get_initial_values(self):
        iv = self.initialize()
        self.update_initial_values(iv=iv)
        return self.get_array_from_dict(comp_dict=iv)

    @abstractmethod
    def update_initial_values(self, iv):
        pass

    @abstractmethod
    def get_model(self, xs, ts, ps, cm):
        pass

