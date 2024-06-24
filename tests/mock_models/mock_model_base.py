from abc import abstractmethod, ABC

import torch


class MockModelBase(ABC):
    """
    Base class for unit testing example models validity.
    """

    def __init__(self, data):
        self.data = data
        self.ps = data.params
        self.cm = data.cm
        self.population = data.age_data.flatten()
        self.n_age = len(self.population)

    @abstractmethod
    def odefun(self, t, y: torch.Tensor) -> torch.Tensor:
        pass

    def get_comp_vals(self, y):
        return y.reshape(self.n_age, y.shape[1] // self.n_age).T

    @staticmethod
    def concat_sol(*args):
        return torch.stack(args).T.flatten()

    def get_transmission(self, infectious_terms):
        return self.ps["beta"] * infectious_terms @ self.cm.T / self.population
