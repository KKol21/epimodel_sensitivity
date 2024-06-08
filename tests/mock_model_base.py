from abc import abstractmethod, ABC

import torch


class MockModelBase(ABC):
    """
    Base class for unit testing example models validity.
    """

    def __init__(self, data):
        self.data = data
        self.ps = data.ps
        self.cm = data.cm
        self.population = data.age_data.flatten()

    @abstractmethod
    def odefun(self, t, y: torch.Tensor) -> torch.Tensor:
        pass
