from abc import abstractmethod, ABC
from typing import Callable


class MockModelBase(ABC):
    """
    Base class for unit testing example models validity.
    """
    def __init__(self, data):
        self.data = data
        self.ps = data.ps
        self.cm = data.cm
        self.age_data = data.age_data

    @abstractmethod
    def odefun(self, t, y) -> Callable:
        pass
