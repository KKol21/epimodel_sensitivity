import torch

from .mock_model_base import MockModelBase


class MockSIRModel(MockModelBase):
    def odefun(self, t, y):
        s, i, r = y.unbind(dim=-1)
        transmission = self.ps["beta"] * s * i / self.population
        recovery = self.ps["gamma"] * i
        return torch.stack((-transmission, transmission - recovery, recovery), dim=-1)
