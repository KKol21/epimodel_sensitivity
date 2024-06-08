from . import MockModelBase
import torch


class MockSEIRModel(MockModelBase):
    def __init__(self, data):
        super().__init__(data)

    def odefun(self, t, y):
        S, E, I, R = y.squeeze(0)
        ps = self.ps
        population = self.population

        dSdt = -ps["beta"] * S / population * I
        dEdt = ps["beta"] * S / population * I - ps["alpha"] * E
        dIdt = ps["alpha"] * E - ps["gamma"] * I
        dRdt = ps["gamma"] * I

        dydt = [dSdt, dEdt, dIdt, dRdt]
        return torch.tensor([dydt])

