from . import MockModelBase
import torch


class MockSEIHRModel(MockModelBase):
    def __init__(self, data):
        super().__init__(data)

    def odefun(self, t, y):
        S, E, I, R = y.squeeze(0)
        ps = self.ps
        age_data = self.age_data

        dSdt = -ps["beta"] * S / age_data * I
        dEdt = ps["beta"] * S / age_data * I - ps["alpha"] * E
        dIdt = ps["alpha"] * E - ps["gamma"] * I
        dRdt = ps["gamma"] * I

        dydt = [dSdt, dEdt, dIdt, dRdt]
        return torch.tensor([dydt])

