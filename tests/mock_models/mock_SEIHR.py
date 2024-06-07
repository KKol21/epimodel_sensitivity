from . import MockModelBase
import torch


class MockSEIHRModel(MockModelBase):
    def __init__(self, data):
        super().__init__(data)

    def odefun(self, t, y):
        S, E, I, H, R = y.squeeze(0)
        ps = self.ps

        dSdt = -ps["beta"] * S / self.age_data * I
        dEdt = ps["beta"] * S / self.age_data * I - ps["alpha"] * E
        dIdt = ps["alpha"] * E * (1 - ps["eta"]) - ps["gamma"] * I
        dHdt = ps["eta"] * ps["alpha"] * E - ps["gamma_h"] * H
        dRdt = ps["gamma"] * I + ps["gamma_h"] * H

        dydt = [dSdt, dEdt, dIdt, dHdt, dRdt]
        return torch.tensor([dydt])
