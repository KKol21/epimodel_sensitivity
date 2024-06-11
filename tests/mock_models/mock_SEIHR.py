import torch

from . import MockModelBase


class MockSEIHRModel(MockModelBase):
    def __init__(self, data):
        super().__init__(data)

    def odefun(self, t, y):
        ps = self.ps
        population = self.population
        S, E, I, H, R = self.get_comp_vals(y)

        dSdt = -ps["beta"] * S / population * I
        dEdt = ps["beta"] * S / population * I - ps["alpha"] * E
        dIdt = ps["alpha"] * E * (1 - ps["eta"]) - ps["gamma"] * I
        dHdt = ps["eta"] * ps["alpha"] * E - ps["gamma_h"] * H
        dRdt = ps["gamma"] * I + ps["gamma_h"] * H

        return self.concat_sol(dSdt, dEdt, dIdt, dHdt, dRdt)
