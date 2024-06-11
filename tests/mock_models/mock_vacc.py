import torch
from . import MockModelBase


class MockVaccinatedModel(MockModelBase):
    def __init__(self, data):
        super().__init__(data)

    def odefun(self, t, y):
        (s, e1, e2, e3, i1, i2, i3, i4, i5,
         h, ic, icr, r, d, v) = self.get_comp_vals(y)

        ps = self.ps
        actual_population = torch.tensor(self.population, dtype=torch.float32)

        # Compute transmission
        infectious_terms = i1 + i2 + i3 + i4 + i5
        transmission = self.get_transmission(infectious_terms)

        ds = (s / actual_population) * transmission - \
             s / (s + r) * ps["daily_vacc"]
        de1 = (s / actual_population) * transmission - 3 * ps["alpha_l"] * e1
        de2 = 3 * ps["alpha_l"] * e1 - 2 * ps["alpha"] * e2
        de3 = 3 * ps["alpha_l"] * e2 - 3 * ps["alpha"] * e3

        di1 = ps["alpha"] * e3 - 5 * ps["gamma"] * i1
        di2 = 5 * ps["gamma"] * i1 - 5 * ps["gamma"] * i2
        di3 = 5 * ps["gamma"] * i2 - 5 * ps["gamma"] * i3
        di4 = 5 * ps["gamma"] * i3 - 5 * ps["gamma"] * i4
        di5 = 5 * ps["gamma"] * i4 - 5 * ps["gamma"] * i5

        dh = ps["h"] * (1 - ps["xi"]) * 5 * ps["gamma"] * i5 - ps["gamma_h"] * h
        dic = ps["h"] * ps["xi"] * 5 * ps["gamma"] * i5 - ps["gamma_c"] * ic
        dicr = (1 - ps["mu"]) * ps["gamma_c"] * ic - ps["gamma_cr"] * icr

        dr = (3 * ps["gamma_a"] *
              ps["gamma_h"] * h + ps["gamma_cr"] * icr)
        dd = ps["mu"] * ps["gamma_c"] * ic
        dv = s / (s + r) * ps["daily_vacc"]

        return self.concat_sol(ds, de1, de2, de3, di1, di2, di3, di4, di5, dh, dic, dicr, dr, dd, dv)
