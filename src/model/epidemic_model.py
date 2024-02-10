import torch

from src.model.model_base import EpidemicModelBase


class EpidemicModel(EpidemicModelBase):
    def __init__(self, data):
        super().__init__(data)

    def get_solution(self, y0, t_eval, **kwargs):
        return self.get_sol_from_ode(y0=y0,
                                     t_eval=t_eval,
                                     odefun=self.get_ode())

    def get_ode(self):
        if self.is_vaccinated:
            v_div = torch.ones(self.n_eq).to(self.device)
            div_idx = self.idx('s_0') + self.idx('v_0')

            def vaccinated_ode(t, y):
                base_result = torch.mul(y @ self.A, y @ self.T) + y @ self.B
                if self.ps["t_start"] < t[0] < (self.ps["t_start"] + self.ps["T"]):
                    v_div[div_idx] = (y @ self.V_2)[0, div_idx]
                    vacc = torch.div(y @ self.V_1,
                                     v_div)
                    return base_result + vacc
                return base_result

            return vaccinated_ode

        def basic_ode(t, y):
            return torch.mul(y @ self.A, y @ self.T) + y @ self.B

        return basic_ode
