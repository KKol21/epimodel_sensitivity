from model_base import EpidemicModelBase
import torch


class EpidemicModel(EpidemicModelBase):
    def __init__(self, data):
        super().__init__(data)

    def get_solution(self, y0, t_eval, **kwargs):
        return self.get_sol_from_ode(y0=y0,
                                     t_eval=t_eval,
                                     odefun=self.get_ode())

    def get_ode(self):
        if self.is_vaccinated:
            def vaccinated_ode(t, y):
                base_result = torch.mul(y @ self.A, y @ self.T) + y @ self.B
                if self.ps["t_start"] < t[0] < (self.ps["t_start"] + self.ps["T"]):
                    vacc = torch.div(y @ self.V_1,
                                     y @ self.V_2)
                    return base_result + vacc
                return base_result
            return vaccinated_ode

        def basic_ode(t, y):
            return torch.mul(y @ self.A, y * self.T) + y @ self.B
        return basic_ode
