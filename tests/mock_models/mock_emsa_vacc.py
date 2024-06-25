import torch

from emsa.model.epidemic_model import EpidemicModel


class EMSAMockVaccinatedModel(EpidemicModel):
    def __init__(self, model_struct, data):
        super().__init__(model_struct=model_struct, data=data)

        mgen = self.matrix_generator
        self.V_1 = mgen.get_V_1()
        self.V_2 = mgen.get_V_2()

    def get_vacc_ode(self):
        v_div = torch.ones(self.n_eq).to(self.device)
        div_idx = self.idx("s_0") + self.idx("v_0")

        def vaccinated_ode(t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            """
            ODE function for the vaccinated scenario.

            Parameters:
                t (torch.Tensor): Current time.
                y (torch.Tensor): Current state.

            Returns:
                torch.Tensor: Derivative of the system.
            """
            base_result = torch.mul(y @ self.A, y @ self.T) + y @ self.B
            if self.ps["t_start"] <= t[0] < (self.ps["t_start"] + self.ps["T"]):
                v_div[div_idx] = (y @ self.V_2)[0, div_idx]
                vacc = torch.div(y @ self.V_1, v_div)
                return base_result + vacc
            return base_result

        return vaccinated_ode
