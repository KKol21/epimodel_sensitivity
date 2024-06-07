import torch
import torchode as to
from emsa.model.model_base import EpidemicModelBase


class EpidemicModel(EpidemicModelBase):
    def __init__(self, data, model_struct: dict):
        """
        Initialize the EpidemicModel, a class for running single simulations of a given model.

        Parameters:
            data: Model data.
            model_struct (dict): The structure of the model.
        """
        super().__init__(data, model_struct)

    def get_solution(self, y0: torch.Tensor, t_eval: torch.Tensor, **kwargs) -> to.Solution:
        """
        Get the solution of the ODE using the initial conditions and evaluation times,
        using the ODE solver in EpidemicModelBase .

        Parameters:
            y0 (torch.Tensor): Initial state.
            t_eval (torch.Tensor): Times at which to evaluate the solution.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: Solution of the ODE.
        """
        return self.get_sol_from_ode(
            y0=torch.atleast_2d(y0),
            t_eval=torch.atleast_2d(t_eval),
            odefun=self.get_ode()
        )

    def get_ode(self):
        """
        Get the ODE function based on whether the model includes vaccination.

        Returns:
            Callable: ODE function.
        """
        if self.is_vaccinated:
            v_div = torch.ones(self.n_eq).to(self.device)
            div_idx = self.idx('s_0') + self.idx('v_0')

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
                if self.ps["t_start"] < t[0] < (self.ps["t_start"] + self.ps["T"]):
                    v_div[div_idx] = (y @ self.V_2)[0, div_idx]
                    vacc = torch.div(y @ self.V_1, v_div)
                    return base_result + vacc
                return base_result

            return vaccinated_ode

        def basic_ode(t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            """
            Basic ODE function without vaccination.

            Parameters:
                t (torch.Tensor): Current time.
                y (torch.Tensor): Current state.

            Returns:
                torch.Tensor: Derivative of the system.
            """
            return torch.mul(y @ self.A, y @ self.T) + y @ self.B

        return basic_ode