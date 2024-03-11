from src.sensitivity.sensitivity_model_base import SensitivityModelBase


class SEIRModel(SensitivityModelBase):
    def __init__(self, sim_obj):
        """
        Initializes the VaccinatedModel class.

        This method initializes the ContactModel class by calling the parent class (EpidemicModelBase)
        constructor, and instantiating the matrix generator used in solving the model.

        Args:
            sim_obj (SimulationContact): Simulation object

        """
        super().__init__(sim_obj=sim_obj)

    def get_solution(self, y0, t_eval, **kwargs):
        odefun = self.get_basic_ode()
        return self.get_sol_from_ode(y0, t_eval, odefun)
