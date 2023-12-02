import torch

from src.sensitivity.target_calc.sensitivity_model_base import SensitivityModelBase


class VaccinatedModel(SensitivityModelBase):
    def __init__(self, sim_obj):
        """
        Initializes the VaccinatedModel class.

        This method initializes the VaccinatedModel class by calling the parent class (EpidemicModelBase)
        constructor, and instantiating the matrix generator used in solving the model.

        Args:
            sim_obj (SimulationVaccinated): Simulation object

        """
        super().__init__(sim_obj=sim_obj)
        self.s_mtx = self.n_age * self.n_comp

    def get_solution(self, y0, t_eval, **kwargs):
        lhs_table = kwargs["lhs_table"]
        self.V_1 = self._get_V_1_from_lhs(lhs_table=torch.from_numpy(lhs_table).float())
        odefun = self.get_vaccinated_ode(curr_batch_size=lhs_table.shape[0])
        return self.get_sol_from_ode(y0, t_eval, odefun)

    def _get_V_1_from_lhs(self, lhs_table):
        daily_vacc = lhs_table * self.ps['total_vaccines'] / self.ps["T"]
        s_mtx = self.s_mtx
        V_1 = torch.zeros((daily_vacc.shape[0], s_mtx, s_mtx)).to(self.device)
        for idx, sample in enumerate(daily_vacc):
            V_1[idx, :, :] = self.matrix_generator.get_V_1(daily_vac=sample)
        return V_1
