import torch

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
        lhs_table = kwargs["lhs_table"]
        #self.A = self._get_A_from_betas(lhs_table)
        #self.B = self._get_B_from_lhs(lhs_table)
        if self.is_vaccinated:
            odefun = self.get_vaccinated_ode(lhs_table.shape[0])
        else:
            odefun = self.get_basic_ode()
        return self.get_sol_from_ode(y0, t_eval, odefun)

    def _get_A_from_betas_(self, betas):
        s_mtx = self.n_eq
        A = torch.zeros((len(betas), s_mtx, s_mtx)).to(self.device)
        for idx, beta in enumerate(betas):
            self.matrix_generator.ps.update({"beta": beta})
            A[idx, :, :] = self.matrix_generator.get_A()
        return A

    def _get_A_from_betas(self, lhs):
        lhs_dict = {"beta": lhs[:, 1]}
        return self.get_matrix_from_lhs(lhs_dict=lhs_dict, matrix_name="A")

    def _get_B_from_lhs(self, lhs):
        lhs_dict = {"alpha": lhs[:, 0], "gamma": lhs[:, 2]}
        return self.get_matrix_from_lhs(lhs_dict=lhs_dict, matrix_name="B")
