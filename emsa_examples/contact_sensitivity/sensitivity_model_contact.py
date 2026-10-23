import numpy as np
import torch

from emsa.model import R0Generator
from emsa.sensitivity import SensitivityModelBase


class ContactModel(SensitivityModelBase):
    def __init__(self, sim_object, base_r0):
        """
        Initializes the ContactModel class.

        This method initializes the ContactModel class by calling the parent class (EpidemicModelBase)
        constructor, and instantiating the matrix generator used in solving the model.

        Args:
            sim_object (SimulationContact): Simulation object

        """
        super().__init__(sim_object=sim_object)

        self.base_r0 = base_r0
        self.s_mtx = self.n_age * self.n_comp
        self.upper_tri_size = sim_object.upper_tri_size

    def get_solution(self, y0, t_eval, **kwargs):
        lhs_table = kwargs["lhs_table"]
        cm_samples = self.get_contacts_from_lhs(lhs_table=lhs_table)
        betas = self._get_betas_from_contacts(cm_samples=cm_samples)
        self.T = self._get_T_from_contacts(cm_samples=cm_samples, betas=betas)
        odefun = self.get_basic_ode()
        return self.get_sol_from_ode(y0, t_eval, odefun, solver_config=kwargs.get("solver_config"))

    def _get_T_from_contacts(self, cm_samples: torch.Tensor, betas: torch.Tensor):
        T = torch.zeros((cm_samples.size(0), self.s_mtx, self.s_mtx)).to(self.device)
        ps_original = self.matrix_generator.ps
        try:
            for idx, (cm, beta) in enumerate(zip(cm_samples, betas)):
                self.matrix_generator.ps = {**ps_original, "beta": beta}
                T[idx, :, :] = self.matrix_generator.get_T(cm=cm)
        finally:
            self.matrix_generator.ps = ps_original
        return T

    def _get_betas_from_contacts(self, cm_samples: torch.Tensor):
        r0gen = R0Generator(data=self.data, model_struct=self.model_struct)
        betas = [
            self.base_r0
            / r0gen.get_eig_val(
                contact_mtx=cm,
                susceptibles=self.sim_object.susceptibles.flatten(),
                population=self.sim_object.population,
            )
            for cm in cm_samples
        ]
        return torch.tensor(betas, device=self.device)

    def get_contacts_from_lhs(self, lhs_table: np.ndarray):
        contact_sim = torch.zeros(
            (lhs_table.shape[0], self.sim_object.n_age, self.sim_object.n_age),
            device=self.device,
        )
        cm_home = self.data.contact_matrices["home"]
        cm_diff = self.data.cm - cm_home
        for idx, sample in enumerate(lhs_table):
            ratio_mtx = get_ratio_matrix_from_upper_triu(
                rvector=sample, age_vector=self.sim_object.population.flatten()
            )
            contact_sim[idx, :, :] = cm_home + (1 - ratio_mtx) * cm_diff
        return contact_sim


def get_ratio_matrix_from_upper_triu(rvector, age_vector):
    return get_rectangular_matrix_from_upper_triu(rvector=rvector, matrix_size=age_vector.size(0))


def get_rectangular_matrix_from_upper_triu(rvector, matrix_size):
    values = torch.as_tensor(rvector)
    rows, cols = torch.triu_indices(matrix_size, matrix_size, device=values.device)
    matrix = values.new_zeros((matrix_size, matrix_size))
    matrix[rows, cols] = values
    matrix[cols, rows] = values
    return matrix
