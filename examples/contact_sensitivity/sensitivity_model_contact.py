import numpy as np
import torch

from src.model.r0 import R0Generator
from src.sensitivity.sensitivity_model_base import SensitivityModelBase


class ContactModel(SensitivityModelBase):
    def __init__(self, sim_obj, base_r0):
        """
        Initializes the VaccinatedModel class.

        This method initializes the ContactModel class by calling the parent class (EpidemicModelBase)
        constructor, and instantiating the matrix generator used in solving the model.

        Args:
            sim_obj (SimulationContact): Simulation object

        """
        super().__init__(sim_obj=sim_obj)

        self.base_r0 = base_r0
        self.s_mtx = self.n_age * self.n_comp
        self.upper_tri_size = sim_obj.upper_tri_size

    def get_solution(self, y0, t_eval, **kwargs):
        lhs_table = kwargs["lhs_table"]
        cm_samples = self.get_contacts_from_lhs(lhs_table)
        betas = self._get_betas_from_contacts(cm_samples)
        self.A = self._get_A_from_betas(betas)
        self.T = self._get_T_from_contacts(cm_samples)
        if self.is_vaccinated:
            odefun = self.get_vaccinated_ode(lhs_table.shape[0])
        else:
            odefun = self.get_basic_ode()
        return self.get_sol_from_ode(y0, t_eval, odefun)

    def _get_A_from_betas(self, betas):
        lhs_dict = {"beta": betas}
        return self.get_matrix_from_lhs(lhs_dict=lhs_dict, matrix_name="A")

    def _get_T_from_contacts(self, cm_samples: torch.Tensor):
        T = torch.zeros((cm_samples.size(0), self.s_mtx, self.s_mtx)).to(self.device)
        for idx, cm in enumerate(cm_samples):
            T[idx, :, :] = self.matrix_generator.get_T(cm)
        return T

    def _get_betas_from_contacts(self, cm_samples):
        r0gen = R0Generator(self.data, **self.sim_obj.model_struct)
        betas = [self.base_r0 / r0gen.get_eig_val(contact_mtx=cm,
                                                  susceptibles=self.sim_obj.susceptibles.flatten(),
                                                  population=self.sim_obj.population)
                 for cm in cm_samples]
        return torch.tensor(betas, device=self.device)

    def get_contacts_from_lhs(self, lhs_table):
        contact_sim = torch.zeros((lhs_table.shape[0], self.sim_obj.n_age, self.sim_obj.n_age))
        for idx, sample in enumerate(lhs_table):
            contact_sim[idx, :, :] = get_contact_matrix_from_upper_triu(rvector=sample,
                                                                        age_vector=self.sim_obj.population.flatten())
        return contact_sim


def get_contact_matrix_from_upper_triu(rvector, age_vector):
    new = get_rectangular_matrix_from_upper_triu(rvector=rvector, matrix_size=age_vector.size(0)) / age_vector
    return new


def get_rectangular_matrix_from_upper_triu(rvector, matrix_size):
    upper_tri_indexes = np.triu_indices(matrix_size)
    new_contact_mtx = np.zeros((matrix_size, matrix_size))
    new_contact_mtx[upper_tri_indexes] = rvector
    new_2 = new_contact_mtx.T
    new_2[upper_tri_indexes] = rvector
    return torch.from_numpy(new_2)
