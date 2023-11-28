import numpy as np
import torch

from src.model.model_base import EpidemicModelBase


class ContactModel(EpidemicModelBase):
    def __init__(self, sim_obj):
        """
        Initializes the VaccinatedModel class.

        This method initializes the ContactModel class by calling the parent class (EpidemicModelBase)
        constructor, and instantiating the matrix generator used in solving the model.

        Args:
            sim_obj (SimulationContact): Simulation object

        """
        from src.model.matrix_generator import MatrixGenerator
        super().__init__(sim_obj=sim_obj)
        self.matrix_generator = MatrixGenerator(model=self, cm=sim_obj.cm)
        self.s_mtx = self.n_age * self.n_comp
        self.upper_tri_size = sim_obj.upper_tri_size

    def initialize_constant_matrices(self):
        mtx_gen = self.matrix_generator
        self.A = mtx_gen.get_A()
        self.T = mtx_gen.get_T()
        self.B = mtx_gen.get_B()
        if self.is_vaccinated:
            self.V_1 = mtx_gen.get_V_1()
            self.V_2 = mtx_gen.get_V_2()

    def get_solution(self, t_eval, y0, lhs_table):
        cm_samples = self.get_contacts_from_lhs(lhs_table)
        betas = self._get_betas_from_contacts(cm_samples)
        self.A = self._get_A_from_betas(betas)
        self.T = self._get_T_from_contacts(cm_samples)
        if self.is_vaccinated:
            odefun = self.get_vaccinated_solver(lhs_table.shape[0])
        else:
            odefun = self.get_basic_solver()
        return self.get_sol_from_solver(y0, t_eval, odefun)

    def _get_A_from_betas(self, betas):
        s_mtx = self.s_mtx
        A = torch.zeros((len(betas), s_mtx, s_mtx)).to(self.device)
        for idx, beta in enumerate(betas):
            self.matrix_generator.ps.update({"beta": beta})
            A[idx, :, :] = self.matrix_generator.get_A()
        return A

    def _get_T_from_contacts(self, cm_samples: torch.Tensor):
        T = torch.zeros((cm_samples.size(0), self.s_mtx, self.s_mtx)).to(self.device)
        for idx, cm in enumerate(cm_samples):
            T[idx, :, :] = self.matrix_generator.get_T(cm)
        return T

    def _get_betas_from_contacts(self, cm_samples):
        base_r0 = self.sim_state["base_r0"]
        r0gen = self.sim_state["r0generator"]
        betas = [base_r0 / r0gen.get_eig_val(contact_mtx=cm,
                                             susceptibles=self.sim_obj.susceptibles.flatten(),
                                             population=self.sim_obj.population)
                 for cm in cm_samples]
        return betas

    def get_contacts_from_lhs(self, lhs_table):
        contact_sim = torch.zeros((lhs_table.shape[0], self.sim_obj.n_age, self.sim_obj.n_age))
        for idx, sample in enumerate(lhs_table):
            contact_sim[idx, :, :] = get_contact_matrix_from_upper_triu(rvector=sample,
                                                                        age_vector=self.sim_obj.population.flatten())
        return contact_sim


def get_contact_matrix_from_upper_triu(rvector, age_vector):
    new_2 = get_rectangular_matrix_from_upper_triu(rvector=rvector,
                                                   matrix_size=age_vector.size(0))
    vector = new_2 / age_vector
    return vector


def get_rectangular_matrix_from_upper_triu(rvector, matrix_size):
    upper_tri_indexes = np.triu_indices(matrix_size)
    new_contact_mtx = np.zeros((matrix_size, matrix_size))
    new_contact_mtx[upper_tri_indexes] = rvector
    new_2 = new_contact_mtx.T
    new_2[upper_tri_indexes] = rvector
    return torch.from_numpy(new_2)
