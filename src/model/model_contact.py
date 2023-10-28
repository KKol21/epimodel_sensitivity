import torch

from src.model.model_base import EpidemicModelBase


class ContactModel(EpidemicModelBase):
    def __init__(self, sim_obj):
        """
        Initializes the VaccinatedModel class.

        This method initializes the VaccinatedModel class by calling the parent class (EpidemicModelBase)
        constructor, and instantiating the matrix generator used in solving the model.

        Args:
            sim_obj (SimulationVaccinated): Simulation object

        """
        from src.model.matrix_generator import MatrixGenerator
        super().__init__(sim_obj=sim_obj)
        self.matrix_generator = MatrixGenerator(model=self, cm=sim_obj.cm)
        self.s_mtx = self.n_age * self.n_comp

    def initialize_constant_matrices(self):
        mtx_gen = self.matrix_generator
        self.A = mtx_gen.get_A()
        self.T = mtx_gen.get_T()
        self.B = mtx_gen.get_B()
        self.V_2 = mtx_gen.get_V_2()

    def _get_A_from_betas(self, betas):
        s_mtx = self.s_mtx
        A = torch.zeros((len(betas), s_mtx, s_mtx)).to(self.device)
        for idx, beta in enumerate(betas):
            self.matrix_generator.ps.update({"beta": beta})
            A[idx, :, :] = self.matrix_generator.get_A()
        return A

    def _get_T_from_contacts(self, contact_samples):
        T = torch.zeros((contact_samples.shape[0], self.s_mtx, self.s_mtx)).to(self.device)
        for idx, sample in enumerate(contact_samples):
            T[idx, :, :] = self.matrix_generator.get_T(sample)
        return T

    def get_solution(self, t_eval, y0, lhs_table):
        contact_samples = lhs_table[:, -1:]
        betas = lhs_table[:, -1]
        self.A = self._get_A_from_betas(betas)
        self.T = self._get_T_from_contacts(contact_samples)
        if "vaccination" in [trans["type"] for trans in self.data.trans_data.values()]:
            odefun = self.get_vaccinated_solver()
        else:
            odefun = self.get_basic_solver()
        return self.get_sol_from_solver(y0, t_eval, odefun)
