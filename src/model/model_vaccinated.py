import torch

from src.model.model_base import EpidemicModelBase


class VaccinatedModel(EpidemicModelBase):
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

    def get_solution(self, t_eval, y0, lhs_table):
        self.V_1 = self._get_V_1_from_lhs(lhs_table=torch.from_numpy(lhs_table).float())
        odefun = self.get_vaccinated_solver(curr_batch_size=lhs_table.shape[0])
        return self.get_sol_from_solver(y0, t_eval, odefun)

    def _get_V_1_from_lhs(self, lhs_table):
        daily_vacc = lhs_table * self.ps['total_vaccines'] / self.ps["T"]
        s_mtx = self.s_mtx
        V_1 = torch.zeros((daily_vacc.shape[0], s_mtx, s_mtx)).to(self.device)
        for idx, sample in enumerate(daily_vacc):
            V_1[idx, :, :] = self.matrix_generator.get_V_1(daily_vac=sample)
        return V_1
