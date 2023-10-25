import torch
import torchode as to

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

    def get_vacc_tensors(self, daily_vac):
        n_samples = daily_vac.shape[0]
        s_mtx = self.s_mtx
        V = torch.zeros((n_samples, s_mtx, s_mtx)).to(self.device)
        for idx, sample in zip(range(n_samples), daily_vac):
            V[idx, :, :] = self.matrix_generator.get_V_1(sample)
        return V

    def get_solution(self, t_eval, y0, lhs_table):
        n_samples = y0.shape[0]
        v_div = torch.ones((n_samples, self.s_mtx)).to(self.device)
        div_idx = self.idx('s_0') + self.idx('v_0')

        daily_vac = lhs_table * self.ps['total_vaccines'] / self.ps["T"]
        V = self.get_vacc_tensors(daily_vac)

        def odefun(t, y, V):
            base_result = torch.mul(y @ self.A, y @ self.T) + y @ self.B
            if self.ps["t_start"] < t[0] < self.ps["t_start"] + self.ps["T"]:
                v_div[:, div_idx] = (y @ self.V_2)[:, div_idx]
                vacc = torch.div(torch.einsum('ij,ijk->ik', y, V),
                                 v_div)
                return base_result + vacc
            return base_result

        term = to.ODETerm(odefun, with_args=True)
        step_method = to.Euler(term=term)
        step_size_controller = to.FixedStepController()
        solver = to.AutoDiffAdjoint(step_method, step_size_controller)
        problem = to.InitialValueProblem(y0=y0, t_eval=t_eval)
        dt0 = torch.full((n_samples,), 1)

        return solver.solve(problem, args=V, dt0=dt0)
