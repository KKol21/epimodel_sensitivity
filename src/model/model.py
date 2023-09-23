import torch
import torchode as to

from src.model.model_base import EpidemicModelBase, get_substates
from src.dataloader import DataLoader


class VaccinatedModel(EpidemicModelBase):
    def __init__(self, model_data: DataLoader, cm: torch.Tensor):
        """
        Initializes the VaccinatedModel class.

        This method initializes the VaccinatedModel class by calling the parent class (EpidemicModelBase)
        constructor, and creating the appropriate the matrix generator.

        Args:
            model_data (DataLoader): DataLoader object containing model data
            cm (torch.Tensor): Contact matrix.

        """
        super().__init__(model_data=model_data)
        self.s_mtx = self.n_age * self.n_comp

        from src.model.matrix_generator import MatrixGenerator
        self.matrix_generator = MatrixGenerator(model=self, cm=cm, ps=self.ps)

    def _get_constant_matrices(self):
        mtx_gen = self.matrix_generator
        self.A = mtx_gen.get_A()
        self.T = mtx_gen.get_T()
        self.B = mtx_gen.get_B()
        self.V_2 = mtx_gen.get_V_2()

    def get_vacc_tensors(self, lhs_table):
        n_samples = lhs_table.shape[0]
        daily_vac = lhs_table * self.ps['total_vaccines'] / self.ps["T"]
        s_mtx = self.s_mtx
        V = torch.zeros((n_samples, s_mtx, s_mtx)).to(self.device)
        for idx, r in zip(range(n_samples), daily_vac):
            V[idx, :, :] = self.matrix_generator.get_V_1(r).T
        return V

    def get_solution(self, t_eval, y0, daily_vac):
        n_samples = y0.shape[0]
        V = self.get_vacc_tensors(daily_vac)
        # from numpy import array
        # array(y @ self.A)
        # array(y @ self.T)
        # array(y @ self.B)
        # array(torch.ones(self.s_mtx) @ self.B)
        # array(torch.mul(y @ self.A, y @ self.T))
        # array(base_result)
        # base_result[0, :].sum()

        def odefun(t, y, V):
            base_result = torch.mul(y @ self.A, y @ self.T) + y @ self.B
            if self.ps["t_start"] < t[0] < (self.ps["t_start"] + self.ps["T"]):
                vacc = torch.div(torch.einsum('ij,ijk->ik', y, V),
                                 y @ self.V_2)
                return base_result + vacc
            return base_result

        term = to.ODETerm(odefun, with_args=True)
        step_method = to.Euler(term=term)
        step_size_controller = to.FixedStepController()
        solver = to.AutoDiffAdjoint(step_method, step_size_controller)
        problem = to.InitialValueProblem(y0=y0, t_eval=t_eval)
        dt0 = torch.full((n_samples,), 1)

        return solver.solve(problem, args=V, dt0=dt0)
