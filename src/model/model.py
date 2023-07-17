import torch
import torchode as to

from src.model.model_base import EpidemicModelBase, get_n_states
from src.dataloader import DataLoader


class VaccinatedModel(EpidemicModelBase):
    def __init__(self, model_data: DataLoader, cm: torch.Tensor):
        """
        Initializes the VaccinatedModel class.

        This method initializes the VaccinatedModel class by setting the device, defining the compartments
        with multiple substates, and calling the parent class (EpidemicModelBase) constructor. It also
        initializes the matrix generator.

        Args:
            model_data (DataLoader): DataLoader object containing model data
            cm (torch.Tensor): Contact matrix.

        Returns:
            None
        """
        self.n_state_comp = ["e", "i", "h", "ic", "icr"]
        compartments = ["s"] + self.get_n_compartments(model_data.model_parameters_data) + ["v_0", "r", "d"]
        super().__init__(model_data=model_data, compartments=compartments)
        self.s_mtx = self.n_age * self.n_comp

        from src.model.matrix_generator import MatrixGenerator
        self.matrix_generator = MatrixGenerator(model=self, cm=cm, ps=self.ps)

    def get_constant_matrices(self) -> None:
        """
        Calculates and stores the constant matrices required for evaluation of the model, using
        the matrix generator to calculate and store the constant matrices: A, T, B, and V_2.

        Returns:
            None
        """
        mtx_gen = self.matrix_generator
        self.A = mtx_gen.get_A()
        self.T = mtx_gen.get_T()
        self.B = mtx_gen.get_B().T
        self.V_2 = mtx_gen.get_V_2()

    def get_n_compartments(self, params):
        """
        Returns a list of the compartments for each class made up of multiple substates.

        This method returns a list of states for each class made up of multiple substates, based on
        the number of substates specified in the model parameters. It is used to define the compartments
        for the model.

        Args:
            params (dict): Model parameters dictionary.

        Returns:
            list: List of substates.
        """
        compartments = []
        for comp in self.n_state_comp:
            compartments.append(get_n_states(comp_name=comp, n_classes=params[f'n_{comp}']))
        return [state for n_comp in compartments for state in n_comp]

    def aggregate_by_age(self, solution, comp):
        """
        This method aggregates the solution by age for a specific compartment. It checks whether the compartment
        has substates and calls the appropriate method for aggregation.

        Args:
            solution (torch.Tensor): Model solution tensor.
            comp (str): Compartment name.

        Returns:
            torch.Tensor: Aggregated solution by age.
        """
        is_n_state = comp in self.n_state_comp
        return self._aggregate_by_age_n_state(solution, comp) if is_n_state\
            else solution[:, self.idx(comp)].sum(axis=1)

    def _aggregate_by_age_n_state(self, solution, comp):
        """
        This method aggregates the solution by age for a compartment with substates by summing the solution
        values of individual states within the compartment.

        Args:
            solution (torch.Tensor): Model solution tensor.
            comp (str): Compartment name.

        Returns:
            torch.Tensor: Aggregated solution by age.
        """
        result = 0
        for state in get_n_states(self.ps[f'n_{comp}'], comp):
            result += solution[:, self.idx(state)].sum(axis=1)
        return result

    def get_vacc_tensors(self, lhs_table):
        n_samples = lhs_table.shape[0]
        daily_vac = lhs_table * self.ps['total_vaccines'] / self.ps["T"]
        s_mtx = self.s_mtx
        V = torch.zeros((n_samples, s_mtx, s_mtx)).to(self.device)
        for idx, r in zip(range(n_samples), daily_vac):
            V[idx, :, :] = self.matrix_generator.get_V_1(r).T
        return V

    def get_solution(self, t_eval, y0, lhs_table):
        V = self.get_vacc_tensors(lhs_table)
        n_samples = y0.shape[0]
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
