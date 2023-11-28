from abc import ABC, abstractmethod

import torch
import torchode as to


class EpidemicModelBase(ABC):
    def __init__(self, sim_obj):
        """
        Initialises Abstract base class for epidemic models.

        This class provides the base functionality for epidemic models. It contains methods to initialize the model,
        retrieve initial values, and obtain the model solutution

        Returns:
            None
        """
        self.sim_obj = sim_obj
        self.sim_state = None
        self.data = sim_obj.data
        self.ps = sim_obj.params
        self.population = sim_obj.population
        self.n_age = sim_obj.n_age
        self.device = sim_obj.device
        self.test = sim_obj.test

        self.compartments = self.get_compartments()
        self.n_comp = len(self.compartments)

        self.c_idx = {comp: idx for idx, comp in enumerate(self.compartments)}
        self.s_mtx = self.n_age * self.n_comp
        self.is_vaccinated = "vaccination" in [trans["type"] for trans in self.data.trans_data.values()]

        self.A = None
        self.T = None
        self.B = None
        self.V_1 = None
        self.V_2 = None

    @abstractmethod
    def initialize_constant_matrices(self):
        pass

    @abstractmethod
    def get_solution(self, t_eval, y0, lhs_table):
        pass

    @staticmethod
    def get_sol_from_solver(y0, t_eval, odefun):
        term = to.ODETerm(odefun)
        step_method = to.Euler(term=term)
        step_size_controller = to.FixedStepController()
        solver = to.AutoDiffAdjoint(step_method, step_size_controller)
        problem = to.InitialValueProblem(y0=y0, t_eval=t_eval)
        dt0 = torch.full((y0.shape[0],), 1)

        return solver.solve(problem, dt0=dt0)

    def get_basic_solver(self):
        A_mul = self.get_mul_method(self.A)
        T_mul = self.get_mul_method(self.T)
        B_mul = self.get_mul_method(self.B)

        def odefun(t, y):
            return torch.mul(A_mul(y, self.A), T_mul(y, self.T)) + B_mul(y, self.B)

        return odefun

    def get_vaccinated_solver(self, curr_batch_size):
        A_mul = self.get_mul_method(self.A)
        T_mul = self.get_mul_method(self.T)
        B_mul = self.get_mul_method(self.B)
        V_1_mul = self.get_mul_method(self.V_1)

        v_div = torch.ones((curr_batch_size, self.s_mtx)).to(self.device)
        div_idx = self.idx('s_0') + self.idx('v_0')

        def odefun(t, y):
            base_result = torch.mul(A_mul(y, self.A), T_mul(y, self.T)) + B_mul(y, self.B)
            if self.ps["t_start"] < t[0] < self.ps["t_start"] + self.ps["T"]:
                v_div[:, div_idx] = (y @ self.V_2)[:, div_idx]
                vacc = torch.div(V_1_mul(y, self.V_1),
                                 v_div)
                return base_result + vacc
            return base_result

        return odefun

    @staticmethod
    def get_mul_method(tensor: torch.Tensor):
        def mul_by_2d(y, tensor):
            return y @ tensor

        def mul_by_3d(y, tensor):
            return torch.einsum('ij,ijk->ik', y, tensor)

        return mul_by_2d if len(tensor.size()) < 3 else mul_by_3d

    def get_compartments(self):
        compartments = []
        for name, data in self.data.state_data.items():
            compartments += get_substates(data["n_substates"], name)
        return compartments

    def get_initial_values(self):
        """

        This method retrieves the initial values for the model. It sets the initial value for the infected compartment
        of the 3rd age group (i_0^3) to 1 and subtracts 1 from the susceptible (s_0^3) compartment for the appropriate
        age group.

        Returns:
            torch.Tensor: Initial values of the model.

        """
        iv = torch.zeros(self.s_mtx).to(self.device)
        age_group = 3
        iv[age_group + self.c_idx['i_0']] = 1
        iv[self.idx('s_0')] = self.population
        iv[age_group * self.n_comp + self.c_idx['s_0']] -= 1
        return iv

    def idx(self, state: str) -> bool:
        return torch.arange(self.s_mtx) % self.n_comp == self.c_idx[state]

    def aggregate_by_age(self, solution, comp):
        """

        This method aggregates the solution by age for a compartment by summing the solution
        values of individual substates for each age group.

        Args:
            solution (torch.Tensor): Model solution tensor.
            comp (str): Compartment name.

        Returns:
            torch.Tensor: Aggregated solution by age.

        """
        result = 0
        for state in get_substates(self.data.state_data[comp]["n_substates"], comp):
            result += solution[:, self.idx(state)].sum(axis=1)
        return result


def get_substates(n_substates, comp_name):
    """
   Args:
       n_substates (int): Number of substates
       comp_name (str): Compartment name.

   Returns:
       list: List of state names.

   """
    return [f"{comp_name}_{i}" for i in range(n_substates)]
