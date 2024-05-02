from abc import ABC, abstractmethod

import torch
import torchode as to


class EpidemicModelBase(ABC):
    def __init__(self, data, model_struct):
        """
        Initialises Abstract base class for epidemic models.

        This class provides the base functionality for epidemic models. It contains methods to initialize the model,
        retrieve initial values, and obtain the model solutution

        Returns:
            None
        """
        self.data = data
        self.state_data = model_struct["state_data"]
        self.trans_data = model_struct["trans_data"]
        self.tms_rules = model_struct["tms_rules"]

        self.population = data.age_data.flatten()
        self.n_age = len(self.population)
        self.compartments = self.get_compartments()
        self.n_comp = len(self.compartments)
        self.ps = data.params
        self.device = data.device

        self.c_idx = {comp: idx for idx, comp in enumerate(self.compartments)}
        self.n_eq = self.n_age * self.n_comp
        self.is_vaccinated = "vaccination" in [trans.get("type") for trans in self.trans_data]

        from src.model.matrix_generator import MatrixGenerator
        self.matrix_generator = MatrixGenerator(model=self, cm=data.cm)

        self.A = None
        self.T = None
        self.B = None
        self.V_1 = None
        self.V_2 = None

    def initialize_matrices(self):
        mtx_gen = self.matrix_generator
        self.A = mtx_gen.get_A()
        self.T = mtx_gen.get_T()
        self.B = mtx_gen.get_B()
        if self.is_vaccinated:
            self.V_1 = mtx_gen.get_V_1()
            self.V_2 = mtx_gen.get_V_2()

    def visualize_transmission_graph(self):
        from src.plotter import visualize_transmission_graph
        visualize_transmission_graph(state_data=self.state_data,
                                     trans_data=self.trans_data,
                                     tms_rules=self.tms_rules)

    @abstractmethod
    def get_solution(self, y0, t_eval, **kwargs):
        pass

    def get_sol_from_ode(self, y0, t_eval, odefun):
        term = to.ODETerm(odefun)
        step_method = to.Euler(term=term)
        step_size_controller = to.FixedStepController()
        solver = to.AutoDiffAdjoint(step_method, step_size_controller)
        problem = to.InitialValueProblem(y0=y0, t_eval=t_eval)
        dt0 = torch.full((y0.shape[0],), 1).to(self.device)

        return solver.solve(problem, dt0=dt0)

    def get_compartments(self) -> list:
        compartments = []
        for name, data in self.state_data.items():
            compartments += get_substates(data.get("n_substates", 1), name)
        return compartments

    def get_initial_values_from_dict(self, init_val_dict: dict) -> torch.FloatTensor:
        """

        This method retrieves the initial values for the model based
        on the provided values in the corresponding configuration json.

        Returns:
            torch.Tensor: Initial values of the model.

        """
        iv = torch.zeros(self.n_eq).to(self.device)
        susc_state = [state for state, data in self.state_data.items()
                      if data.get("type") == "susceptible"][0] + "_0"
        iv[self.idx(susc_state)] = self.population
        for comp, comp_iv in init_val_dict.items():
            comp_iv = torch.FloatTensor(comp_iv, device=self.device)
            iv[self.idx(f"{comp}_0")] = comp_iv
            iv[self.idx(susc_state)] -= comp_iv
        return iv

    def idx(self, state: str) -> torch.BoolTensor:
        return torch.arange(self.n_eq) % self.n_comp == self.c_idx[state]

    def aggregate_by_age(self, solution, comp):
        """

        This method aggregates the solution by age for a compartment by summing the solution
        values of individual substates for each age group.

        """
        substates = get_substates(n_substates=self.state_data[comp].get("n_substates", 1), comp_name=comp)
        return torch.stack(
            tensors=[solution[:, self.idx(state)].sum(dim=1) for state in substates],
            dim=0).sum(dim=0)


def get_substates(n_substates, comp_name):
    """
   Args:
       n_substates (int): Number of substates
       comp_name (str): Compartment name.

   Returns:
       list: List of state names.

   """
    return [f"{comp_name}_{i}" for i in range(n_substates)]
