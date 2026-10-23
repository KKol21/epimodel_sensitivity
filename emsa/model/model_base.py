import math
import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, Callable

import torch
import torchode as to


class EpidemicModelBase(ABC):
    def __init__(self, data, model_struct: Dict[str, Any], solver_config=None):
        """
        Initialises Abstract base class for epidemic models.

        This class provides the base functionality for epidemic models. It contains methods to initialise the model,
        retrieve initial values, and obtain the model solution

        Args:
            data (Any): Data for the epidemic model.
            model_struct (Dict[str, Any]): Structure of the epidemic model.

        Returns:
            None
        """
        self.data = data
        self.model_struct = model_struct
        self.state_data = model_struct["state_data"]
        self.trans_data = model_struct["trans_data"]
        self.tms_rules = model_struct["tms_rules"]

        self.population = data.age_data.flatten()
        self.n_age = len(self.population)
        self.compartments = self.get_compartments()
        self.n_comp = len(self.compartments)
        self.ps = data.params
        self.validate_params()
        self.device = data.device

        self.c_idx = {comp: idx for idx, comp in enumerate(self.compartments)}
        self.n_eq = self.n_age * self.n_comp

        from emsa.model.matrix_generator import MatrixGenerator

        self.matrix_generator = MatrixGenerator(model=self, cm=data.cm)

        self.A = None
        self.T = None
        self.B = None
        self.solver_config = dict(solver_config or {})

    def validate_params(self):
        for param, value in self.ps.items():
            msg = f"Parameter {param} was given a non-positive value - are you sure you meant to do this?"
            if torch.is_tensor(value):
                if (value <= 0).any():
                    warnings.warn(msg)
            elif isinstance(value, (int, float)):
                if value <= 0:
                    warnings.warn(msg)

    def initialize_matrices(self):
        """
        Initialize the matrices used in the model.
        """
        mtx_gen = self.matrix_generator
        self.A = mtx_gen.get_A()
        self.T = mtx_gen.get_T()
        self.B = mtx_gen.get_B()

    def visualize_transmission_graph(self):
        from emsa.utils.plotter import visualize_transmission_graph

        visualize_transmission_graph(
            state_data=self.state_data,
            trans_data=self.trans_data,
            tms_rules=self.tms_rules,
        )

    @abstractmethod
    def get_solution(self, y0, t_eval, **kwargs):
        """
        Get the solution of the epidemic model.

        Args:
            y0 (torch.Tensor): Initial values.
            t_eval (torch.Tensor): Evaluation times.
            **kwargs: Additional keyword arguments.

        Returns:
            Any: Solution of the model.
        """
        pass

    def get_sol_from_ode(
        self,
        y0: torch.Tensor,
        t_eval: torch.Tensor,
        odefun: Callable,
        solver_config=None,
    ) -> to.Solution:
        """
        Solve with configurable Euler or adaptive Dopri5 and validate the result.

        Args:
            y0 (torch.Tensor): Initial values.
            t_eval (torch.Tensor): Evaluation times.
            odefun (Any): ODE function.

        Returns:
            Any: Solution of the ODE system.
        """
        config = {**self.solver_config, **(solver_config or {})}
        allowed = {
            "method",
            "step_size",
            "rtol",
            "atol",
            "max_steps",
            "negative_tolerance",
        }
        if set(config) - allowed:
            raise ValueError(f"Unknown solver settings: {set(config) - allowed}")
        method = config.get("method", "euler").lower()
        step_size = config.get("step_size", 1.0)
        rtol, atol = config.get("rtol", 1e-5), config.get("atol", 1e-6)
        negative_tolerance = config.get("negative_tolerance", 1e-4)
        if any(not math.isfinite(v) or v <= 0 for v in (step_size, rtol, atol)):
            raise ValueError("Solver step size and tolerances must be finite and positive")
        if not math.isfinite(negative_tolerance) or negative_tolerance < 0:
            raise ValueError("negative_tolerance must be finite and nonnegative")
        max_steps = config.get("max_steps", 100000)
        if not isinstance(max_steps, int) or max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        y0 = torch.atleast_2d(y0).to(self.device)
        t_eval = torch.atleast_2d(t_eval).to(device=self.device, dtype=y0.dtype)
        if t_eval.shape[0] == 1 and y0.shape[0] > 1:
            t_eval = t_eval.expand(y0.shape[0], -1)
        if (
            y0.ndim != 2
            or y0.shape[1] != self.n_eq
            or t_eval.ndim != 2
            or t_eval.shape[0] != y0.shape[0]
            or t_eval.shape[1] < 2
        ):
            raise ValueError("Expected y0 [batch, equations] and t_eval [batch, times >= 2]")
        if not torch.isfinite(y0).all() or (y0 < -negative_tolerance).any():
            raise ValueError("Initial populations must be finite and nonnegative")
        if not torch.isfinite(t_eval).all() or (t_eval.diff(dim=1) <= 0).any():
            raise ValueError("Evaluation times must be finite and strictly increasing")
        term = to.ODETerm(odefun)
        if method == "euler":
            step_method = to.Euler(term=term)
            controller = to.FixedStepController()
        elif method == "dopri5":
            step_method = to.Dopri5(term=term)
            controller = to.IntegralController(atol=atol, rtol=rtol, dt_max=step_size)
        else:
            raise ValueError("Solver method must be 'euler' or 'dopri5'")
        solver = to.AutoDiffAdjoint(step_method, controller, max_steps=max_steps)
        problem = to.InitialValueProblem(y0=y0, t_eval=t_eval)
        dt0 = y0.new_full((y0.shape[0],), step_size)
        solution = solver.solve(problem, dt0=dt0)
        if (solution.status != to.Status.SUCCESS.value).any():
            raise RuntimeError(f"ODE solver failed with status codes {solution.status.tolist()}")
        if not torch.isfinite(solution.ys).all():
            raise RuntimeError("ODE solver returned non-finite populations")
        if (solution.ys < -negative_tolerance).any():
            raise RuntimeError(
                "ODE solver returned negative populations; reduce step_size or use dopri5"
            )
        return solution

    def get_compartments(self) -> list:
        """
        Get the list of compartments.

        Returns:
            List[str]: List of compartments.
        """
        compartments = []
        for name, data in self.state_data.items():
            compartments += get_substates(data.get("n_substates", 1), name)
        return compartments

    def get_initial_values_from_dict(self, init_val_dict: dict) -> torch.FloatTensor:
        """

        Retrieve the initial values for the model based on
        the provided values in the corresponding configuration json.

        Returns:
            torch.Tensor: Initial values of the model.

        """
        iv = torch.zeros(self.n_eq).to(self.device)
        susc_state = [
            state for state, data in self.state_data.items() if data.get("type") == "susceptible"
        ][0] + "_0"
        iv[self.idx(susc_state)] = self.population
        for comp, comp_iv in init_val_dict.items():
            comp_iv = torch.as_tensor(comp_iv, dtype=torch.float32, device=self.device)
            iv[self.idx(f"{comp}_0")] = comp_iv
            iv[self.idx(susc_state)] -= comp_iv
        if not torch.isfinite(iv).all() or (iv < 0).any():
            raise ValueError(
                "Initial compartment populations must be finite, nonnegative and fit the population"
            )
        return iv

    def idx(self, state: str) -> torch.BoolTensor:
        """
        Get the index tensor for a given state.

        Args:
            state (str): State name.

        Returns:
            torch.BoolTensor: Index tensor.
        """
        return torch.arange(self.n_eq) % self.n_comp == self.c_idx[state]

    def aggregate_by_age(self, solution, comp):
        """
        Aggregate the solution by age for a compartment.

        Args:
            solution (torch.Tensor): Solution tensor.
            comp (str): Compartment name.

        Returns:
            torch.Tensor: Aggregated solution.
        """
        substates = get_substates(
            n_substates=self.state_data[comp].get("n_substates", 1), comp_name=comp
        )
        return torch.stack(
            tensors=[solution[:, self.idx(state)].sum(dim=1) for state in substates],
            dim=0,
        ).sum(dim=0)

    def get_beta_from_r0(self, base_r0, susceptibles):
        from emsa.model import R0Generator

        r0generator = R0Generator(self.data, self.model_struct)
        if isinstance(base_r0, tuple):
            base_r0 = base_r0[0]
        return base_r0 / r0generator.get_eig_val(
            contact_mtx=self.data.cm,
            susceptibles=susceptibles.flatten(),
            population=self.population,
        )


def get_substates(n_substates, comp_name):
    """
    Get the list of substates for a compartment.

    Args:
        n_substates (int): Number of substates.
        comp_name (str): Compartment name.

    Returns:
        List[str]: List of state names.
    """
    return [f"{comp_name}_{i}" for i in range(n_substates)]
