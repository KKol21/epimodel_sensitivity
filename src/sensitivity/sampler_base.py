from abc import ABC, abstractmethod
import os

import numpy as np
from smt.sampling_methods import LHS


class SamplerBase(ABC):
    """
    Base class for performing sampling-based simulations.

    This abstract class defines the common structure and interface for samplers used to explore
    parameter combinations and collect simulation results.

    Args:
        sim_state (dict): The state of the simulation as a dictionary containing relevant parameters.
        sim_obj: The simulation object representing the underlying simulation model.

    Attributes:
        sim_obj: The simulation object representing the underlying simulation model.
        sim_state (dict): The state of the simulation as a dictionary containing relevant parameters.
        base_r0 (float): The base reproduction number (R0) of the simulation.
        beta (float or None): The infection rate parameter (beta) of the simulation, if available.
        type (str or None): The type of the simulation, if available.
        r0generator: The R0 generator used in the simulation.
        lhs_boundaries (dict): The boundaries for Latin Hypercube Sampling (LHS) parameter ranges.

    Methods:
        run_sampling(): Runs the sampling-based simulation to explore different parameter combinations
            and collect simulation results.
    """

    def __init__(self, sim_state: dict, sim_obj):
        self.sim_obj = sim_obj
        self.sim_state = sim_state
        self.base_r0 = sim_state["base_r0"]
        self.beta = sim_state["beta"] if "beta" in sim_state.keys() else None
        self.type = sim_state["type"] if "type" in sim_state.keys() else None
        self.r0generator = sim_state["r0generator"]
        self.lhs_boundaries = None
        self.mode = None

    @abstractmethod
    def run_sampling(self):
        pass

    @abstractmethod
    def _get_variable_parameters(self):
        pass

    def _get_lhs_table(self, number_of_samples: int):
        # Get actual limit matrices
        lower_bound = self.lhs_boundaries[self.type]["lower"]
        upper_bound = self.lhs_boundaries[self.type]["upper"]
        # Get LHS tables
        lhs_table = create_latin_table(n_of_samples=number_of_samples,
                                       lower=lower_bound,
                                       upper=upper_bound)
        print("Simulation for", number_of_samples,
              "samples (", "-".join(self._get_variable_parameters()), ")")
        return lhs_table

    def _save_output(self, output, folder_name):
        # Create directories for saving calculation outputs
        base_name = f"../sens_data_{self.mode}"
        os.makedirs(base_name, exist_ok=True)

        # Save LHS output
        os.makedirs(f"{base_name}/{folder_name}", exist_ok=True)
        filename = f"{base_name}/{folder_name}/{folder_name}_{self._get_variable_parameters()}"
        np.savetxt(fname=filename + ".csv", X=output.cpu(), delimiter=";")


def create_latin_table(n_of_samples, lower, upper):
    bounds = np.array([lower, upper]).T
    sampling = LHS(xlimits=bounds)
    return sampling(n_of_samples)
