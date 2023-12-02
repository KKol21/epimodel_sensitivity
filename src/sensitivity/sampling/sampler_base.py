import os
import time
from abc import ABC, abstractmethod

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
        lhs_boundaries (dict): The boundaries for Latin Hypercube Sampling (LHS) parameter ranges.

    Methods:
        run_sampling(): Runs the sampling-based simulation to explore different parameter combinations
            and collect simulation results.
    """

    def __init__(self, sim_state: dict, sim_obj):
        self.sim_obj = sim_obj
        self.sim_state = sim_state
        self.base_r0 = sim_state["base_r0"]
        self.r0generator = sim_state["r0generator"]
        self.lhs_boundaries = None
        self.target_var = None
        self.n_samples = sim_obj.n_samples
        self.batch_size = sim_obj.batch_size
        self.target = "peak"

    @abstractmethod
    def run_sampling(self):
        pass

    @abstractmethod
    def _get_variable_parameters(self):
        pass

    def _get_lhs_table(self):
        n_samples = self.n_samples
        batch_size = self.batch_size
        # Create samples
        bounds = np.array([bounds for bounds in self.lhs_boundaries.values()]).T
        sampling = LHS(xlimits=bounds)
        lhs_table = sampling(n_samples)
        print(f"\n Simulation for {n_samples} samples ({self._get_variable_parameters()})")
        print(f"Batch size: {batch_size}\n")
        return lhs_table

    def _get_sim_output(self, lhs_table):
        from src.sensitivity.target_calc.peak_calc import PeakCalculator
        # Calculate values of target variable for each sample
        PeakCalculator = PeakCalculator(self.sim_obj)
        sim_output = PeakCalculator.get_output(lhs_table=lhs_table,
                                               batch_size=self.batch_size,
                                               target_var=self.target_var)
        # Sort tables by target values
        sorted_idx = sim_output.argsort()
        sim_output = sim_output[sorted_idx]
        lhs_table = lhs_table[sorted_idx]

        time.sleep(0.3)

        # Save samples, target values
        self._save_output(output=lhs_table, output_type='lhs')
        self._save_output(output=sim_output, output_type='simulations')

    def _save_output(self, output, output_type):
        # Create directories for saving calculation outputs
        folder_name = self.sim_obj.folder_name
        os.makedirs(folder_name, exist_ok=True)

        # Save LHS output
        os.makedirs(f"{folder_name}/{output_type}", exist_ok=True)
        filename = f"{folder_name}/{output_type}/{output_type}_{self._get_variable_parameters()}"
        np.savetxt(fname=filename + ".csv", X=output, delimiter=";")


def create_latin_table(n_of_samples, lower, upper):
    bounds = np.array([lower, upper]).T
    sampling = LHS(xlimits=bounds)
    return sampling(n_of_samples)
