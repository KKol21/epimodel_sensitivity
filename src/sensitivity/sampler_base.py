import os
import time
from abc import ABC, abstractmethod

import numpy as np
import torch
from smt.sampling_methods import LHS


class SamplerBase(ABC):
    """
    Base class for performing sampling-based simulations.

    This abstract class defines the common structure and interface for samplers used to explore
    parameter combinations and collect simulation results.

    Args:
        sim_state (dict): The state of the simulation as a dictionary containing relevant params.
        sim_obj: The simulation object representing the underlying simulation model.

    Attributes:
        sim_obj: The simulation object representing the underlying simulation model.
        sim_state (dict): The state of the simulation as a dictionary containing relevant params.
        base_r0 (float): The base reproduction number (R0) of the simulation.
        lhs_boundaries (dict): The boundaries for Latin Hypercube Sampling (LHS) parameter ranges.

    Methods:
        run_sampling(): Runs the sampling-based simulation to explore different parameter combinations
            and collect simulation results.
    """

    def __init__(self, sim_state: dict, sim_obj):
        self.sim_obj = sim_obj
        self.sim_state = sim_state
        self.lhs_boundaries = None
        self.target_var = None
        self.n_samples = sim_obj.n_samples
        self.batch_size = sim_obj.batch_size

    @abstractmethod
    def run_sampling(self):
        pass

    @abstractmethod
    def _get_variable_parameters(self):
        pass

    def _get_lhs_table(self):
        n_samples = self.n_samples
        batch_size = self.batch_size

        bounds = np.array([bounds for bounds in self.lhs_boundaries.values()]).T
        sampling = LHS(xlimits=bounds)
        lhs_table = sampling(n_samples)

        print(f"\n Simulation for {n_samples} samples ({self._get_variable_parameters()})")
        print(f"Batch size: {batch_size}\n")
        return lhs_table

    def _get_sim_output(self, lhs_table):
        sim_output = self.calculate_target(lhs_table=lhs_table)
        # Sort tables by target values
        sorted_idx = sim_output.argsort()
        sim_output = sim_output[sorted_idx]
        lhs_table = lhs_table[sorted_idx.cpu()]

        time.sleep(0.3)

        # Save samples, target values
        self._save_output(output=lhs_table, output_name='lhs')
        self._save_output(output=sim_output.cpu(), output_name='simulations')

    def calculate_target(self, lhs_table):
        #  from src.sensitivity.target_calc.r0_calc import R0Calculator

        #  if self.target_var == "r0":
        #      r0_calculator = R0Calculator(self.sim_obj.model)
        #      return r0_calculator.calculate_R0s(lhs_table=lhs_table)

        from src.sensitivity.target_calc.peak_calc import PeakCalculator
        from src.sensitivity.target_calc.final_size_calc import FinalSizeCalculator

        if self.target_var in ["d_max", "r_max"]:
            target_calculator = FinalSizeCalculator(self.sim_obj.model)
        else:
            target_calculator = PeakCalculator(self.sim_obj.model)
        lhs = torch.from_numpy(lhs_table).float().to(self.sim_obj.device)
        return target_calculator.get_output(lhs_table=lhs,
                                            batch_size=self.batch_size,
                                            target_var=self.target_var)

    def _save_output(self, output, output_name):
        # Create directories for saving calculation outputs
        folder_name = self.sim_obj.folder_name
        os.makedirs(folder_name, exist_ok=True)

        # Save LHS output
        os.makedirs(f"{folder_name}/{output_name}", exist_ok=True)
        filename = f"{folder_name}/{output_name}/{output_name}_{self._get_variable_parameters()}"
        np.savetxt(fname=filename + ".csv", X=output, delimiter=";")


def create_latin_table(n_of_samples, lower, upper):
    bounds = np.array([lower, upper]).T
    sampling = LHS(xlimits=bounds)
    return sampling(n_of_samples)
