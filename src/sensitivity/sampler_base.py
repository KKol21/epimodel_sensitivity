import os
import time
from abc import ABC, abstractmethod

import numpy as np
from smt.sampling_methods import LHS

from src.sensitivity.target_calc.output_generator import OutputGenerator


class SamplerBase(ABC):
    """
    Base class for performing sampling-based simulations.

    This abstract class defines the common structure and interface for samplers used to explore
    parameter combinations and collect simulation results.

    Args:
        sim_obj: The simulation object representing the underlying simulation model.

    Attributes:
        sim_obj: The simulation object representing the underlying simulation model.
        lhs_boundaries (dict): The boundaries for Latin Hypercube Sampling (LHS) parameter ranges.

    Methods:
        run_sampling(): Runs the sampling-based simulation to explore different parameter combinations
            and collect simulation results.
    """

    def __init__(self, sim_obj, sim_option):
        self.sim_obj = sim_obj
        self.sim_option = sim_option
        self._process_sampling_config()

    def _process_sampling_config(self):
        sim_obj = self.sim_obj
        self.n_samples = sim_obj.n_samples
        self.batch_size = sim_obj.batch_size

        spm = sim_obj.sampled_params_boundaries
        if spm is not None and all([param in sim_obj.params for param in spm.keys()]):
            self.lhs_boundaries = {param: np.array(spm[param]) for param in spm}

    @abstractmethod
    def run_sampling(self):
        pass

    def _get_lhs_table(self):
        bounds = np.array([bound for bound in self.lhs_boundaries.values()
                           if len(bound.shape) == 1])
        age_spec_bounds = self._get_age_spec_bounds()

        if age_spec_bounds.shape[0] == 0:
            bounds = bounds
        elif bounds.shape[0] == 0:
            bounds = age_spec_bounds
        else:
            np.concatenate((bounds, age_spec_bounds), axis=0)
        sampling = LHS(xlimits=bounds)
        return sampling(self.n_samples)

    def _get_age_spec_bounds(self):
        age_spec_bounds = [bounds for param in self.lhs_boundaries
                           for bounds in self.lhs_boundaries[param]
                           if len(self.lhs_boundaries[param].shape) == 2]
        return np.array(age_spec_bounds).T

    def _get_sim_output(self, lhs_table):
        print(f"\n Simulation for {self.n_samples} samples ({self.sim_obj.get_filename(self.sim_option)})")
        print(f"Batch size: {self.batch_size}\n")

        output_generator = OutputGenerator(self.sim_obj, self.sim_option)
        sim_outputs = output_generator.get_output(lhs_table=lhs_table)

        time.sleep(0.3)

        # Save samples, target values
        filename = self.sim_obj.get_filename(self.sim_option)
        self.save_output(output=lhs_table, output_name='lhs', filename=filename)
        for target_var, sim_output in sim_outputs.items():
            self.save_output(output=sim_output.cpu(), output_name='simulations', filename=filename + f"_{target_var}")

    def save_output(self, output, output_name, filename):
        folder_name = self.sim_obj.folder_name
        os.makedirs(folder_name, exist_ok=True)

        os.makedirs(f"{folder_name}/{output_name}", exist_ok=True)
        filename = f"{folder_name}/{output_name}/{output_name}_{filename}"
        np.savetxt(fname=filename + ".csv", X=output, delimiter=";")


def create_latin_table(n_of_samples, lower, upper):
    bounds = np.array([lower, upper]).T
    sampling = LHS(xlimits=bounds)
    return sampling(n_of_samples)
