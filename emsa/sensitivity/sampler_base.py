import os
import json
from abc import ABC, abstractmethod

import numpy as np
from smt.sampling_methods import LHS

from .sensitivity_model_base import get_params_col_idx
from .target_calc import OutputGenerator
from emsa.utils.results import join_result_name


class SamplerBase(ABC):
    """
    Base class for performing sampling-based simulations.

    This abstract class defines the common structure and interface for samplers used to explore
    parameter combinations and collect simulation results.

    Args:
        sim_object: The simulation object representing the underlying simulation model.

    Attributes:
        sim_object: The simulation object representing the underlying simulation model.
        lhs_bounds_dict (dict): The boundaries for Latin Hypercube Sampling (LHS) parameter ranges.

    Methods:
        run_sampling(): Runs the sampling-based simulation to explore different parameter combinations
            and collect simulation results.
    """

    def __init__(self, sim_object, variable_params=None):
        self.sim_object = sim_object
        self.variable_params = variable_params or {}
        self.sampled_params_boundaries = sim_object.sampled_params_boundaries
        self._process_sampling_config()

    def _process_sampling_config(self):
        sim_object = self.sim_object
        self.n_samples = sim_object.n_samples
        self.batch_size = sim_object.batch_size

        if spb := self.sampled_params_boundaries:
            self.lhs_bounds_dict = {param: np.array(spb[param]) for param in spb}
            self.pci = get_params_col_idx(spb)

    @abstractmethod
    def run(self):
        pass

    def get_lhs_table(self):
        bounds = self._get_lhs_bounds()
        if not np.isfinite(bounds).all() or np.any(bounds[:, 0] >= bounds[:, 1]):
            raise ValueError("Sampling bounds must be finite with lower < upper")
        sampling = LHS(xlimits=bounds, random_state=getattr(self.sim_object, "seed", None))
        return sampling(self.n_samples)

    def _get_lhs_bounds(self):
        # Each scalar or age-specific value gets its own [lower, upper] row.
        return np.concatenate(
            [np.atleast_2d(bounds.T) for bounds in self.lhs_bounds_dict.values()],
            axis=0,
        )

    def get_sim_output(self, lhs_table: np.ndarray):
        print(
            f"\n Simulation for {self.n_samples} samples ({self.sim_object.get_filename(self.variable_params)})"
        )
        print(f"Batch size: {self.batch_size}\n")

        output_generator = OutputGenerator(sim_object=self.sim_object)
        sim_outputs = output_generator.get_output(lhs_table=lhs_table)
        if sim_outputs == {}:
            raise Exception("No output was produced by OutputGenerator instance!")

        filename = self.sim_object.get_filename(self.variable_params)
        result = {
            "samples": np.array(lhs_table, copy=True),
            "targets": {
                name: values.detach().cpu().numpy().copy() for name, values in sim_outputs.items()
            },
            "metadata": self.sim_object.get_run_metadata(self.variable_params),
        }
        self.sim_object.results[filename] = result
        if not self.sim_object.save_results:
            return result
        # Retain the existing file layout for callers using the file-based API.
        self.save_output(output=lhs_table, output_name="lhs", filename=filename)
        for target_var, sim_output in sim_outputs.items():
            self.save_output(
                output=sim_output.cpu(),
                output_name="simulations",
                filename=join_result_name(filename, target_var),
            )
        with open(
            os.path.join(
                self.sim_object.folder_name, join_result_name("metadata", filename) + ".json"
            ),
            "w",
        ) as f:
            json.dump(result["metadata"], f, indent=2)
        return result

    def save_output(self, output, output_name: str, filename: str):
        folder_name = self.sim_object.folder_name
        os.makedirs(folder_name, exist_ok=True)

        dirname = os.path.join(folder_name, output_name)
        filename = os.path.join(dirname, join_result_name(output_name, filename))
        os.makedirs(dirname, exist_ok=True)
        np.savetxt(fname=filename + ".csv", X=output)


def create_latin_table(n_of_samples, lower, upper, seed=None):
    bounds = np.array([lower, upper]).T
    sampling = LHS(xlimits=bounds, random_state=seed)
    return sampling(n_of_samples)
