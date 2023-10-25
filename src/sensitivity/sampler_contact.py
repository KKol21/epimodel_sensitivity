import time

import numpy as np
from smt.sampling_methods import LHS
import torch

from src.sensitivity.sampler_base import SamplerBase


class SamplerVaccinated(SamplerBase):
    def __init__(self, sim_state: dict, sim_obj, n_samples):
        super().__init__(sim_state, sim_obj)
        self.mode = "vacc"
        self.n_samples = n_samples
        self.sim_obj = sim_obj
        self.susc = sim_state["susc"]
        self.target_var = sim_state["target_var"]

        # Get number of elements in the upper triangular matrix
        self.upper_tri_size = int((self.sim_obj.no_ag + 1) * self.sim_obj.no_ag / 2)

        self.lhs_boundaries = {"lower": np.zeros(self.upper_tri_size),
                               "upper": np.ones(self.upper_tri_size)
                               }
        self.batch_size = 1000

    def run_sampling(self):
        n_samples = self.n_samples
        batch_size = self.batch_size
        # Create samples
        bounds = np.array([bounds for bounds in self.lhs_boundaries.values()]).T
        sampling = LHS(xlimits=bounds)
        lhs_table = sampling(n_samples)
        # Make sure that total vaccines given to an age group
        # doesn't exceed the population of that age group
        print(f"\n Simulation for {n_samples} samples ({self._get_variable_parameters()})")
        print(f"Batch size: {batch_size}\n")

        # Calculate values of target variable for each sample
        results = self.sim_obj.model.get_batched_output(lhs_table,
                                                        batch_size,
                                                        self.target_var)
        # Sort tables by target values
        sorted_idx = results.argsort()
        results = results[sorted_idx]
        lhs_table = lhs_table[sorted_idx]

        sim_output = results
        time.sleep(0.3)

        # Save samples, target values, and the most optimal vaccination strategy found with sampling
        self._save_output(output=lhs_table, folder_name='lhs')
        self._save_output(output=sim_output, folder_name='simulations')

    def _get_sim_output_cm_entries_lockdown(self, lhs_sample: np.ndarray):
        # Get ratio matrix
        ratio_matrix = get_rectangular_matrix_from_upper_triu(rvector=lhs_sample,
                                                              matrix_size=self.sim_obj.no_ag)
        # Get modified full contact matrix
        cm_sim = (1 - ratio_matrix) * (self.sim_obj.contact_matrix - self.sim_obj.contact_home)
        cm_sim += self.sim_obj.contact_home
        # Get output
        output = self._get_output(cm_sim=cm_sim)
        cm_total_sim = (cm_sim * self.sim_obj.age_vector)[self.sim_obj.upper_tri_indexes]
        output = np.append(cm_total_sim, output)
        return list(output)

    def _get_variable_parameters(self):
        return f'{self.susc}-{self.base_r0}-{self.target_var}'


def get_rectangular_matrix_from_upper_triu(rvector, matrix_size):
    upper_tri_indexes = np.triu_indices(matrix_size)
    new_contact_mtx = np.zeros((matrix_size, matrix_size))
    new_contact_mtx[upper_tri_indexes] = rvector
    new_2 = new_contact_mtx.T
    new_2[upper_tri_indexes] = rvector
    return np.array(new_2)


def get_contact_matrix_from_upper_triu(rvector, age_vector):
    new_2 = get_rectangular_matrix_from_upper_triu(rvector=rvector,
                                                   matrix_size=age_vector.shape[0])
    vector = np.array(new_2 / age_vector)
    return vector