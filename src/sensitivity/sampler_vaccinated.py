from functools import partial
from time import sleep, time

import numpy as np
from smt.sampling_methods import LHS
import torch
from tqdm import tqdm

from src.sensitivity.sampler_base import SamplerBase


class SamplerVaccinated(SamplerBase):
    def __init__(self, sim_state: dict, sim_obj, n_samples):
        super().__init__(sim_state, sim_obj)
        self.n_samples = n_samples
        self.sim_obj = sim_obj
        self.susc = sim_state["susc"]
        self.target_var = sim_state["target_var"]
        self.lhs_boundaries = {"lower": np.zeros(sim_obj.n_age),    # Ratio of daily vaccines given to each age group
                               "upper": np.ones(sim_obj.n_age)
                               }
        self.optimal_vacc = None
        self.batch_size = 5000

    def run_sampling(self):
        """
        Runs the sampling-based simulation to explore different parameter combinations and
        collect simulation results for analysis.

        This method performs Latin Hypercube Sampling (LHS) to generate vaccination distributions.
        It then allocates vaccines to ensure that the total vaccines given to an age group does
        not exceed the population of that age group. The simulation is executed for each parameter
        combination, and the maximum value of a specified component (comp) is obtained using the
        `get_max` method. The simulation results are sorted based on the target variables values
        and saved in separate output files.

        Returns:
            None

        """
        # Create samples
        bounds = np.array([bounds for bounds in self.lhs_boundaries.values()]).T
        sampling = LHS(xlimits=bounds)
        lhs_table = sampling(self.n_samples)
        # Make sure that total vaccines given to an age group
        # doesn't exceed the population of that age group
        lhs_table = self.allocate_vaccines(lhs_table).to(self.sim_obj.data.device)

        print("Simulation for", self.n_samples,
              "samples (", self._get_variable_parameters(), ") \n"
              "Batch size: ", self.batch_size, "\n")

        # Calculate values of target variable for each sample
        results = self.get_batched_output(lhs_table)
        # Sort tables by target values
        sorted_idx = results.argsort()
        results = results[sorted_idx]
        lhs_table = lhs_table[sorted_idx]
        sim_output = results
        sleep(0.3)

        # Save samples, target values, and the most optimal vaccination strategy found with sampling
        self._save_output(output=lhs_table, folder_name='lhs')
        self._save_output(output=sim_output, folder_name='simulations')
        self._save_output(output=self.optimal_vacc, folder_name='optimal_vaccination')

    def get_batched_output(self, lhs_table):
        batch_size = self.batch_size
        n_batches = self.n_samples // batch_size
        batches = []

        for i in tqdm(range(n_batches), desc="Batches completed", leave=False):
            indices = slice(i * batch_size, (i + 1) * batch_size) if i < n_batches - 1 \
                else slice(i * batch_size, -1)

            solutions = self.get_sol_from_lhs(lhs_table[indices])
            comp_maxes = self.get_max(sol=solutions, comp=self.target_var.split('_')[0])
            batches.append(comp_maxes)

        result = torch.concat(batches)
        self.optimal_vacc = lhs_table[result.argmin()]
        return result

    def get_sol_from_lhs(self, lhs_table):
        # Generate matrices used in model representation
        self.sim_obj.model.get_constant_matrices()

        # Initialize timesteps and initial values
        t_eval = torch.stack(
            [torch.linspace(1, 1100, 1100)] * lhs_table.shape[0]
        ).to(self.sim_obj.data.device)

        y0 = torch.stack(
            [self.sim_obj.model.get_initial_values()] * lhs_table.shape[0]
        ).to(self.sim_obj.data.device)

        sol = self.sim_obj.model.get_solution(t_eval=t_eval,
                                              y0=y0,
                                              lhs_table=lhs_table
                                              ).ys
        if self.sim_obj.test:
            # Check if population size changed
            if any([abs(self.sim_obj.population.sum() - sol[i, -1, :].sum()) > 50 for i in range(sol.shape[0])]):
                raise Exception("Unexpected change in population size!")
        return sol

    def get_max(self, sol, comp: str):
        comp_max = []
        for i in range(sol.shape[0]):
            comp_sol = self.sim_obj.model.aggregate_by_age(solution=sol[i, :, :], comp=comp)
            comp_max.append(torch.max(comp_sol))
        return torch.tensor(comp_max)

    def _get_variable_parameters(self):
        """
        Returns:
            str: A string representing the variable parameters in the format: susc-base_r0-target_var
        """
        return f'{self.susc}-{self.base_r0}-{self.target_var}'

    @staticmethod
    def norm_table_rows(table: np.ndarray):
        return table / np.sum(table, axis=1, keepdims=True)

    def allocate_vaccines(self, lhs_table: np.ndarray):
        """
        Allocates vaccines to ensure that the number of allocated vaccines does not exceed
        the population size of any given age group.

        Args:
            lhs_table (torch.Tensor): The table of reallocated vaccines.

        Returns:
            torch.Tensor: The adjusted table of vaccination allocations.

        """
        lhs_table = self.norm_table_rows(lhs_table)
        params = self.sim_obj.params
        total_vac = params["total_vaccines"] * lhs_table
        population = np.array(self.sim_obj.population.cpu())

        while np.any(total_vac > population):
            mask = total_vac > population
            excess = population - total_vac
            redistribution = excess * lhs_table

            total_vac[mask] = np.tile(population, (lhs_table.shape[0], 1))[mask]
            total_vac[~mask] += redistribution[~mask]

            lhs_table = self.norm_table_rows(total_vac / params['total_vaccines'])
            total_vac = params["total_vaccines"] * lhs_table
        return torch.Tensor(lhs_table)
