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
        self.min_target = 1E10
        self.optimal_vacc = None

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
              "samples (", self._get_variable_parameters(), ")")

        get_output = partial(self.get_max_to, comp=self.target_var.split('_')[0])
        # Calculate values of target variable for each sample
        results = get_output(lhs_table)
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

    def get_max_to(self, lhs_table, comp: str):
        # Generate matrices used in model representation
        self.sim_obj.model.get_constant_matrices()
        start_t = time()

        t_eval = torch.stack(
            [torch.linspace(1, 1200, 1200)] * self.n_samples
                             ).to(self.sim_obj.data.device)
        y0 = torch.stack(
            [self.sim_obj.model.get_initial_values()] * self.n_samples
                         ).to(self.sim_obj.data.device)
        sol = self.sim_obj.model.get_solution(t_eval=t_eval,
                                              y0=y0,
                                              lhs_table=lhs_table
                                                  ).ys
        if self.sim_obj.test:
            # Check if population size changed
            if any([abs(self.sim_obj.population.sum() - sol[i, -1, :].sum()) > 50 for i in range(self.n_samples)]):
                raise Exception("Unexpected change in population size!")

        comp_max = []
        for i in range(self.n_samples):
            comp_sol = self.sim_obj.model.aggregate_by_age(solution=sol[i, :, :], comp=comp)
            comp_max.append(torch.max(comp_sol))
            if comp_max[i] < self.min_target:
                self.min_target = torch.clone(comp_max[i])
                self.optimal_vacc = torch.clone(lhs_table[i])
        elapsed = time() - start_t
        print("Elapsed time: ", elapsed,
              "\n Average iterations/second: ", round(self.n_samples / elapsed, 3),
              "\n")
        return torch.tensor(comp_max)

    def get_max(self, vaccination_sample: torch.Tensor, comp: str):
        """
        Returns the maximum value of a specified compartment (comp) in the simulation model,
        considering different scenarios based on the base reproduction number (r0) and
        a given vaccination sample.

        Args:
            vaccination_sample (torch.Tensor): The vaccination sample representing the distribution
                of available vaccines between the age groups of the population.
            comp (str): The compartment for which the maximum value is to be determined.
                This should be a valid compartment name recognized by the simulation model.

        Returns:
            float: The maximum value of the specified compartment (comp) in the simulation model.

        Raises:
            Exception: If there is an unexpected change in the population size during testing.
        """
        parameters = self.sim_obj.params
        days = 1200
        t = torch.linspace(1, days, days).to(self.sim_obj.data.device)
        daily_vac = vaccination_sample * parameters["total_vaccines"] / parameters["T"]
        sol = self.sim_obj.model.get_solution(t=t, cm=self.sim_obj.contact_matrix, daily_vac=daily_vac)
        if self.sim_obj.test:
            # Check if population size changed
            if abs(self.sim_obj.population.sum() - sol[-1, :].sum()) > 50:
                raise Exception("Unexpected change in population size!")
        self.sol = sol
        comp_sol = self.sim_obj.model.aggregate_by_age(solution=sol, comp=comp)
        comp_max = torch.max(comp_sol)
        if comp_max < self.min_target:
            self.min_target = comp_max
            self.optimal_vacc = daily_vac
            self.sim_obj.model.aggregate_by_age(sol, 'd').max()
        return comp_max

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
