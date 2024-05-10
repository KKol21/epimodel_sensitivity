import numpy as np
import torch

from emsa.sensitivity.sampler_base import SamplerBase


class SamplerVaccinated(SamplerBase):
    def __init__(self, sim_object, variable_params):
        super().__init__(sim_object, variable_params)
        self.sim_object = sim_object
        self.lhs_bounds_dict = {
            "vaccines": np.array([np.zeros(sim_object.n_age),  # Ratio of daily vaccines given to each age group
                                  np.ones(sim_object.n_age)])
        }

    def run(self):
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
        lhs_table = self.get_lhs_table()
        # Make sure that total vaccines given to an age group
        # doesn't exceed the population of that age group
        lhs_table = self.allocate_vaccines(lhs_table)
        self.get_sim_output(lhs_table)

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
        params = self.sim_object.params
        total_vac = params["total_vaccines"] * lhs_table
        population = np.array(self.sim_object.population.cpu())

        while np.any(total_vac > population):
            mask = total_vac > population
            excess = population - total_vac
            redistribution = excess * lhs_table

            total_vac[mask] = np.tile(population, (lhs_table.shape[0], 1))[mask]
            total_vac[~mask] += redistribution[~mask]

            lhs_table = self.norm_table_rows(total_vac / params['total_vaccines'])
            total_vac = params["total_vaccines"] * lhs_table
        return lhs_table
