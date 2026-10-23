import numpy as np
import torch

from emsa.sensitivity.sampler_base import SamplerBase


class SamplerVaccinated(SamplerBase):
    def __init__(self, sim_object, variable_params):
        super().__init__(sim_object, variable_params)
        self.sim_object = sim_object
        self.lhs_bounds_dict = {
            "vaccines": np.array(
                [
                    np.zeros(sim_object.n_age),  # Ratio of daily vaccines given to each age group
                    np.ones(sim_object.n_age),
                ]
            )
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
        return self.get_sim_output(lhs_table)

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
        population = np.asarray(self.sim_object.population.cpu(), dtype=float)
        weights = np.asarray(lhs_table, dtype=float)
        if weights.ndim != 2 or weights.shape[1] != len(population):
            raise ValueError("Vaccine samples must have one column per age group")
        if (
            not np.isfinite(weights).all()
            or np.any(weights < 0)
            or np.any(weights.sum(axis=1) <= 0)
        ):
            raise ValueError(
                "Vaccine samples must contain nonnegative weights with positive row totals"
            )
        if not np.isfinite(population).all() or np.any(population < 0):
            raise ValueError("Age-group populations must be finite and nonnegative")

        total = self.sim_object.params["total_vaccines"]
        if not np.isfinite(total) or total < 0 or total > population.sum():
            raise ValueError("Total vaccines must be between zero and the total population")

        weights = self.norm_table_rows(weights)
        if total == 0:
            return weights
        allocated = total * weights
        # Each redistribution caps at least one additional age group, so this terminates.
        for _ in range(len(population)):
            excess = np.maximum(allocated - population, 0).sum(axis=1, keepdims=True)
            allocated = np.minimum(allocated, population)
            if not np.any(excess > 0):
                break
            capacity = population - allocated
            redistribution_weights = weights * (capacity > 0)
            # A zero-weight group can still receive doses when all preferred groups are full.
            redistribution_weights = np.where(
                redistribution_weights.sum(axis=1, keepdims=True) > 0,
                redistribution_weights,
                capacity,
            )
            weight_sum = redistribution_weights.sum(axis=1, keepdims=True)
            shares = redistribution_weights / np.where(weight_sum > 0, weight_sum, 1)
            allocated += excess * shares
        return np.minimum(allocated, population) / total
