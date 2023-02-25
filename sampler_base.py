from abc import ABC, abstractmethod
import os

import numpy as np
from smt.sampling_methods import LHS


class SamplerBase(ABC):
    def __init__(self, sim_state: dict, sim_obj):
        self.sim_obj = sim_obj
        self.sim_state = sim_state
        self.base_r0 = sim_state["base_r0"]
        self.beta = sim_state["beta"] if "beta" in sim_state.keys() else None
        self.type = sim_state["type"] if "type" in sim_state.keys() else None
        self.r0generator = sim_state["r0generator"]
        self.lhs_boundaries = None

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def _get_variable_parameters(self):
        pass

    def _get_lhs_table(self, number_of_samples: int = 40000):
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
        os.makedirs("./sens_data", exist_ok=True)

        # Save LHS output
        os.makedirs("./sens_data/" + folder_name, exist_ok=True)
        filename = f"./sens_data/{folder_name}/{folder_name}_{self._get_variable_parameters()}"
        np.savetxt(fname=filename + ".csv", X=output, delimiter=";")


def create_latin_table(n_of_samples, lower, upper):
    bounds = np.array([lower, upper]).T
    sampling = LHS(xlimits=bounds)
    return sampling(n_of_samples)
