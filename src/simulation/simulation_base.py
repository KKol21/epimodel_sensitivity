import itertools
import os
from abc import ABC, abstractmethod

import numpy as np
from scipy import stats as ss

from src.dataloader import DataLoader, PROJECT_PATH
from src.sensitivity.prcc import get_prcc_values


class SimulationBase(ABC):
    def __init__(self):
        # Load data
        self.data = DataLoader()
        self.test = True

        # User-defined parameters
        self.susc_choices = [1.0]
        self.r0_choices = [1.8]
        self.target_var_choices = ["i_max", "ic_max", "d_max"]  # i_max, ic_max, d_max
        self.n_samples = 50
        self.batch_size = 500

        # Define initial configs
        self._get_initial_config()

    def _get_initial_config(self):
        self.params = self.data.model_params
        self.n_age = self.data.n_age
        self.param_names = np.array([f'daily_vac_{i}' for i in range(self.n_age)])
        self.cm = self.data.cm
        self.device = self.data.device
        self.population = self.data.age_data.flatten()
        self.age_vector = self.population.reshape((-1, 1))
        self.simulations = [(susc, r0, target_var) for susc, r0, target_var in
                            itertools.product(self.susc_choices, self.r0_choices, self.target_var_choices)]
        self.folder_name = PROJECT_PATH

    @abstractmethod
    def run_sampling(self):
        pass

    def calculate_prcc(self):
        """

        Calculates PRCC (Partial Rank Correlation Coefficient) values from saved LHS tables and simulation results.

        This method reads the saved LHS tables and simulation results for each parameter combination and calculates
        the PRCC values. The PRCC values are saved in separate files in the 'sens_data_contact/prcc' directory.

        """
        folder_name = self.folder_name
        os.makedirs(f"{folder_name}/prcc", exist_ok=True)
        for susc, base_r0, target_var in self.simulations:
            filename = f'{susc}-{base_r0}-{target_var}'
            lhs_table = np.loadtxt(f'{folder_name}/lhs/lhs_{filename}.csv', delimiter=';')
            sim_output = np.loadtxt(f'{folder_name}/simulations/simulations_{filename}.csv', delimiter=';')

            prcc = get_prcc_values(np.c_[lhs_table, sim_output.T])
            np.savetxt(fname=f'{folder_name}/prcc/prcc_{filename}.csv', X=prcc)

    def calculate_p_values(self, significance=0.05):
        os.makedirs(self.folder_name + '/p_values', exist_ok=True)
        for susc, base_r0, target_var in self.simulations:
            filename = f'{susc}-{base_r0}-{target_var}'
            prcc = np.loadtxt(fname=f'{self.folder_name}/prcc/prcc_{filename}.csv')
            t = prcc * np.sqrt((self.n_samples - 2 - self.n_age) / (1 - prcc ** 2))
            # p-value for 2-sided test
            dof = self.n_samples - 2 - self.n_age
            p_values = 2 * (1 - ss.t.cdf(x=abs(t), df=dof))
            np.savetxt(fname=f'{self.folder_name}/p_values/p_values_{filename}.csv', X=p_values)
            is_first = True
            if len(p_values) < 30:
                for idx, p_val in enumerate(p_values):
                    if p_val > significance:
                        if is_first:
                            print("\nInsignificant p-values in ", filename, " case: \n")
                            is_first = False
                        print(f"\t {idx}. p-val: ", p_val)
