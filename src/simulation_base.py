import os
from abc import ABC, abstractmethod

import numpy as np
from scipy import stats as ss

from src.dataloader import PROJECT_PATH
from src.sensitivity.prcc import get_prcc_values


class SimulationBase(ABC):
    def __init__(self, data):
        # Load data
        self.data = data
        self.test = True

        self.n_samples = 500
        self.batch_size = 500
        self.target_var = None

        self._get_initial_config()

    def _get_initial_config(self):
        self.params = self.data.model_params
        self.n_age = self.data.n_age
        self.cm = self.data.cm
        self.device = self.data.device
        self.population = self.data.age_data.flatten()
        self.age_vector = self.population.reshape((-1, 1))
        self.folder_name = PROJECT_PATH + "\sens_data"

    @abstractmethod
    def run_sampling(self):
        pass

    def calculate_prcc(self, filename):
        """

        Calculates PRCC (Partial Rank Correlation Coefficient) values from saved LHS tables and simulation results.

        This method reads the saved LHS tables and simulation results for each parameter combination and calculates
        the PRCC values. The PRCC values are saved in separate files in the 'sens_data_"folder_name"/prcc' directory.

        """
        folder_name = self.folder_name
        os.makedirs(f"{folder_name}/prcc", exist_ok=True)
        lhs_table = np.loadtxt(f'{folder_name}/lhs/lhs_{filename}.csv', delimiter=';')
        sim_output = np.loadtxt(f'{folder_name}/simulations/simulations_{filename}.csv', delimiter=';')

        prcc = get_prcc_values(np.c_[lhs_table, sim_output.T])
        np.savetxt(fname=f'{folder_name}/prcc/prcc_{filename}.csv', X=prcc)

    def calculate_p_values(self, filename, significance=0.05):
        os.makedirs(self.folder_name + '/p_values', exist_ok=True)
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
