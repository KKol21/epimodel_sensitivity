import itertools
import os
from abc import ABC

import numpy as np

from src.misc.dataloader import DataLoader
from src.sensitivity.prcc import get_prcc_values
from src.misc.plotter import generate_prcc_plot


class SimulationBase(ABC):
    def __init__(self):
        # Load data
        self.data = DataLoader()
        self.test = True

        # User-defined parameters
        self.susc_choices = [1.0]
        self.r0_choices = [1.8]
        self.target_var_choices = ["i_max", "ic_max", "d_max"]  # i_max, ic_max, d_max
        self.n_samples = 100
        self.batch_size = 1000

        # Define initial configs
        self._get_initial_config()

    def _get_initial_config(self):
        self.params = self.data.model_parameters
        self.n_age = self.data.contact_data["home"].shape[0]
        self.param_names = np.array([f'daily_vac_{i}' for i in range(self.n_age)])
        self.cm = self.data.contact_data["home"] + self.data.contact_data["work"] + \
                  self.data.contact_data["school"] + self.data.contact_data["other"]
        self.device = self.data.device
        self.population = self.data.age_data.flatten()
        self.age_vector = self.population.reshape((-1, 1))
        self.simulations = [(susc, r0, target_var) for susc, r0, target_var in
                            itertools.product(self.susc_choices, self.r0_choices, self.target_var_choices)]
        self.folder_name = None

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

    def plot_prcc(self):
        """

        Generates and saves PRCC plots based on the calculated PRCC values.

        This method reads the saved PRCC values for each parameter combination and generates
        PRCC plots using the `generate_prcc_plot` function. The plots are saved in separate files
        in the subfolder sens_data_contact/prcc_plots.


        """
        os.makedirs(f'{self.folder_name}/prcc_plots', exist_ok=True)
        for susc, base_r0, target_var in self.simulations:
            filename = f'{susc}-{base_r0}-{target_var}'
            prcc = np.loadtxt(fname=f'{self.folder_name}/prcc/prcc_{filename}.csv')

            generate_prcc_plot(params=self.param_names,
                               target_var=target_var,
                               prcc=prcc,
                               filename=filename,
                               r0=base_r0)
