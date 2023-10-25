import itertools
import os

import numpy as np
import scipy.stats as ss
import torch

from src.misc.dataloader import DataLoader
from src.model.model_vaccinated import VaccinatedModel
from src.sensitivity.prcc import get_prcc_values
from src.model.r0 import R0Generator
from src.sensitivity.sampler_vaccinated import SamplerVaccinated
from src.misc.plotter import generate_prcc_plot, generate_epidemic_plot, generate_epidemic_plot_


class SimulationVaccinated:
    """
    Class for running simulations and analyzing results of sensitivity of the model
    to the vaccination of age groups, considering different target variables.

    Methods:
        __init__(): Initializes the SimulationVaccinated object.
        run_sampling(): Generates samples, runs simulations, and saves the results.
        calculate_prcc(): Calculates PRCC (Partial Rank Correlation Coefficient) values from saved LHS tables.
        plot_prcc(): Creates and saves tornado plots from sensitivity data.
        plot_optimal_vaccine_distributions(): Creates and saves epidemic plots for the most
        optimal vaccine distributions.
        plot_subopt(): Creates and saves epidemic plots for suboptimal vaccine distributions.
        _get_initial_config(): Retrieves initial configurations for the simulation.

    """

    def __init__(self):
        # Load data
        self.data = DataLoader()
        self.test = True

        # User-defined parameters
        self.susc_choices = [1.0]
        self.r0_choices = [1.8]
        self.target_var_choices = ["i_max", "ic_max", "d_max"]  # i_max, ic_max, d_max
        self.n_samples = 100

        # Define initial configs
        self._get_initial_config()

        # Initalize model
        self.model = VaccinatedModel(sim_obj=self)
        self.susceptibles = self.model.get_initial_values()[self.model.idx("s_0")]

    def run_sampling(self):
        """

        Runs the sampling-based simulation with different parameter combinations.

        This method generates Latin Hypercube Sampling (LHS) samples of vaccine distributions for each parameter
        combination. The LHS tables and simulation results are saved in separate files in the 'sens_data_vacc/lhs' and
        'sens_data_vacc/simulations' directories, respectively.

        """
        susceptibility = torch.ones(self.n_age).to(self.data.device)
        for susc, base_r0, target_var in self.simulations:
            susceptibility[:4] = susc
            self.params.update({"susc": susceptibility})
            r0generator = R0Generator(self.data, device=self.data.device, n_age=self.n_age)
            # Calculate base transmission rate
            beta = base_r0 / r0generator.get_eig_val(contact_mtx=self.cm,
                                                     susceptibles=self.susceptibles.reshape(1, -1),
                                                     population=self.population)
            self.params.update({"beta": beta})
            # Generate matrices used in model representation
            self.model.initialize_constant_matrices()
            sim_state = {"base_r0": base_r0,
                         "susc": susc,
                         "r0generator": r0generator,
                         "target_var": target_var}
            param_generator = SamplerVaccinated(sim_state=sim_state,
                                                sim_obj=self,
                                                n_samples=self.n_samples)
            param_generator.run_sampling()

    def calculate_prcc(self):
        """

        Calculates PRCC (Partial Rank Correlation Coefficient) values from saved LHS tables and simulation results.

        This method reads the saved LHS tables and simulation results for each parameter combination and calculates
        the PRCC values. The PRCC values are saved in separate files in the 'sens_data_vacc/prcc' directory.

        """

        os.makedirs(f'../sens_data_vacc/prcc', exist_ok=True)
        for susc, base_r0, target_var in self.simulations:
            filename = f'{susc}-{base_r0}-{target_var}'
            lhs_table = np.loadtxt(f'../sens_data_vacc/lhs/lhs_{filename}.csv', delimiter=';')
            sim_output = np.loadtxt(f'../sens_data_vacc/simulations/simulations_{filename}.csv', delimiter=';')

            prcc = get_prcc_values(np.c_[lhs_table, sim_output.T])
            np.savetxt(fname=f'../sens_data_vacc/prcc/prcc_{filename}.csv', X=prcc)

    def plot_prcc(self):
        """

        Generates and saves PRCC plots based on the calculated PRCC values.

        This method reads the saved PRCC values for each parameter combination and generates
        PRCC plots using the `generate_prcc_plot` function. The plots are saved in separate files
        in the subfolder sens_data_vacc/prcc_plots.


        """
        os.makedirs(f'../sens_data_vacc/prcc_plots', exist_ok=True)
        for susc, base_r0, target_var in self.simulations:
            filename = f'{susc}-{base_r0}-{target_var}'
            prcc = np.loadtxt(fname=f'../sens_data_vacc/prcc/prcc_{filename}.csv')

            generate_prcc_plot(params=self.data.param_names,
                               target_var=target_var,
                               prcc=prcc,
                               filename=filename,
                               r0=base_r0)

    def calculate_p_values(self, significance=0.05):
        os.makedirs(f'../sens_data_vacc/p_values', exist_ok=True)
        for susc, base_r0, target_var in self.simulations:
            filename = f'{susc}-{base_r0}-{target_var}'
            prcc = np.loadtxt(fname=f'../sens_data_vacc/prcc/prcc_{filename}.csv')
            t = prcc * np.sqrt((self.n_samples - 2 - self.n_age) / (1 - prcc ** 2))
            # p-value for 2-sided test
            dof = self.n_samples - 2 - self.n_age
            p_values = 2 * (1 - ss.t.cdf(x=abs(t), df=dof))
            np.savetxt(fname=f'../sens_data_vacc/p_values/p_values_{filename}.csv', X=p_values)
            is_first = True
            for idx, p_val in enumerate(p_values):
                if p_val > significance:
                    if is_first:
                        print("\nInsignificant p-values in ", filename, " case: \n")
                        is_first = False
                    print("  age group: ", idx, " p-val: ", p_val)

    def plot_optimal_vaccine_distributions(self):
        """

        Generates epidemic plots based on the most optimal vaccine distributions found by LHS sampling.

        This method reads the saved optimal vaccine distributions for each parameter combination
        and generates epidemic plots using the `generate_epidemic_plot` function.

        The plots are saved in separate files in the 'sens_data_vacc/epidemic_plots' directory.

        """
        os.makedirs(f'../sens_data_vacc//epidemic_plots', exist_ok=True)
        for susc, base_r0, target_var in self.simulations:
            filename = f'{susc}-{base_r0}-{target_var}'
            vaccination = np.loadtxt(fname=f'../sens_data_vacc/optimal_vaccination/optimal_vaccination_{filename}.csv')
            generate_epidemic_plot(self, vaccination, filename, target_var, base_r0,
                                   compartments=["ic", "d"])

    def plot_subopt(self):
        """

        Generates epidemic plots for suboptimal vaccine distributions.

        This method reads the saved optimal vaccine distributions for a specific target variable and 2 base
        reproduction numbers: one with which the simulation will run, and another to showcase the consequences
        of not choosing the correct vaccination strategy. The epidemic plots are generated using the
        `generate_epidemic_plot_` function.

        The plots are saved in separate files in the 'sens_data_vacc/epidemic_plots_' directory.

        """

        os.makedirs('../sens_data_vacc//epidemic_plots_', exist_ok=True)
        target_var = 'ic_max'
        r0 = 3
        r0_bad = 3
        filename = f'1.0-{r0_bad}-{target_var}'
        filename_opt = f'1.0-{r0}-{target_var}'
        vaccination = np.loadtxt(fname=f'../sens_data_vacc/optimal_vaccination/optimal_vaccination_{filename}.csv')
        vaccination_opt = np.loadtxt(
            fname=f'../sens_data_vacc/optimal_vaccination/optimal_vaccination_{filename_opt}.csv')
        generate_epidemic_plot_(sim_obj=self,
                                vaccination=vaccination,
                                vaccination_opt=vaccination_opt,
                                filename=filename,
                                target_var=target_var,
                                r0=r0,
                                r0_bad=r0_bad,
                                compartments=['ic'])

    def _get_initial_config(self):
        self.params = self.data.model_parameters
        self.n_age = self.data.contact_data["home"].shape[0]
        self.cm = self.data.contact_data["home"] + self.data.contact_data["work"] + \
                  self.data.contact_data["school"] + self.data.contact_data["other"]
        self.device = self.data.device
        self.population = self.data.age_data.flatten()
        self.age_vector = self.population.reshape((-1, 1))
        self.simulations = [(susc, r0, target_var) for susc, r0, target_var in
                            itertools.product(self.susc_choices, self.r0_choices, self.target_var_choices)]
