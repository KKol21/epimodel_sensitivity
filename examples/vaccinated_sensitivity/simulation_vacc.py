import os

import numpy as np
import torch

from examples.vaccinated_sensitivity.sampler_vaccinated import SamplerVaccinated
from examples.vaccinated_sensitivity.sensitivity_model_vaccinated import VaccinatedModel
from src.model.r0 import R0Generator
from src.plotter import generate_tornado_plot, generate_epidemic_plot, generate_epidemic_plot_
from src.simulation_base import SimulationBase
from src.dataloader import PROJECT_PATH


class SimulationVaccinated(SimulationBase):
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

    def __init__(self, data):
        struct_path = PROJECT_PATH + "/examples/structures/vaccinated_model_struct.json"
        config_path = PROJECT_PATH + "/examples/configs/vaccinated_sampling_config.json"
        super().__init__(data, model_struct_path=struct_path, config_path=config_path)

        self.folder_name += "/sens_data_vacc"
        self.model = VaccinatedModel(sim_obj=self)

    def run_sampling(self):
        """

        Runs the sampling-based simulation with different parameter combinations.

        This method generates Latin Hypercube Sampling (LHS) samples of vaccine distributions for each parameter
        combination. The LHS tables and simulation results are saved in separate files in the 'sens_data_vacc/lhs' and
        'sens_data_vacc/simulations' directories, respectively.

        """
        for sim_opt in self.sim_options_prod:
            base_r0 = sim_opt["r0"]
            beta = self.get_beta_from_r0(base_r0)
            self.params.update({"beta": beta})
            # Generate matrices used in model representation
            self.model.initialize_matrices()

            param_generator = SamplerVaccinated(sim_obj=self, sim_option=sim_opt)
            param_generator.run()

    def calculate_prcc_for_simulations(self):
        for susc, base_r0, target_var in self.simulations:
            filename = f'{susc}-{base_r0}-{target_var}'
            self.calculate_prcc(filename=filename)

    def plot_prcc_tornado_with_p_values(self):
        """

        Generates and saves PRCC plots based on the calculated PRCC values.

        This method reads the saved PRCC values for each parameter combination and generates
        PRCC plots using the `generate_prcc_plot` function. The plots are saved in separate files
        in the subfolder sens_data_contact/prcc_plots.


        """
        os.makedirs(f'{self.folder_name}/prcc_plots', exist_ok=True)

        def get_age_group(idx, bin_size):
            age_start = idx * bin_size
            max_age = bin_size * (self.n_age - 1)
            return f'{age_start}-{age_start + bin_size - 1} ' if age_start != max_age else f'{max_age}+ '

        labels = [get_age_group(idx, 5) for idx in range(self.n_age)]

        for susc, base_r0, target_var in self.simulations:
            filename = f'{susc}-{base_r0}-{target_var}'
            prcc = np.loadtxt(fname=f'{self.folder_name}/prcc/prcc_{filename}.csv')
            p_val = np.loadtxt(fname=f'{self.folder_name}/p_values/p_values_{filename}.csv')

            generate_tornado_plot(sim_obj=self, labels=labels,
                                  title="PRCC values based on vaccine distribution samples",
                                  target_var=target_var, prcc=prcc, p_val=p_val,
                                  filename=filename, r0=base_r0)


    def plot_optimal_vaccine_distributions(self):
        """

        Generates epidemic plots based on the most optimal vaccine distributions found by LHS sampling.

        This method reads the saved optimal vaccine distributions for each parameter combination
        and generates epidemic plots using the `generate_epidemic_plot` function.

        The plots are saved in separate files in the 'sens_data_vacc/epidemic_plots' directory.

        """
        os.makedirs(f'{self.folder_name}/epidemic_plots', exist_ok=True)
        for susc, base_r0, target_var in self.simulations:
            filename = f'{susc}-{base_r0}-{target_var}'
            vaccination = np.loadtxt(fname=f'{self.folder_name}/optimal_vaccination/optimal_vaccination_{filename}.csv')
            generate_epidemic_plot(self, torch.from_numpy(vaccination).float(), filename, target_var, base_r0,
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

        os.makedirs(self.folder_name + '/epidemic_plots_', exist_ok=True)
        target_var = 'ic_max'
        r0 = 3
        r0_bad = 3
        filename = f'1.0-{r0_bad}-{target_var}'
        filename_opt = f'1.0-{r0}-{target_var}'
        vaccination = np.loadtxt(fname=f'{self.folder_name}/optimal_vaccination/optimal_vaccination_{filename}.csv')
        vaccination_opt = np.loadtxt(
            fname=f'{self.folder_name}/optimal_vaccination/optimal_vaccination_{filename_opt}.csv')
        generate_epidemic_plot_(sim_obj=self,
                                vaccination=vaccination,
                                vaccination_opt=vaccination_opt,
                                filename=filename,
                                target_var=target_var,
                                r0=r0,
                                r0_bad=r0_bad,
                                compartments=['ic'])

    def calculate_all_p_values(self):
        for susc, base_r0, target_var in self.simulations:
            filename = f'{susc}-{base_r0}-{target_var}'
            self.calculate_p_values(filename=filename)
