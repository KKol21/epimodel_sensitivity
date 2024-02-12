import itertools
import os

import numpy as np
import torch

from examples.contact_sensitivity.sensitivity_model_contact import ContactModel
from src.model.r0 import R0Generator
from examples.contact_sensitivity.sampler_contact import SamplerContact
from src.simulation_base import SimulationBase
from src.plotter import plot_prcc_p_values_as_heatmap


class SimulationContact(SimulationBase):
    """
    Class for running simulations and analyzing results of sensitivity of the model
    to the contact matrix

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
        super().__init__(data)
        self.upper_tri_size = int((self.n_age + 1) * self.n_age / 2)
        self.folder_name += "/sens_data_contact"

        # Initalize model
        self.model = ContactModel(sim_obj=self, sim_state=None)
        self.susceptibles = self.model.get_initial_values()[self.model.idx("s_0")]

        # User-defined params
        self.susc_choices = [1.0]
        self.r0_choices = [1.8]
        self.target_var_choices = ["d_max"]  # ["i_max", "ic_max", "d_max"]
        self.simulations = list(itertools.product(self.susc_choices, self.r0_choices, self.target_var_choices))

    def run_sampling(self):
        """

        Runs the sampling-based simulation with different parameter combinations.

        This method generates Latin Hypercube Sampling (LHS) samples of vaccine distributions for each parameter
        combination. The LHS tables and simulation results are saved in separate files in the 'sens_data_contact/lhs' and
        'sens_data_contact/simulations' directories, respectively.

        """
        susceptibility = torch.ones(self.n_age).to(self.data.device)
        for susc, base_r0, target_var in self.simulations:
            susceptibility[:4] = susc
            self.params.update({"susc": susceptibility})
            r0generator = R0Generator(self.data)
            # Calculate base transmission rate
            beta = base_r0 / r0generator.get_eig_val(contact_mtx=self.cm,
                                                     susceptibles=self.susceptibles.reshape(1, -1),
                                                     population=self.population)
            self.params.update({"beta": beta})
            # Generate matrices used in model representation
            sim_state = {"base_r0": base_r0,
                         "susc": susc,
                         "r0generator": r0generator,
                         "target_var": target_var}
            self.model = ContactModel(sim_obj=self, sim_state=sim_state)
            self.model.initialize_matrices()

            param_generator = SamplerContact(sim_state=sim_state,
                                             sim_obj=self)
            param_generator.run_sampling()

    def calculate_prcc_for_simulations(self):
        for susc, base_r0, target_var in self.simulations:
            filename = f'{susc}-{base_r0}-{target_var}'
            self.calculate_prcc(filename=filename)

    def plot_prcc_and_p_values(self):
        """

        Generates and saves PRCC plots based on the calculated PRCC values.

        This method reads the saved PRCC values for each parameter combination and generates
        PRCC plots using the `generate_prcc_plot` function. The plots are saved in separate files
        in the subfolder sens_data_contact/prcc_plots.


        """
        os.makedirs(f'{self.folder_name}/prcc_p_val_plots', exist_ok=True)
        for susc, base_r0, target_var in self.simulations:
            filename = f'{susc}-{base_r0}-{target_var}'
            prcc = np.loadtxt(fname=f'{self.folder_name}/prcc/prcc_{filename}.csv')
            p_val = np.loadtxt(fname=f'{self.folder_name}/p_values/p_values_{filename}.csv')

            plot_prcc_p_values_as_heatmap(n_age=self.n_age,
                                          prcc_vector=prcc,
                                          p_values=p_val,
                                          filename_to_save=f"{self.folder_name}/prcc_p_val_plots/{filename}.pdf",
                                          plot_title="test")

    def calculate_all_p_values(self):
        for susc, base_r0, target_var in self.simulations:
            filename = f'{susc}-{base_r0}-{target_var}'
            self.calculate_p_values(filename=filename)
