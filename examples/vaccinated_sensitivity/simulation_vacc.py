import itertools
import os

from examples.vaccinated_sensitivity.sampler_vaccinated import SamplerVaccinated
from examples.vaccinated_sensitivity.sensitivity_model_vaccinated import VaccinatedModel
from src.dataloader import PROJECT_PATH
from src.simulation_base import SimulationBase


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
        struct_path = PROJECT_PATH + "/examples/vaccinated_sensitivity/configs/vaccinated_model_struct.json"
        config_path = PROJECT_PATH + "/examples/vaccinated_sensitivity/configs/vaccinated_sampling_config.json"
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
        for option, target in itertools.product(self.sim_options_prod, self.target_vars):
            filename = self.get_filename(option) + f"_{target}"
            self.plot_prcc(filename, labels)
