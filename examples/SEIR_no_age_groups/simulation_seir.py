import os

import numpy as np
import torch

from examples.SEIR_no_age_groups.model_seir import SEIRModel
from examples.SEIR_no_age_groups.sampler_seir import SamplerSEIR
from src.plotter import generate_tornado_plot
from src.simulation_base import SimulationBase
from src.dataloader import PROJECT_PATH


class SimulationSEIR(SimulationBase):
    def __init__(self, data):
        model_struct_path = PROJECT_PATH + "/examples/structures/SEIR_model_struct.json"
        config_path = PROJECT_PATH + "/examples/configs/SEIR_sampling_config.json"
        super().__init__(data, model_struct_path=model_struct_path, config_path=config_path)
        self.folder_name += "/sens_data_SEIR_no_ag"

        # Initalize model
        self.model = SEIRModel(sim_obj=self)

    def run_sampling(self):
        """

        Runs the sampling-based simulation with different parameter combinations.

        This method generates Latin Hypercube Sampling (LHS) samples of vaccine distributions for each parameter
        combination. The LHS tables and simulation results are saved in separate files in the 'sens_data_contact/lhs' and
        'sens_data_contact/simulations' directories, respectively.

        """
        for option in self.sim_options_prod:
            susc = torch.Tensor(next(iter(option["susc"].values())), device=self.device)
            self.params.update({"susc": susc})
            base_r0 = option["r0"]
            beta = self.get_beta_from_r0(base_r0)
            self.params["beta"] = beta

            self.model.initialize_matrices()

            param_generator = SamplerSEIR(sim_obj=self, sim_option=option)
            param_generator.run_sampling()

    def plot_prcc_for_simulations(self, filename):
        os.makedirs(f'{self.folder_name}/prcc_plots', exist_ok=True)
        labels = list(self.sampled_params_boundaries.keys())
        prcc = np.loadtxt(fname=f'{self.folder_name}/prcc/prcc_{filename}.csv')
        p_val = np.loadtxt(fname=f'{self.folder_name}/p_values/p_values_{filename}.csv')

        generate_tornado_plot(sim_obj=self,
                              labels=labels,
                              prcc=prcc,
                              p_val=p_val,
                              filename=filename)
