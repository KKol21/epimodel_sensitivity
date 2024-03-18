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
        model_struct_path = PROJECT_PATH + "/examples/SEIR_no_age_groups/configs/SEIR_model_struct.json"
        config_path = PROJECT_PATH + "/examples/SEIR_no_age_groups/configs/SEIR_sampling_config.json"
        super().__init__(data, model_struct_path=model_struct_path, config_path=config_path)
        self.folder_name += "/sens_data_SEIR_no_ag"

        # Initalize model
        self.model = SEIRModel(sim_obj=self)

    def run_sampling(self):
        for option in self.sim_options_prod:
            susc = torch.Tensor(list(option["susc"].values())[0], device=self.device)
            self.params.update({"susc": susc})
            base_r0 = option["r0"]
            beta = self.get_beta_from_r0(base_r0)
            self.params["beta"] = beta

            self.model.initialize_matrices()

            param_generator = SamplerSEIR(sim_obj=self, sim_option=option)
            param_generator.run()
