import torch

from examples.SEIR_no_age_groups.model_seir import SEIRModel
from examples.SEIR_no_age_groups.sampler_seir import SamplerSEIR
from src.dataloader import PROJECT_PATH
from src.simulation_base import SimulationBase


class SimulationSEIR(SimulationBase):
    def __init__(self, data):
        model_struct_path = PROJECT_PATH + "/examples/SEIHR_2_age_group/configs/SEIHR_model_struct.json"
        config_path = PROJECT_PATH + "/examples/SEIHR_2_age_group/configs/sampling_config.json"
        super().__init__(data=data, model_struct_path=model_struct_path, config_path=config_path)
        self.folder_name += "/sens_data_SEIR_2_ag"

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
            susc = torch.Tensor(list(option["susc"].values())[0], device=self.device)
            self.params.update({"susc": susc})
            base_r0 = option["r0"]
            beta = self.get_beta_from_r0(base_r0)
            self.params["beta"] = beta

            param_generator = SamplerSEIR(sim_obj=self, sim_option=option)
            param_generator.run()
