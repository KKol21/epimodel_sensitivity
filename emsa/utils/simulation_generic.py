import os

from emsa.sensitivity.generic_model import GenericModel
from emsa.sensitivity.generic_sampler import GenericSampler
from emsa.utils.dataloader import PROJECT_PATH
from emsa.utils.simulation_base import SimulationBase


class SimulationGeneric(SimulationBase):
    def __init__(self, data, model_struct_path, config_path):
        model_struct_path = os.path.join(
            PROJECT_PATH,
            model_struct_path
        )
        config_path = os.path.join(
            PROJECT_PATH,
            config_path
        )
        super().__init__(data=data, model_struct_path=model_struct_path, config_path=config_path)

        # Initalize model
        self.model = GenericModel(sim_object=self)

    def run_sampling(self):
        param_generator = GenericSampler(sim_object=self)
        param_generator.run()
