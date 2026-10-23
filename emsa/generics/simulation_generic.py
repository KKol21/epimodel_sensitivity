import os

from emsa.generics import GenericModel
from emsa.generics import GenericSampler
from emsa.utils.dataloader import PROJECT_PATH
from emsa.utils import SimulationBase


class SimulationGeneric(SimulationBase):
    def __init__(
        self,
        data,
        model_struct_path,
        sampling_config_path,
        folder_name=None,
        *,
        output_dir=None,
        save_results=True,
    ):
        # Both inputs accept dictionaries or the existing JSON file paths.
        if not isinstance(model_struct_path, dict):
            model_struct_path = os.path.join(PROJECT_PATH, model_struct_path)
        if not isinstance(sampling_config_path, dict):
            sampling_config_path = os.path.join(PROJECT_PATH, sampling_config_path)
        super().__init__(
            data=data,
            model_struct_path=model_struct_path,
            sampling_config_path=sampling_config_path,
            output_dir=output_dir,
            save_results=save_results,
        )

        if folder_name:
            self.folder_name = os.path.join(self.folder_name, folder_name)

        # Initalize model
        self.model = GenericModel(sim_object=self)

    def run_sampling(self):
        param_generator = GenericSampler(sim_object=self)
        return param_generator.run()
