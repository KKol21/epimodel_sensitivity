import os

import numpy as np

from dataloader import DataLoader
from model import VaccinatedModel
from prcc import generate_prcc_plot
from r0 import R0Generator
from sampler_vaccinated import SamplerVaccinated


class SimulationVaccinated:
    def __init__(self):
        # Load data
        self.data = DataLoader()

        # User-defined param_names
        self.susc_choices = [0.5, 1.0]
        self.r0_choices = [1.1, 2.5]

        # Define initial configs
        self._get_initial_config()

    def run(self):
        susceptibility = np.ones(self.no_ag)
        is_lhs_generated = False
        is_prcc_plot_generated = False
        for susc in self.susc_choices:
            susceptibility[:4] = susc
            self.params.update({"susc": susceptibility})
            for base_r0 in self.r0_choices:
                r0generator = R0Generator(param=self.params)
                sim_state = {"base_r0": base_r0, "susc": susc, "r0generator": r0generator,
                             "target_var": "r0"}
                param_generator = SamplerVaccinated(sim_state=sim_state, sim_obj=self)
                sim_state.update({"params": param_generator.param_names})

                if not is_lhs_generated:
                    param_generator.run_sampling()

                if not is_prcc_plot_generated:
                    os.makedirs('./sens_data/plots', exist_ok=True)
                    generate_prcc_plot(sim_state=sim_state)

    def _get_initial_config(self):
        self.no_ag = self.data.contact_data["home"].shape[0]
        self.model = VaccinatedModel(model_data=self.data)
        self.population = self.model.population
        self.age_vector = self.population.reshape((-1, 1))
        self.susceptibles = self.model.get_initial_values()[self.model.c_idx["s"] *
                                                            self.no_ag:(self.model.c_idx["s"] + 1) * self.no_ag]
        self.contact_matrix = self.data.contact_data["home"] + self.data.contact_data["work"] + \
                              self.data.contact_data["school"] + self.data.contact_data["other"]
        self.contact_home = self.data.contact_data["home"]
        self.upper_tri_indexes = np.triu_indices(self.no_ag)
        self.params = self.data.model_parameters_data
