import os

import numpy as np

from dataloader import DataLoader
from model import VaccinatedModel
from prcc import generate_prcc_plot, get_prcc_values
from r0 import R0Generator
from sampler_vaccinated import SamplerVaccinated


class SimulationVaccinated:
    def __init__(self):
        # Load data
        self.data = DataLoader()
        self.test = True

        # User-defined param_names
        self.susc_choices = [0.5, 1.0]
        self.r0_choices = [1.1, 2.5]
        self.target_var = "ic_max"  # r0, i_max, ic_max, d_max

        # Define initial configs
        self._get_initial_config()

    def run_sampling(self):
        susceptibility = np.ones(self.no_ag)
        for susc in self.susc_choices:
            susceptibility[:4] = susc
            self.params.update({"susc": susceptibility})
            r0generator = R0Generator(param=self.params)
            for base_r0 in self.r0_choices:
                sim_state = {"base_r0": base_r0, "susc": susc, "r0generator": r0generator,
                             "target_var": self.target_var}
                param_generator = SamplerVaccinated(sim_state=sim_state, sim_obj=self)
                sim_state.update({"params": self.data.param_names})
                param_generator.run_sampling()

    def calculate_prcc(self):
        os.makedirs('./sens_data/prcc', exist_ok=True)
        for susc in self.susc_choices:
            for base_r0 in self.r0_choices:
                filename = f'{susc}_{base_r0}'
                lhs_table = np.loadtxt(f'./sens_data/lhs/lhs_{filename}.csv', delimiter=';')
                sim_output = np.loadtxt(f'./sens_data/simulations/simulations_{filename}.csv', delimiter=';')

                prcc = get_prcc_values(np.c_[lhs_table, sim_output.T])
                np.savetxt(fname=f'./sens_data/prcc/prcc_{filename}.csv', X=prcc)

    def plot_prcc(self):
        os.makedirs('../sens_data/plots', exist_ok=True)
        for susc in self.susc_choices:
            for base_r0 in self.r0_choices:
                filename = f'{susc}_{base_r0}'
                prcc = np.loadtxt(fname=f'./sens_data/prcc/prcc_{filename}.csv')

                generate_prcc_plot(params=self.data.param_names,
                                   target_var=self.target_var,
                                   prcc=prcc,
                                   filename=filename)

    def _get_initial_config(self):
        self.params = self.data.model_parameters_data
        self.no_ag = self.data.contact_data["home"].shape[0]
        self.model = VaccinatedModel(model_data=self.data, test_bool=self.test)
        self.population = self.model.population
        self.age_vector = self.population.reshape((-1, 1))
        self.susceptibles = self.model.get_initial_values(parameters=self.params)[self.model.c_idx["s"] *
                                                            self.no_ag:(self.model.c_idx["s"] + 1) * self.no_ag]
        self.contact_matrix = self.data.contact_data["home"] + self.data.contact_data["work"] + \
            self.data.contact_data["school"] + self.data.contact_data["other"]
        self.contact_home = self.data.contact_data["home"]
        self.upper_tri_indexes = np.triu_indices(self.no_ag)

