import os

import numpy as np
import torch

from src.dataloader import DataLoader
from src.model.model import VaccinatedModel, VaccinatedModel2
from src.sensitivity.prcc import generate_prcc_plot, get_prcc_values
from src.model.r0 import R0Generator
from src.sensitivity.sampler_vaccinated import SamplerVaccinated


class SimulationVaccinated:
    def __init__(self):
        # Load data
        self.data = DataLoader()
        self.test = True

        # User-defined param_names
        self.susc_choices = [0.5, 1.0]
        self.r0_choices = [2.5, 5]
        self.target_var = "d_max"  # i_max, ic_max, d_max

        # Define initial configs
        self._get_initial_config()

    # Generate samples and run simulations, then save the result
    def run_sampling(self):
        susceptibility = torch.ones(self.n_age).to(self.data.device)
        for susc in self.susc_choices:
            susceptibility[:4] = susc
            self.params.update({"susc": susceptibility})
            r0generator = R0Generator(param=self.params, device=self.data.device, n_age=self.n_age)
            for base_r0 in self.r0_choices:
                # Calculate base transmission rate
                beta = base_r0 / r0generator.get_eig_val(contact_mtx=self.contact_matrix,
                                                         susceptibles=self.susceptibles.reshape(1, -1),
                                                         population=self.population)
                self.params.update({"beta": beta})
                sim_state = {"base_r0": base_r0, "susc": susc, "r0generator": r0generator,
                             "target_var": self.target_var}
                param_generator = SamplerVaccinated(sim_state=sim_state, sim_obj=self)
                param_generator.run_sampling()

    # Calculate PRCC values from saved LHS tables, then save the result
    def calculate_prcc(self):
        os.makedirs(f'../sens_data/prcc', exist_ok=True)
        for susc in self.susc_choices:
            for base_r0 in self.r0_choices:
                filename = f'{susc}-{base_r0}-{self.target_var}'
                lhs_table = np.loadtxt(f'../sens_data/lhs/lhs_{filename}.csv', delimiter=';')
                sim_output = np.loadtxt(f'../sens_data/simulations/simulations_{filename}.csv', delimiter=';')

                prcc = get_prcc_values(np.c_[lhs_table, sim_output.T])
                np.savetxt(fname=f'../sens_data/prcc/prcc_{filename}.csv', X=prcc)

    # Create and save tornado plots from sensitivity data
    def plot_prcc(self):
        os.makedirs(f'../sens_data//plots', exist_ok=True)
        for susc in self.susc_choices:
            for base_r0 in self.r0_choices:
                filename = f'{susc}-{base_r0}-{self.target_var}'
                prcc = np.loadtxt(fname=f'../sens_data/prcc/prcc_{filename}.csv')

                generate_prcc_plot(params=self.data.param_names,
                                   target_var=self.target_var,
                                   prcc=prcc,
                                   filename=filename)

    def _get_initial_config(self):
        self.params = self.data.model_parameters_data
        self.n_age = self.data.contact_data["home"].shape[0]
        self.contact_matrix = self.data.contact_data["home"] + self.data.contact_data["work"] + \
                              self.data.contact_data["school"] + self.data.contact_data["other"]
        self.model = VaccinatedModel(model_data=self.data)
        self.model2 = VaccinatedModel2(model_data=self.data, cm=self.contact_matrix)
        self.population = self.model.population
        self.age_vector = self.population.reshape((-1, 1))
        self.susceptibles = self.model.get_initial_values(parameters=self.params)[self.model.c_idx["s"] *
                                                            self.n_age:(self.model.c_idx["s"] + 1) * self.n_age]

