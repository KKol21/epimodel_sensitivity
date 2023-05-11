from functools import partial
from time import sleep

import numpy as np
from smt.sampling_methods import LHS
import torch
from tqdm import tqdm

from src.sensitivity.sampler_base import SamplerBase


class SamplerVaccinated(SamplerBase):
    def __init__(self, sim_state: dict, sim_obj):
        super().__init__(sim_state, sim_obj)
        self.n_samples = 10000
        self.sim_obj = sim_obj
        self.susc = sim_state["susc"]
        self.target_var = sim_state["target_var"]
        self.lhs_boundaries = {"lower": np.zeros(sim_obj.n_age),    # Ratio of daily vaccines given to each age group
                               "upper": np.ones(sim_obj.n_age)
                               }
        self.lhs_table = None
        self.sim_output = None
        self.param_names = self.sim_obj.data.param_names
        self.min_target = 1E10
        self.optimal_vacc = None

    @staticmethod
    def norm_table_rows(table):
        return table / np.sum(table, axis=1, keepdims=True)

    def run_sampling(self):
        bounds = np.array([bounds for bounds in self.lhs_boundaries.values()]).T
        sampling = LHS(xlimits=bounds)
        lhs_table = sampling(self.n_samples)
        # Make sure that total vaccines given to an age group
        # doesn't exceed the population of that age group
        lhs_table = self.allocate_vaccines(lhs_table).to(self.sim_obj.data.device)
        print("Simulation for", self.n_samples,
              "samples (", self._get_variable_parameters(), ")")
        if self.target_var == "r0":
            get_output = self.get_r0
        else:
            get_output = partial(self.get_max, comp=self.target_var.split('_')[0])
        self.sim_obj.model.get_constant_matrices()
        results = list(tqdm(map(get_output, lhs_table), total=lhs_table.shape[0]))
        results = torch.tensor(results)
        # Sort tables by R0 values
        sorted_idx = results.argsort()
        results = results[sorted_idx]
        lhs_table = lhs_table[sorted_idx]
        sim_output = results
        sleep(0.3)

        self._save_output(output=lhs_table, folder_name='lhs')
        self._save_output(output=sim_output, folder_name='simulations')
        self._save_output(output=self.optimal_vacc, folder_name='optimal_vaccination')

    def get_r0(self, params):
        params_dict = {key: value for (key, value) in zip(self.param_names, params)}
        self.r0generator.parameters.update(params_dict)
        r0_lhs = params[2] * self.r0generator.get_eig_val(contact_mtx=self.sim_obj.contact_matrix,
                                                          susceptibles=self.sim_obj.susceptibles.reshape(1, -1),
                                                          population=self.sim_obj.population)[0]
        return r0_lhs

    def get_max(self, vaccination_sample, comp):
        parameters = self.sim_obj.params
        r0 = self.sim_state["base_r0"]
        is_erlang = self.sim_obj.distr == "erlang"
        if r0 == 1.8:
            len = 1200 if is_erlang else 1000
        elif r0 == 2.4:
            len = 800 if is_erlang else 350
        elif r0 == 3:
            len = 500 if is_erlang else 250
        t = torch.linspace(1, len, len).to(self.sim_obj.data.device)
        daily_vac = vaccination_sample * parameters["total_vaccines"] / parameters["T"]
        sol = self.sim_obj.model.get_solution(t=t, cm=self.sim_obj.contact_matrix, daily_vac=daily_vac)
        if self.sim_obj.test:
            # Check if population size changed
            if abs(self.sim_obj.population.sum() - sol[-1, :].sum()) > 100:
                raise Exception("Unexpected change in population size!")

        comp_sol = self.sim_obj.model.aggregate_by_age(solution=sol, comp=comp)
        comp_max_ = torch.max(comp_sol)
        if comp_max_ < self.min_target:
            self.min_target = comp_max_
            self.optimal_vacc = daily_vac
            self.sim_obj.model.aggregate_by_age(sol, 'd').max()
        return comp_max_

    def _get_variable_parameters(self):
        return f'{self.susc}-{self.base_r0}-{self.target_var}-{self.sim_obj.distr}'

    def allocate_vaccines(self, lhs_table):
        lhs_table = self.norm_table_rows(lhs_table)
        params = self.sim_obj.params
        total_vac = params["total_vaccines"] * lhs_table
        population = np.array(self.sim_obj.population.cpu())

        while np.any(total_vac > population):
            mask = total_vac > population
            excess = population - total_vac
            redistribution = excess * lhs_table

            total_vac[mask] = np.tile(population, (lhs_table.shape[0], 1))[mask]
            total_vac[~mask] += redistribution[~mask]

            lhs_table = self.norm_table_rows(total_vac / params['total_vaccines'])
            total_vac = params["total_vaccines"] * lhs_table
        return torch.Tensor(lhs_table)
