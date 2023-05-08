from functools import partial
from time import sleep, time

import numpy as np
from smt.sampling_methods import LHS
import torch
from tqdm import tqdm

from src.sensitivity.sampler_base import SamplerBase


class SamplerVaccinated(SamplerBase):
    def __init__(self, sim_state: dict, sim_obj):
        super().__init__(sim_state, sim_obj)
        self.n_samples = 100
        self.sim_obj = sim_obj
        self.susc = sim_state["susc"]
        self.lhs_boundaries = {"lower": np.zeros(sim_obj.n_age),    # Ratio of daily vaccines given to each age group
                               "upper": np.ones(sim_obj.n_age)
                               }
        self.lhs_table = None
        self.sim_output = None
        self.param_names = self.sim_obj.data.param_names

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
        target_var = self.sim_state["target_var"]
        if target_var == "r0":
            get_output = self.get_r0
        else:
            get_output = partial(self.get_max, comp=target_var.split('_')[0])
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

    def get_r0(self, params):
        params_dict = {key: value for (key, value) in zip(self.param_names, params)}
        self.r0generator.parameters.update(params_dict)
        r0_lhs = params[2] * self.r0generator.get_eig_val(contact_mtx=self.sim_obj.contact_matrix,
                                                          susceptibles=self.sim_obj.susceptibles.reshape(1, -1),
                                                          population=self.sim_obj.population)[0]
        return r0_lhs

    def get_max(self, vaccination_sample, comp):
        parameters = self.sim_obj.params
        parameters.update({'v': vaccination_sample * parameters["total_vaccines"] / parameters["T"]})

        t = torch.linspace(1, 220, 220).to(self.sim_obj.data.device)

       # start = time()
       # sol = self.sim_obj.model.get_solution_torch(t=t, parameters=parameters, cm=self.sim_obj.contact_matrix)
       # print(time()-start)
        start = time()
        daily_vac = vaccination_sample * parameters["total_vaccines"] / parameters["T"]
        sol_ = self.sim_obj.model2.get_solution_torch_test(t=t,
                                                           cm=self.sim_obj.contact_matrix,
                                                           daily_vac=daily_vac)
        #print(time() - start)
        if self.sim_obj.test:
            # Check if population size changed
            if abs(self.sim_obj.population.sum() - sol_[-1, :].sum()) > 100:
                raise Exception("Unexpected change in population size!")
            if abs(self.sim_obj.population.sum() - sol_[-1, :].sum()) > 100:
                raise Exception("Unexpected change in population size!")

        if comp in self.sim_obj.model.n_state_comp:
            n_states = parameters[f"n_{comp}"]
            idx_start = self.sim_obj.model.n_age * (self.sim_obj.model.c_idx[f"{comp}_0"])
            comp_sol = self.sim_obj.model2.aggregate_by_age_n_state(solution=sol_, comp=comp)
        else:
            n_states = 1
            idx_start = self.sim_obj.model.n_age * self.sim_obj.model.c_idx[comp]
            comp_sol = self.sim_obj.model2.aggregate_by_age_(solution=sol_, comp=comp)

       # comp_max = torch.max(self.sim_obj.model.aggregate_by_age(solution=sol, idx=idx_start, n_states=n_states))
        comp_max_ = torch.max(comp_sol)
        return comp_max_

    def _get_variable_parameters(self):
        return f'{self.susc}-{self.base_r0}-{self.sim_obj.target_var}-{self.sim_obj.is_erlang}'

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
