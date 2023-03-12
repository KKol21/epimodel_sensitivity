from time import sleep

import numpy as np
from smt.sampling_methods import LHS
from tqdm import tqdm

from sampler_base import SamplerBase


class SamplerVaccinated(SamplerBase):
    def __init__(self, sim_state: dict, sim_obj):
        super().__init__(sim_state, sim_obj)
        self.sim_obj = sim_obj
        self.susc = sim_state["susc"]
        self.lhs_boundaries = {"lower": [0.1, 0.1, None, 0, 0, 0.1, 0.1],  # alpha, gamma,  beta_0, daily vaccines,
                               "upper": [1, 1, None, 1, 100, 1, 1],  # t_start, rho, psi
                               }
        self.lhs_table = None
        self.sim_output = None
        self.param_names = self.sim_obj.data.param_names

        self.get_beta_0_boundaries()

    def run_sampling(self):
        n_samples = 1000
        bounds = np.array([bounds for bounds in self.lhs_boundaries.values()]).T
        sampling = LHS(xlimits=bounds)
        lhs_table = sampling(n_samples)
        print("Simulation for", n_samples,
              "samples (", "-".join(self._get_variable_parameters()), ")")

        target_var = self.sim_state["target_var"]
        if target_var == "r0":
            get_output = self.get_r0
        elif target_var == "i_max":
            get_output = self.get_i_max
        elif target_var == "ic_max":
            get_output = self.get_ic_max
        elif target_var == "d_max":
            get_output = self.get_d_max

        results = list(tqdm(map(get_output, lhs_table), total=lhs_table.shape[0]))
        results = np.array(results)
        # Sort tables by R0 values
        sorted_idx = results.argsort()
        results = results[sorted_idx]
        lhs_table = np.array(lhs_table[sorted_idx])
        sim_output = np.array(results)
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

    def get_i_max(self, params):
        return self.get_max(params, 'i')

    def get_ic_max(self, params):
        return self.get_max(params, 'ic')

    def get_d_max(self, params):
        return self.get_max(params, 'd')

    def get_max(self, params, comp):
        params_dict = {key: value for (key, value) in zip(self.param_names, params)}
        parameters = self.sim_obj.params
        parameters.update(params_dict)
        parameters['v'] = parameters['v'] * parameters['daily_vaccines']
        t = np.linspace(1, 300, 300)
        sol = self.sim_obj.model.get_solution(t=t, parameters=parameters, cm=self.sim_obj.contact_matrix)

        if comp in ["i", "ic", "e"]:
            n_states = parameters[f"n_{comp}_states"]
            idx_start = self.sim_obj.model.n_age * (self.sim_obj.model.c_idx[f"{comp}_0"])
        else:
            n_states = 1
            idx_start = self.sim_obj.model.n_age * self.sim_obj.model.c_idx[comp]

        comp_max = np.max(self.sim_obj.model.aggregate_by_age(solution=sol, idx=idx_start, n_states=n_states))
        return comp_max

    def _get_variable_parameters(self):
        return f'{self.susc}_{self.base_r0}'

    def get_beta_0_boundaries(self):
        for bound in ["lower", "upper"]:
            self.r0generator.parameters.update({"alpha": self.lhs_boundaries[bound][0],
                                                "gamma": self.lhs_boundaries[bound][1]})
            beta_0 = self.base_r0 / self.r0generator.get_eig_val(contact_mtx=self.sim_obj.contact_matrix,
                                                                 susceptibles=self.sim_obj.susceptibles.reshape(1, -1),
                                                                 population=self.sim_obj.population)[0]
            self.lhs_boundaries[bound][2] = beta_0
