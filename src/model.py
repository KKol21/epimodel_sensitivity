from functools import partial
# from torchdiffeq import odeint

from model_base import EpidemicModelBase
from eqns_generator import EquationGenerator

import numpy as np


class VaccinatedModel(EpidemicModelBase):
    def __init__(self, model_data):
        self.n_state_comp = ["e", "i", "h", "ic", "icr", "v"]
        compartments = ["s"] + self.get_n_compartments(model_data.model_parameters_data) + ["r", "d"]
        super().__init__(model_data=model_data, compartments=compartments)
        self.pop_diff = 0

    def get_model(self, xs, ts, ps, cm):
        val = xs.reshape(-1, self.n_age)
        n_state_val = self.get_n_state_val(ps, val)
        # the same order as in self.compartments!
        s = val[0]
        r, d = val[-2:]

        i = np.sum([i_state for i_state in n_state_val["i"]], axis=0)
        transmission = ps["beta"] * np.array(i).dot(cm)
        actual_population = self.population
        vacc = self.get_vacc_bool(ts, ps)

        eq_generator = EquationGenerator(n_state_val=n_state_val, ps=ps, actual_population=actual_population,
                                         transmission=transmission, vacc=vacc, s=s, r=r)
        model_eq_dict = eq_generator.get_eqns()
        return self.get_array_from_dict(comp_dict=model_eq_dict)

    @staticmethod
    def get_n_states(n_classes, comp_name):
        return [f"{comp_name}_{i}" for i in range(n_classes)]

    def get_n_compartments(self, params):
        compartments = []
        for comp in self.n_state_comp:
            compartments.append(self.get_n_states(comp_name=comp, n_classes=params[f'n_{comp}_states']))
        return [state for n_comp in compartments for state in n_comp]

    def get_n_state_val(self, ps, val):
        n_state_val = dict()
        slice_start = 1
        slice_end = 1
        for comp in self.n_state_comp:
            n_states = ps[f'n_{comp}_states']
            slice_end += n_states
            n_state_val[comp] = val[slice_start:slice_end]
            slice_start += n_states
        return n_state_val

    @staticmethod
    def get_vacc_bool(ts, ps):
        return np.array(ps["t_start"] < ts < (ps["t_start"] + ps["T"])).astype(float)

    def update_initial_values(self, iv, parameters):
        e_states = self.get_n_states(n_classes=parameters["n_e_states"], comp_name="e")
        i_states = self.get_n_states(n_classes=parameters["n_i_states"], comp_name="i")
        e = np.sum([iv[state] for state in e_states], axis=0)
        i = np.sum([iv[state] for state in i_states], axis=0)
        iv.update({
            "s": self.population - (e + i)
        })

    def get_solution_torch(self, t, parameters, cm):
        initial_values = self.get_initial_values(parameters)
        model = partial(self.get_model, ps=parameters, cm=cm)
        return None  # np.array(odeint(model, t, initial_values))
