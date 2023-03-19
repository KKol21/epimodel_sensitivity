from functools import partial

import torch

from model_base import EpidemicModelBase
from eqns_generator import EquationGenerator

import numpy as np
from torchdiffeq import odeint


class VaccinatedModel(EpidemicModelBase):
    def __init__(self, model_data):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_state_comp = ["e", "i", "h", "ic", "icr", "v"]
        compartments = ["s"] + self.get_n_compartments(model_data.model_parameters_data) + ["r", "d"]
        super().__init__(model_data=model_data, compartments=compartments)

    def get_model(self, ts, xs, ps, cm):

        val = xs.reshape(-1, self.n_age)
        n_state_val = self.get_n_state_val(ps, val)
        # the same order as in self.compartments!
        s = val[0]
        r, d = val[-2:]

        i = torch.stack([i_state for i_state in n_state_val["i"]]).sum(0)
        transmission = ps["beta"] * i.matmul(cm)
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
        e = torch.stack([iv[state] for state in e_states]).sum(0)
        i = torch.stack([iv[state] for state in i_states]).sum(0)
        iv.update({
            "s": self.population - (e + i)
        })

    def get_solution_torch(self, t, parameters, cm):
        initial_values = self.get_initial_values(parameters)
        model_wrapper = ModelFun(self).to(self.device)
        model = partial(model_wrapper.forward, ps=parameters, cm=cm)
        return odeint(model, initial_values, t, method="euler")


class ModelFun(torch.nn.Module):
    def __init__(self, model):
        super(ModelFun, self).__init__()
        self.model = model

    def forward(self, ts, xs, ps, cm):
        return self.model.get_model(ts, xs, ps, cm)
