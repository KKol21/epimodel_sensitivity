import torch
import numpy as np
from torchdiffeq import odeint
import time

from src.model.eqns_generator import EquationGenerator
from src.model.model_base import EpidemicModelBase


class VaccinatedModel(EpidemicModelBase):
    def __init__(self, model_data):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_state_comp = ["e", "i", "h", "ic", "icr", "v"]
        compartments = ["s"] + self.get_n_compartments(model_data.model_parameters_data) + ["r", "d"]
        super().__init__(model_data=model_data, compartments=compartments)
        self.time_ = 0
        self.eq_solver = EquationGenerator(ps=model_data.model_parameters_data,
                                           actual_population=self.population)

    def get_model(self, ts, xs, ps, cm):

        val = xs.reshape(-1, self.n_age)
        n_state_val = self.get_n_state_val(ps, val)
        # the same order as in self.compartments!
        s = val[0]
        r, d = val[-2:]

        i = torch.stack([i_state for i_state in n_state_val["i"]]).sum(0)
        transmission = ps["beta"] * i.matmul(cm)
        vacc = self.get_vacc_bool(ts, ps)

        start = time.time()
        model_eq = self.eq_solver.evaluate_eqns(n_state_val=n_state_val, s=s, r=r,
                                                transmission=transmission, vacc=vacc)
        self.time_ += time.time() - start
        return torch.cat(tuple(model_eq))

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
        return int(ps["t_start"] < ts < (ps["t_start"] + ps["T"]))

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
        model_wrapper = ModelFun(self, parameters, cm).to(self.device)
        return odeint(model_wrapper.forward, initial_values, t, method='euler')

    def get_solution_torch_test(self, t, parameters, cm):
        initial_values = torch.cat([self.population - torch.ones(self.n_age).to(self.device),
                                    torch.zeros(self.n_age).to(self.device),
                                    torch.ones(self.n_age).to(self.device),
                                    torch.zeros(self.n_age).to(self.device)]).to(self.device)
        model_wrapper = ModelFunTest(self, parameters, cm).to(self.device)
        return odeint(model_wrapper.forward, initial_values, t, method='euler')


class ModelFun(torch.nn.Module):
    """
    Wrapper class for VaccinatedModel.get_model. Inherits from torch.nn.Module, enabling
    the use of a GPU for evaluation through the library torchdiffeq.
    """
    def __init__(self, model, ps, cm):
        super(ModelFun, self).__init__()
        self.model = model
        self.ps = ps
        self.cm = cm

    def forward(self, ts, xs):
        return self.model.get_model(ts, xs, self.ps, self.cm)


class ModelFunTest(torch.nn.Module):
    """
    Wrapper class for VaccinatedModel.get_model. Inherits from torch.nn.Module, enabling
    the use of a GPU for evaluation through the library torchdiffeq.
    """
    def __init__(self, model, ps, cm):
        super(ModelFunTest, self).__init__()
        self.model = model
        self.ps = ps
        self.cm = cm

    def forward(self, ts, xs):
        ps = self.ps
        s, e, i, r = xs.reshape(-1, self.model.n_age)

        transmission = 0.1 * i.matmul(self.cm)
        actual_population = self.model.population

        model_eq = [-ps["susc"] * (s / actual_population) * transmission,  # S'(t)
                     ps["susc"] * (s / actual_population) * transmission - ps["alpha"] * e,  # E'(t)
                     ps["alpha"] * e - ps["gamma"] * i,  # I'(t)
                     ps["gamma"] * i]  # R'(t)
        return torch.cat(tuple(model_eq))



