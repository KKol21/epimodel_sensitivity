import torch
from torchdiffeq import odeint

from src.model.eqns_generator import EquationGenerator
from src.model.model_base import EpidemicModelBase


def get_n_states(n_classes, comp_name):
    return [f"{comp_name}_{i}" for i in range(n_classes)]


class VaccinatedModel(EpidemicModelBase):
    def __init__(self, model_data):
        self.device = model_data.device
        self.n_state_comp = ["e", "i", "h", "ic", "icr", "v"]
        compartments = ["s"] + self.get_n_compartments(model_data.model_parameters_data) + ["r", "d"]
        super().__init__(model_data=model_data, compartments=compartments)
        self.eq_solver = EquationGenerator(ps=model_data.model_parameters_data,
                                           actual_population=self.population)

    def get_model(self, ts, xs, ps, cm):

        val = xs.reshape(-1, self.n_age)
        n_state_val = self.get_n_state_val(ps, val)
        # the same order as in self.compartments!
        s = val[0]
        r = val[-2]
        i = torch.stack([i_state for i_state in n_state_val["i"]]).sum(0)
        transmission = ps["beta"] * i.matmul(cm)
        vacc = self.get_vacc_bool(ts, ps)

        model_eq = self.eq_solver.evaluate_eqns(n_state_val=n_state_val, s=s, r=r,
                                                transmission=transmission, vacc=vacc)
        return torch.cat(tuple(model_eq))

    def get_n_compartments(self, params):
        compartments = []
        for comp in self.n_state_comp:
            compartments.append(get_n_states(comp_name=comp, n_classes=params[f'n_{comp}']))
        return [state for n_comp in compartments for state in n_comp]

    def get_n_state_val(self, ps, val):
        n_state_val = dict()
        slice_start = 1
        slice_end = 1
        for comp in self.n_state_comp:
            n_states = ps[f'n_{comp}']
            slice_end += n_states
            n_state_val[comp] = val[slice_start:slice_end]
            slice_start += n_states
        return n_state_val

    def update_initial_values(self, iv, parameters):
        iv["e_0"][0] = 1
        e_states = get_n_states(n_classes=parameters["n_e"], comp_name="e")
        i_states = get_n_states(n_classes=parameters["n_i"], comp_name="i")
        e = torch.stack([iv[state] for state in e_states]).sum(0)
        i = torch.stack([iv[state] for state in i_states]).sum(0)
        iv.update({
            "s": self.population - (e + i)
        })

    def get_solution_torch(self, t, parameters, cm):
        initial_values = self.get_initial_values(parameters)
        model_wrapper = ModelFun(self, parameters, cm).to(self.device)
        return odeint(model_wrapper, initial_values, t, method='euler')

    @staticmethod
    def get_vacc_bool(ts, ps):
        return int(ps["t_start"] < ts < (ps["t_start"] + ps["T"]))


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


class VaccinatedModel2(EpidemicModelBase):
    def __init__(self, model_data, cm):
        self.device = model_data.device
        self.n_state_comp = ["e", "i", "h", "ic", "icr", "v"]
        compartments = ["s"] + self.get_n_compartments(model_data.model_parameters_data) + ["r", "d"]
        super().__init__(model_data=model_data, compartments=compartments)
        self.n_comp = len(self.compartments)

        from src.model.matrix_generator import MatrixGenerator
        self.matrix_generator = MatrixGenerator(model=self, cm=cm, ps=self.ps)

    def update_initial_values(self, iv, parameters):
        pass

    def get_model(self, ts, xs, ps, cm):
        pass

    def get_constant_matrices(self):
        mtx_gen = self.matrix_generator
        self.A = mtx_gen.get_A()
        self.T = mtx_gen.get_T()
        self.B = mtx_gen.get_B()
        self.V_2 = mtx_gen.get_V_2()

    def get_n_compartments(self, params):
        compartments = []
        for comp in self.n_state_comp:
            compartments.append(get_n_states(comp_name=comp, n_classes=params[f'n_{comp}']))
        return [state for n_comp in compartments for state in n_comp]

    def get_solution_torch_test(self, t, cm, daily_vac):
        initial_values = self.get_initial_values_()
        model_wrapper = ModelEq(self,  cm, daily_vac).to(self.device)
        return odeint(model_wrapper, initial_values, t, method='euler')

    def get_initial_values_(self):
        size = self.n_age * len(self.compartments)
        iv = torch.zeros(size)
        iv[self.c_idx['e_0']] = 1
        iv[self.c_idx['s']:size:self.n_comp] = self.population
        iv[0] -= 1
        return iv

    def idx(self, state: str) -> bool:
        return torch.arange(self.n_age * self.n_comp) % self.n_comp == self.c_idx[state]

    def aggregate_by_age_n_state(self, solution, comp):
        result = 0
        for state in get_n_states(self.ps[comp], comp):
            result += solution[-1, self.idx(state)].sum()
        return result

    def aggregate_by_age_(self, solution, comp):
        return solution[-1, self.idx(comp)].sum()


class ModelEq(torch.nn.Module):
    def __init__(self, model: VaccinatedModel2, cm: torch.Tensor, daily_vac):
        super(ModelEq, self).__init__()
        self.model = model
        self.cm = cm
        self.ps = model.ps
        self.device = model.device

        self.matrix_generator = self.model.matrix_generator
        self.V_1 = self.matrix_generator.get_V_1(daily_vac)

    # For increased efficiency, we represent the ODE system in the form
    # y' = (A @ y) * (T @ y) + B @ y + (V_1 * y) / (V_2 @ y),
    # saving every tensor in the module state
    def forward(self, t, y: torch.Tensor) -> torch.Tensor:
        vacc = torch.zeros(self.matrix_generator.s_mtx)
        if self.get_vacc_bool(t):
            vacc = torch.div(self.V_1 @ y, self.model.V_2 @ y)
        return torch.mul(self.model.A @ y, self.model.T @ y) + self.model.B @ y + vacc

    def get_vacc_bool(self, t) -> int:
        return self.ps["t_start"] < t < (self.ps["t_start"] + self.ps["T"])
