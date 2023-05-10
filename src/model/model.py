import torch
from torchdiffeq import odeint

from src.model.model_base import EpidemicModelBase, get_n_states


class VaccinatedModel(EpidemicModelBase):
    def __init__(self, model_data, cm):
        self.device = model_data.device
        self.n_state_comp = ["e", "i", "h", "ic", "icr"]
        compartments = ["s"] + self.get_n_compartments(model_data.model_parameters_data) + ["v_0", "r", "d"]
        super().__init__(model_data=model_data, compartments=compartments)

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

    def get_solution_torch(self, t, cm, daily_vac):
        initial_values = self.get_initial_values_()
        model_wrapper = ModelEq(self,  cm, daily_vac).to(self.device)
        return odeint(model_wrapper, initial_values, t, method='euler')

    def get_initial_values_(self):
        size = self.n_age * len(self.compartments)
        iv = torch.zeros(size)
        idx = 3 * self.n_comp
        iv[idx + self.c_idx['e_0']] = 1
        iv[self.c_idx['s']:size:self.n_comp] = self.population
        iv[idx + self.c_idx['s']] -= 1
        return iv


class ModelEq(torch.nn.Module):
    def __init__(self, model: VaccinatedModel, cm: torch.Tensor, daily_vac):
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
