from abc import ABC

import torch

from src.model.model_base import EpidemicModelBase
from src.simulation_base import SimulationBase


class SensitivityModelBase(EpidemicModelBase, ABC):
    def __init__(self, sim_obj: SimulationBase):
        super().__init__(data=sim_obj.data, **sim_obj.model_struct)
        self.sim_obj = sim_obj
        self.test = sim_obj.test
        self.sim_state = None

    def get_basic_ode(self):
        A_mul = self.get_mul_method(self.A)
        T_mul = self.get_mul_method(self.T)
        B_mul = self.get_mul_method(self.B)

        def odefun(t, y):
            return torch.mul(A_mul(y, self.A), T_mul(y, self.T)) + B_mul(y, self.B)

        return odefun

    def get_vaccinated_ode(self, curr_batch_size):
        A_mul = self.get_mul_method(self.A)
        T_mul = self.get_mul_method(self.T)
        B_mul = self.get_mul_method(self.B)
        V_1_mul = self.get_mul_method(self.V_1)

        v_div = torch.ones((curr_batch_size, self.n_eq)).to(self.device)
        div_idx = self.idx('s_0') + self.idx('v_0')

        def odefun(t, y):
            base_result = torch.mul(A_mul(y, self.A), T_mul(y, self.T)) + B_mul(y, self.B)
            if self.ps["t_start"] < t[0] < self.ps["t_start"] + self.ps["T"]:
                v_div[:, div_idx] = (y @ self.V_2)[:, div_idx]
                vacc = torch.div(V_1_mul(y, self.V_1),
                                 v_div)
                return base_result + vacc
            return base_result

        return odefun

    @staticmethod
    def get_mul_method(tensor: torch.Tensor):
        def mul_by_2d(y, tensor):
            return y @ tensor

        def mul_by_3d(y, tensor):
            return torch.einsum('ij,ijk->ik', y, tensor)

        return mul_by_2d if len(tensor.size()) < 3 else mul_by_3d

    def get_matrix_from_lhs(self, lhs_dict: dict, matrix_name: str):
        n_eq = self.n_eq
        n_samples = next(iter(lhs_dict.values())).shape[0]
        mtx = torch.zeros((n_samples, n_eq, n_eq)).to(self.device)
        for idx in range(n_samples):
            # Select idx. value from lhs table for each parameter
            row_dict = {key: value[idx]
                        if len(value.size()) < 2
                        else value[idx, :]  # Select column/columns based on tensor size
                        for key, value in lhs_dict.items()}
            self.matrix_generator.ps.update(row_dict)

            mtx[idx, :, :] = self.matrix_generator.generate_matrix(matrix_name)
        return mtx

    def get_initial_values(self):
        return self.get_initial_values_from_dict(self.sim_obj.init_vals)

    def generate_3D_matrices(self, samples: torch.Tensor):
        spm = self.sim_obj.sampled_params_boundaries
        if spm is None:
            return
        linear_params = [param for param in spm
                         if param in [trans["param"] for trans in self.trans_data]]
        transmission_params_left = [param for param in spm
                                    if param in global_params.values()] \
            if (global_params := self.tms_data["global_params"]) is not None else []
        transmission_params_right = [param for param in spm
                                     if param in
                                     [param for tms_rule in self.tms_data["transmission_rules"]
                                      for param in tms_rule["actors-params"].values()]]

        params_col_idx = self.get_params_col_idx()
        def get_lhs_dict(params, lhs_table):
            return {param: lhs_table[:, params_col_idx[param]] for param in params}

        tpl_lhs = get_lhs_dict(transmission_params_left, samples)
        tpr_lhs = get_lhs_dict(transmission_params_right, samples)
        tpl_lhs.update(**tpr_lhs) # This is not the proper way,
                                  # but it works until new transmission rules are added
        lp_lhs = get_lhs_dict(linear_params, samples)
        self.A = self.get_matrix_from_lhs(tpl_lhs, "A") if len(tpl_lhs) > 0 else self.A
        self.T = self.get_matrix_from_lhs(tpr_lhs, "T") if len(tpr_lhs) > 0 else self.T
        self.B = self.get_matrix_from_lhs(lp_lhs, "B") if len(lp_lhs) > 0 else self.B

    def get_params_col_idx(self):
        params_col_idx = {}
        last_idx = -1
        for param, bound in self.sim_obj.sampled_params_boundaries.items():
            param_dim = self.sim_obj.n_age if isinstance(bound[0], list) else 1
            params_col_idx[param] = (param_idx := (last_idx + param_dim))
            last_idx = param_idx
        return params_col_idx
