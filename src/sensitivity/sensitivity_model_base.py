from abc import ABC

import torch

from src.model.model_base import EpidemicModelBase


class SensitivityModelBase(EpidemicModelBase, ABC):
    def __init__(self, sim_obj):
        super().__init__(sim_obj.data)
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

    def _get_matrix_from_lhs(self, lhs_dict: dict, matrix_name: str):
        n_eq = self.n_eq
        n_samples = next(iter(lhs_dict.values())).shape[0]
        mtx = torch.zeros((n_samples, n_eq, n_eq)).to(self.device)
        for idx in range(n_samples):
            # Select idx. value from lhs table for each parameter
            row_dict = {key: value[idx] if len(value.size()) < 2 else value[idx, :]
                        for key, value in lhs_dict.items()}
            self.matrix_generator.ps.update(row_dict)

            mtx[idx, :, :] = self.matrix_generator.generate_matrix(matrix_name)
        return mtx
