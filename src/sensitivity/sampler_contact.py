import numpy as np
import torch

from src.sensitivity.sampler_base import SamplerBase


class SamplerContact(SamplerBase):
    def __init__(self, sim_state: dict, sim_obj):
        super().__init__(sim_state, sim_obj)
        self.mode = "contact"
        self.sim_obj = sim_obj
        self.susc = sim_state["susc"]
        self.target_var = sim_state["target_var"]

        self.lhs_boundaries = {
            "lower": np.zeros(self.sim_obj.upper_tri_size),
            "upper": np.ones(self.sim_obj.upper_tri_size)
                               }
        self.batch_size = 1000

    def run_sampling(self):
        lhs_table = self._get_lhs_table()
        self._get_sim_output(lhs_table)

    def _get_variable_parameters(self):
        return f'{self.susc}-{self.base_r0}-{self.target_var}'
