import numpy as np

from src.sensitivity.sampler_base import SamplerBase


class SamplerContact(SamplerBase):
    def __init__(self, sim_obj, sim_option):
        super().__init__(sim_obj, sim_option)
        self.sim_obj = sim_obj

        self.lhs_boundaries = {
            "contacts": np.array([np.full(fill_value=0.1, shape=self.sim_obj.upper_tri_size),
                                  np.full(fill_value=1, shape=self.sim_obj.upper_tri_size)]),
        }

    def run_sampling(self):
        lhs_table = self._get_lhs_table()
        self._get_sim_output(lhs_table)
