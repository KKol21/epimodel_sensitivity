import numpy as np

from src.sensitivity.sampler_base import SamplerBase


class SamplerContact(SamplerBase):
    def __init__(self, sim_obj, variable_params):
        super().__init__(sim_obj, variable_params)

        self.lhs_bounds_dict = {
            "contacts": np.array([np.full(fill_value=0.1, shape=self.sim_obj.upper_tri_size),
                                  np.full(fill_value=1, shape=self.sim_obj.upper_tri_size)]),
        }

    def run(self):
        lhs_table = self._get_lhs_table()
        self._get_sim_output(lhs_table)
