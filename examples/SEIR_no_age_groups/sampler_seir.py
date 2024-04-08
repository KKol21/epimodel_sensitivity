from src.sensitivity.sampler_base import SamplerBase


class SamplerSEIR(SamplerBase):
    def __init__(self, sim_obj, variable_params):
        super().__init__(sim_obj, variable_params)
        self.sim_obj = sim_obj

    def run(self):
        lhs_table = self._get_lhs_table()
        self._get_sim_output(lhs_table)
