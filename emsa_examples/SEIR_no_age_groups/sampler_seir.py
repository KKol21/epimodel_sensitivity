from emsa.sensitivity.sampler_base import SamplerBase


class SamplerSEIR(SamplerBase):
    def __init__(self, sim_obj, variable_params):
        super().__init__(sim_obj, variable_params)

    def run(self):
        lhs_table = self.get_lhs_table()
        self.get_sim_output(lhs_table)
