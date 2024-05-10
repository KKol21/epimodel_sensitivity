from emsa.sensitivity.sampler_base import SamplerBase


class SamplerSEIR(SamplerBase):
    def __init__(self, sim_object, variable_params):
        super().__init__(sim_object, variable_params)

    def run(self):
        lhs_table = self.get_lhs_table()
        self.get_sim_output(lhs_table)
