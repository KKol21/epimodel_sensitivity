from src.sensitivity.sampler_base import SamplerBase


class SamplerSEIR(SamplerBase):
    def __init__(self, sim_state: dict, sim_obj):
        super().__init__(sim_state, sim_obj)
        self.sim_obj = sim_obj
        self.susc = sim_state["susc"]
        self.base_r0 = sim_state["base_r0"]
        self.target_var = sim_state["target_var"]
        self.lhs_boundaries = {"lower": [0.1, 0.2, 0.1],  # alpha, beta, gamma
                               "upper": [0.4, 0.4, 0.4],
                               }

    def run_sampling(self):
        lhs_table = self._get_lhs_table()

        self._get_sim_output(lhs_table)
        self.calculate_r0s(lhs_table)

    def calculate_r0s(self, lhs_table):
        from src.model.r0 import R0Generator

        r0s = []
        r0gen = R0Generator(self.sim_obj.data)
        for beta in lhs_table[:, 1]:
            r0gen.params.update({"beta": beta})
            r0s.append(r0gen.get_eig_val(susceptibles=self.sim_obj.susceptibles,
                                         population=self.sim_obj.data.age_data,
                                         contact_mtx=self.sim_obj.cm))

    def _get_variable_parameters(self):
        return f'{self.susc}-{self.base_r0}-{self.target_var}'
