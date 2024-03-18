from src.sensitivity.sampler_base import SamplerBase


class SamplerSEIR(SamplerBase):
    def __init__(self, sim_obj, sim_option):
        super().__init__(sim_obj, sim_option)
        self.sim_obj = sim_obj
        self.susc = next(iter(sim_option["susc"].values()))
        self.base_r0 = sim_option["r0"]

    def run(self):
        lhs_table = self._get_lhs_table()
        self._get_sim_output(lhs_table)
        self.calculate_r0s(lhs_table)

    def calculate_r0s(self, lhs_table):
        from src.model.r0 import R0Generator

        r0s = []
        r0gen = R0Generator(self.sim_obj.data, **self.sim_obj.model_struct)
        for beta in lhs_table[:, 1]:
            r0gen.params.update({"beta": beta})
            r0s.append(r0gen.get_eig_val(susceptibles=self.sim_obj.susceptibles,
                                         population=self.sim_obj.data.age_data,
                                         contact_mtx=self.sim_obj.cm))
