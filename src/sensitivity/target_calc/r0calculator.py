from src.model.r0 import R0Generator


class R0Calculator:
    def __init__(self, sim_obj):
        self.sim_obj = sim_obj


    def get_output(self):
        r0gen = R0Generator(self.data, self.)
