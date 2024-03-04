import torch

from src.sensitivity.target_calc.final_size_calc import FinalSizeCalculator
from src.sensitivity.target_calc.peak_calc import PeakCalculator


class OutputGenerator:
    def __init__(self, sim_obj, target):
        self.batch_size = sim_obj.batch_size
        self.target_var = target
        self.sim_obj = sim_obj

    def get_output(self, lhs_table):
        if self.target_var in ["d_max", "r_max"]:
            target_calculator = FinalSizeCalculator(self.sim_obj.model)
        else:
            target_calculator = PeakCalculator(self.sim_obj.model)
        lhs = torch.from_numpy(lhs_table).float().to(self.sim_obj.device)
        return target_calculator.get_output(lhs_table=lhs,
                                            batch_size=self.batch_size,
                                            target_var=self.target_var)
