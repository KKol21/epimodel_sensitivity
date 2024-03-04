import torch

from src.sensitivity.target_calc.final_size_calc import FinalSizeCalculator
from src.sensitivity.target_calc.peak_calc import PeakCalculator


class OutputGenerator:
    def __init__(self, sim_obj, sim_option):
        self.batch_size = sim_obj.batch_size
        self.sim_obj = sim_obj
        self.sim_option = sim_option

    def get_output(self, lhs_table):
        lhs_np = torch.from_numpy(lhs_table).float().to(self.sim_obj.device)
        results = {}
        for target in self.sim_obj.target_vars:
            results[target] = self.calculate_target(lhs_np, target)
        return results

    def calculate_target(self, lhs_table, target_var):
        if target_var in ["d_max", "r_max"]:
            target_calculator = FinalSizeCalculator(self.sim_obj.model)
        else:
            target_calculator = PeakCalculator(self.sim_obj.model)
        return target_calculator.get_output(lhs_table=lhs_table,
                                            batch_size=self.batch_size,
                                            target_var=target_var)
