import numpy as np
import torch

from src.sensitivity.target_calc.final_size_calc import FinalSizeCalculator
from src.sensitivity.target_calc.peak_calc import PeakCalculator


class OutputGenerator:
    def __init__(self, sim_obj, variable_params):
        self.batch_size = sim_obj.batch_size
        self.sim_obj = sim_obj
        self.variable_params = variable_params

    def get_output(self, lhs_table: np.ndarray) -> dict:
        lhs = torch.from_numpy(lhs_table).float().to(self.sim_obj.device)
        results = {}
        for target in self.sim_obj.target_vars:
            results[target] = self.calculate_target(lhs, target)
        return results

    def calculate_target(self, lhs_table: torch.Tensor, target_var: str) -> torch.Tensor:
        if target_var.split('_')[1] == "sup":
            target_calculator = FinalSizeCalculator(self.sim_obj.model)
        else:
            target_calculator = PeakCalculator(self.sim_obj.model)
        return target_calculator.get_output(lhs_table=lhs_table,
                                            batch_size=self.batch_size,
                                            target_var=target_var)
