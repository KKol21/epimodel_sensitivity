import numpy as np
import torch

from src.sensitivity.target_calc.sol_based.final_size_calc import FinalSizeCalculator
from src.sensitivity.target_calc.sol_based.peak_calc import PeakCalculator
from src.sensitivity.target_calc.r0calculator import R0Calculator


class OutputGenerator:
    def __init__(self, sim_obj, variable_params):
        self.batch_size = sim_obj.batch_size
        self.sim_obj = sim_obj
        self.variable_params = variable_params

    def get_output(self, lhs_table: np.ndarray) -> dict:
        lhs = torch.from_numpy(lhs_table).float().to(self.sim_obj.device)
        results = {}
        for target in self.sim_obj.target_vars:
            if target == "r0":
                r0calc = R0Calculator(self.sim_obj)
                results[target] = r0calc.get_output(lhs_table=lhs)
            else:
                results[target] = self.calculate_target(lhs_table=lhs, target_var=target)
        return results

    def calculate_target(self, lhs_table: torch.Tensor, target_var: str) -> torch.Tensor:
        if target_var.split('_')[1] == "sup":
            target_calculator = FinalSizeCalculator(model=self.sim_obj.model)
        else:
            target_calculator = PeakCalculator(model=self.sim_obj.model)
        return target_calculator.get_output(lhs_table=lhs_table,
                                            batch_size=self.batch_size,
                                            target_var=target_var)
