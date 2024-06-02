import torch
import numpy as np

from emsa.sensitivity.target_calc.r0calculator import R0Calculator
from emsa.sensitivity.target_calc.sol_based_target_calc import TargetCalc


class OutputGenerator:
    def __init__(self, sim_object, variable_params: dict):
        self.batch_size = sim_object.batch_size
        self.sim_object = sim_object
        self.variable_params = variable_params

    def get_output(self, lhs_table: np.ndarray) -> dict[str: torch.Tensor]:
        lhs = torch.from_numpy(lhs_table).float().to(self.sim_object.device)
        targets = self.sim_object.target_vars
        output = {}
        target_endings = [target[-3:] for target in targets if target != "r0"]

        if "max" in target_endings or "sup" in target_endings:
            target_calc = TargetCalc(model=self.sim_object.model, targets=targets)
            sol_based_output = target_calc.get_output(lhs_table=lhs, batch_size=self.batch_size)
            output.update(sol_based_output)

        if "r0" in targets:
            r0calc = R0Calculator(self.sim_object)
            r0s = r0calc.get_output(lhs_table=lhs)
            output["r0"] = r0s
        return output
