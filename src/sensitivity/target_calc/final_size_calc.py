import torch


from src.sensitivity.target_calc.target_calc_base import TargetCalcBase
from src.sensitivity.sensitivity_model_base import SensitivityModelBase


class FinalSizeCalculator(TargetCalcBase):
    def __init__(self, model: SensitivityModelBase):
        super().__init__(model)

    def metric(self, sol, comp: str):
        return sol[:, -1, self.model.idx(f"{comp}_0")].sum(axis=1)

    def stopping_condition(self, **kwargs):
        last_val = kwargs["solutions"][:, -1, :]  # solutions.shape = (len(indices), t_limit, n_comp)
        comp_idx = self.model.idx("i_0")
        finished = (last_val[:, comp_idx].sum(axis=1) < 1).to(self.model.device)
        return finished, last_val
