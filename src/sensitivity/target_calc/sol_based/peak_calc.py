import torch

from src.sensitivity.target_calc.sol_based.target_calc_base import TargetCalcBase


class PeakCalculator(TargetCalcBase):
    def __init__(self, model):
        super().__init__(model)

    def stopping_condition(self, **kwargs):
        # solutions.shape = (len(indices), t_limit, n_comp)
        comp_idx = self.model.idx(f'{kwargs["comp"]}_0')
        sol = kwargs["solutions"]
        last_val = sol[:, -1, :]  # solutions.shape = (len(indices), t_limit, n_comp)
        finished = (sol[:, -2, comp_idx] - last_val[:, comp_idx]).sum(axis=1) > 0
        return finished, last_val

    def metric(self, sol, comp):
        comp_max = torch.stack(
            [torch.max(self.model.aggregate_by_age(solution=sol[i, :, :],
                                                   comp=comp))
             for i in range(sol.shape[0])]
        ).to(self.model.device)
        return comp_max
