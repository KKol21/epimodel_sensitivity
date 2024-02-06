from abc import ABC, abstractmethod
from tqdm import tqdm
from time import time


import torch


class TargetCalcBase(ABC):
    def __init__(self, model):
        self.model = model

    def get_output(self, lhs_table: torch.Tensor, batch_size: int, target_var: str) -> torch.Tensor:
        device = self.model.device
        model = self.model

        n_samples = lhs_table.shape[0]
        indices = torch.IntTensor(range(0, n_samples)).to(device)
        comp = target_var.split('_')[0]
        output = torch.zeros(n_samples, device=device)

        t_limit = [0, 500]
        y0 = torch.stack(
            [model.get_initial_values()] * n_samples
        ).to(device)
        time_start = time()
        # Iterate until all the eqs are solved or we reach t=5000
        while indices.numel() and t_limit[1] < 5000:
            t_eval = torch.stack(
                [torch.arange(*t_limit)] * len(indices)
            ).to(self.model.device)
            ind_to_keep = []
            for batch_idx in tqdm(range(0, len(indices), batch_size),
                                  leave=False,
                                  desc=f"\n Batches solved, time limit: {t_limit[1]}, samples left: {indices.numel()}"):
                batch_slice = slice(batch_idx, batch_idx + batch_size)
                curr_indices = indices[batch_slice]
                batch = lhs_table[curr_indices]

                solutions = self.get_batch_solution(y0=y0[curr_indices], t_eval=t_eval[batch_slice], samples=batch)
                # Check which simulations have finished
                finished, last_val = self.stopping_condition(solutions=solutions, comp=comp)

                if any(finished):
                    output[curr_indices[finished]] = self.metric(sol=solutions[finished],
                                                                 comp=comp)
                # Save the last values of unfinished simulations to use as initial values in the next iteration
                y0[curr_indices[~finished]] = last_val[~finished]
                ind_to_keep += curr_indices[~finished]
            # Adjust time period
            t_limit[0] = t_limit[1]
            t_limit[1] += 50
            indices = indices[torch.isin(indices,
                                         torch.Tensor(ind_to_keep).to(device))]
        print("\n Elapsed time: ", time() - time_start)
        return output

    def get_batch_solution(self, y0: torch.Tensor, t_eval: torch.Tensor, samples: torch.Tensor) -> torch.Tensor:
        sol = self.model.get_solution(y0=y0, t_eval=t_eval, lhs_table=samples).ys
        if self.model.test:
            # Check if population size changed
            if any([abs(self.model.population.sum() - sol[i, -1, :].sum()) > 50 for i in range(sol.shape[0])]):
                raise Exception("Unexpected change in population size!")
        return sol

    @abstractmethod
    def stopping_condition(self, **kwargs) -> tuple[torch.BoolTensor, torch.Tensor]:
        pass

    @abstractmethod
    def metric(self, sol: torch.Tensor, comp: str) -> torch.Tensor:
        pass
