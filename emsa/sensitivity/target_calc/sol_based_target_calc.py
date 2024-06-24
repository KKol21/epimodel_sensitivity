import math
from time import time

import torch
from typing import Dict
from emsa.sensitivity.sensitivity_model_base import SensitivityModelBase


class TargetCalc:
    def __init__(self, model: SensitivityModelBase, targets):
        self.model = model
        self.max_targets = [
            target.rsplit("_", 1)[0] for target in targets if target.endswith("max")
        ]
        self.sup_targets = [
            target.rsplit("_", 1)[0] for target in targets if target.endswith("max")
        ]

        self.max_targets_finished: Dict[str, torch.Tensor] = {}
        self.sup_finished: torch.Tensor
        self.max_targets_output: Dict[str, torch.Tensor] = {}
        self.sup_targets_output: Dict[str, torch.Tensor] = {}
        self.finished: torch.Tensor

    def get_output(self, lhs_table: torch.Tensor, batch_size: int) -> Dict[str, torch.Tensor]:
        device = self.model.device
        model = self.model

        n_samples = lhs_table.shape[0]
        self.finished = torch.zeros(n_samples, dtype=torch.bool, device=self.model.device)

        indices = torch.IntTensor(range(0, n_samples)).to(device)

        self.max_targets_output = {
            comp: torch.zeros(n_samples, device=device) for comp in self.max_targets
        }
        self.sup_targets_output = {
            comp: torch.zeros(n_samples, device=device) for comp in self.sup_targets
        }
        self.max_targets_finished = {
            comp: torch.BoolTensor(range(0, n_samples)).to(device) for comp in self.max_targets
        }
        self.sup_finished = torch.BoolTensor(range(0, n_samples)).to(device)

        t_limit = [0, 300]
        y0 = torch.stack([model.get_initial_values()] * n_samples).to(device)
        time_start = time()
        # Iterate until all the eqs are solved or we reach t=5000
        while indices.numel() and t_limit[1] < 5000:
            t_eval = torch.stack([torch.arange(*t_limit)] * len(indices)).to(self.model.device)
            ind_to_keep = []
            print(f"\n Time limit: {t_limit[1]} \n" f" Samples left: {indices.numel()} \n")
            for batch_idx in range(0, len(indices), batch_size):
                print(
                    f" Solving batch {int(batch_idx / batch_size) + 1} / {math.ceil(len(indices) / batch_size)}"
                )
                batch_slice = slice(batch_idx, batch_idx + batch_size)
                curr_indices = indices[batch_slice]
                batch = lhs_table[curr_indices]

                # Solve for the current batch
                self.model.generate_3D_matrices(
                    samples=batch
                )  # Only relevant with automatic sampling
                solutions = self.get_batch_solution(
                    y0=y0[curr_indices], t_eval=t_eval[batch_slice], samples=batch
                )
                # Save finished indices and outputs
                self.save_finished_indices(solutions=solutions, indices=curr_indices)
                self.save_output_for_finished(solutions=solutions, indices=curr_indices)

                # Save the last values and indices of unfinished simulations
                # to use as initial values in the next iteration
                true_finished = self.get_true_finished()
                last_val = solutions[:, -1, :]
                batch_unfinished_indices = curr_indices[~true_finished[curr_indices]]
                y0[batch_unfinished_indices] = last_val[~true_finished[curr_indices]]
                ind_to_keep += batch_unfinished_indices
            # Adjust time period
            t_limit[0] = t_limit[1]
            t_limit[1] += 50
            # Remove indices of completed simulations
            indices = indices[torch.isin(indices, torch.Tensor(ind_to_keep).to(device))]
        print("\n Elapsed time: ", time() - time_start)
        return {
            **{f"{comp}_max": output for comp, output in self.max_targets_output.items()},
            **{f"{comp}_sup": output for comp, output in self.sup_targets_output.items()},
        }

    def get_batch_solution(
        self, y0: torch.Tensor, t_eval: torch.Tensor, samples: torch.Tensor
    ) -> torch.Tensor:
        sol = self.model.get_solution(y0=y0, t_eval=t_eval, lhs_table=samples).ys
        if self.model.test:
            # Check if population size changed
            if any(
                [
                    abs(self.model.population.sum() - sol[i, -1, :].sum()) > 50
                    for i in range(sol.shape[0])
                ]
            ):
                raise Exception("Unexpected change in population size!")
        return sol

    def save_finished_indices(self, solutions, indices) -> None:
        last_val = solutions[:, -1, :]
        last_diff = solutions[:, -2, :] - last_val

        for comp in self.max_targets:
            if all(self.max_targets_finished[comp]):
                continue
            self.max_targets_finished[comp][indices] = self.max_stopping_condition(
                comp=comp, last_diff=last_diff
            )
        self.sup_finished[indices] = self.sup_stopping_condition(last_val)

    def max_stopping_condition(self, comp, last_diff):
        comp_idx = self.model.idx(f"{comp}_0")
        return last_diff[:, comp_idx].sum(axis=1) > 0

    def sup_stopping_condition(self, last_val):
        inf_sum = torch.zeros(last_val.shape[0], device=self.model.device)
        for state, data in self.model.state_data.items():
            if data.get("type") in ["infected"]:
                inf_sum += self.model.aggregate_by_age(solution=last_val, comp=state)
        finished = inf_sum < 1
        return finished

    def save_output_for_finished(self, solutions: torch.Tensor, indices) -> None:
        for comp in self.max_targets:
            finished = self.max_targets_finished[comp][indices]
            if any(finished):
                # Only update maximums if the current value is 0, not to overwrite the true maximum
                maxes = self.max_targets_output[comp][indices[finished]]
                self.max_targets_output[comp][indices[finished]] = torch.where(
                    maxes == 0,
                    self.max_metric(solutions=solutions[finished], comp=comp),
                    maxes,
                )

        finished = self.sup_finished[indices]
        if any(finished):
            for comp in self.sup_targets:
                self.sup_targets_output[comp][indices[finished]] = self.sup_metric(
                    solutions=solutions[finished], comp=comp
                )

    def max_metric(self, solutions, comp) -> torch.Tensor:
        comp_max = torch.stack(
            [
                torch.max(self.model.aggregate_by_age(solution=solutions[i, :, :], comp=comp))
                for i in range(solutions.shape[0])
            ]
        ).to(self.model.device)
        return comp_max

    def sup_metric(self, solutions, comp) -> torch.Tensor:
        return solutions[:, -1, self.model.idx(f"{comp}_0")].sum(axis=1)

    def get_true_finished(self) -> torch.BoolTensor:
        finished = self.finished
        finished = finished | self.sup_finished
        for comp in self.max_targets:
            finished = finished & self.max_targets_finished[comp]
        return finished
