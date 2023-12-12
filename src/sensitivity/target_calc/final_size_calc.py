import torch
from tqdm import tqdm
from time import time


class FinalSizeCalculator:
    def __init__(self, model):
        self.model = model

    def get_output(self, lhs_table, batch_size, target_var):
        device = self.model.device
        model = self.model

        n_samples = lhs_table.shape[0]
        indices = torch.IntTensor(range(0, n_samples)).to(device)
        final_sizes = torch.zeros(n_samples, device=device)

        t_limit = [0, 400]
        y0 = torch.stack(
            [model.get_initial_values()] * n_samples
        ).to(device)
        time_start = time()
        # Iterate until all the eqs are solved or some of them don't seem to converge
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

                solutions = self.get_sol_from_batch(y0[curr_indices],
                                                    t_eval[batch_slice],
                                                    samples=batch)
                last_val = solutions[:, -1, :]  # solutions.shape = (len(indices), t_limit, n_comp)
                # Check which simulations have finished
                finished = (last_val[:, model.idx("i_0")].sum(axis=1) < 1).to(device)
                if any(finished):
                    final_sizes[curr_indices[finished]] = self.get_size(sol=solutions[finished],
                                                                        comp=target_var.split('_')[0])
                # Save the last values of unfinished simulations to use as initial values in next iter
                y0[curr_indices[~finished]] = last_val[~finished]
                ind_to_keep += curr_indices[~finished]
            # Adjust time period
            t_limit[0] = t_limit[1]
            t_limit[1] += 100
            indices = indices[torch.isin(indices, torch.Tensor(ind_to_keep).to(device))]
        print("\n Elapsed time: ", time() - time_start)
        return final_sizes

    def get_sol_from_batch(self, y0, t_eval, samples):
        sol = self.model.get_solution(y0=y0, t_eval=t_eval, lhs_table=samples).ys
        if self.model.test:
            # Check if population size changed
            if any([abs(self.model.population.sum() - sol[i, -1, :].sum()) > 50 for i in range(sol.shape[0])]):
                raise Exception("Unexpected change in population size!")
        return sol

    def get_size(self, sol, comp: str):
        return sol[:, -1, self.model.idx(f"{comp}_0")].sum(axis=1)
