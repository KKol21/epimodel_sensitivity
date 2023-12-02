from time import time

import torch
from tqdm import tqdm


class PeakCalculator:
    def __init__(self, sim_obj):
        self.sim_obj = sim_obj
        self.model = sim_obj.model

    def get_output(self, lhs_table, batch_size, target_var):
        batches = []
        n_samples = lhs_table.shape[0]

        t_start = time()
        for batch_idx in tqdm(range(0, n_samples, batch_size),
                              desc="Batches completed"):
            batch = lhs_table[batch_idx: batch_idx + batch_size]
            solutions = self.get_sol_from_lhs(lhs_table=batch)
            comp_maxes = self.get_max(sol=solutions,
                                      comp=target_var.split('_')[0])
            batches.append(comp_maxes)
        elapsed = time() - t_start
        print(f"\n Average speed = {round(n_samples / elapsed, 3)} iterations/second \n")
        return torch.concat(batches)

    def get_sol_from_lhs(self, lhs_table):
        # Initialize timesteps and initial values
        t_eval = torch.stack(
            [torch.linspace(1, 1100, 1100)] * lhs_table.shape[0]
        ).to(self.model.device)

        y0 = torch.stack(
            [self.model.get_initial_values()] * lhs_table.shape[0]
        ).to(self.model.device)

        sol = self.model.get_solution(y0=y0, t_eval=t_eval, lhs_table=lhs_table).ys
        if self.model.test:
            # Check if population size changed
            if any([abs(self.model.population.sum() - sol[i, -1, :].sum()) > 50 for i in range(sol.shape[0])]):
                raise Exception("Unexpected change in population size!")
        return sol

    def get_max(self, sol, comp: str):
        comp_max = []
        for i in range(sol.shape[0]):
            comp_sol = self.model.aggregate_by_age(solution=sol[i, :, :], comp=comp)
            comp_max.append(torch.max(comp_sol))
        return torch.tensor(comp_max)
