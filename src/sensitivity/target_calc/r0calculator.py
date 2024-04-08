import torch
from tqdm import tqdm

from src.model.r0 import R0Generator
from src.sensitivity.sensitivity_model_base import get_params_col_idx, get_lhs_dict
from src.simulation_base import SimulationBase


class R0Calculator:
    def __init__(self, sim_obj: SimulationBase):
        self.sim_obj = sim_obj

    def get_output(self, lhs_table: torch.Tensor):
        sim_obj = self.sim_obj
        n_samples = lhs_table.shape[0]
        spb = sim_obj.sampled_params_boundaries
        if spb is None:
            raise ValueError("Sampled parameters boundaries not specified, automatic R0 generation isn't possible")
        pci = get_params_col_idx(sampled_params_boundaries=spb)
        lhs_dict = get_lhs_dict(params=spb.keys(), lhs_table=lhs_table, params_col_idx=pci)
        r0gen = R0Generator(sim_obj.data, **sim_obj.model_struct)
        r0s = []
        print(f"Calculating R0 for {n_samples} samples")
        for idx in tqdm(range(n_samples)):
            # Select idx. value from lhs table for each parameter
            row_dict = {key: value[idx] if len(value.size()) < 2 else value[idx, :]
                        for key, value in lhs_dict.items()}
            r0gen.params.update(row_dict)
            r0 = sim_obj.params["beta"] * r0gen.get_eig_val(contact_mtx=sim_obj.cm,
                                                            susceptibles=sim_obj.susceptibles.reshape(1, -1),
                                                            population=sim_obj.population)
            r0s.append(r0)
        return torch.tensor(r0s, device=sim_obj.device)
