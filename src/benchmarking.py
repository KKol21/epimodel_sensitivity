import cProfile
import os
import pstats
import torch
from tqdm import tqdm

from src.model.r0 import R0Generator


def benchmark_evaluation(sim_obj):
    os.makedirs("../benchmarks", exist_ok=True)

    sim_obj.params["susc"] = torch.ones(sim_obj.n_age).to(sim_obj.data.device)
    base_r0 = 2
    r0gen = R0Generator(param=sim_obj.params, device=sim_obj.data.device, n_age=sim_obj.n_age)
    beta = base_r0 / r0gen.get_eig_val(contact_mtx=sim_obj.contact_matrix,
                                       susceptibles=sim_obj.susceptibles.reshape(1, -1),
                                       population=sim_obj.population)

    sim_obj.params.update({"beta": beta})
    daily_vac = torch.full(size=(16,), fill_value=2000)
    sim_obj.params.update({"v": daily_vac})

    t = torch.linspace(1, 220, 220).to(sim_obj.data.device)
    with cProfile.Profile() as pr:
        for _ in tqdm(range(200)):
            sim_obj.model.get_solution_torch(t=t, cm=sim_obj.contact_matrix, parameters=sim_obj.params)
    stats = pstats.Stats(pr)
    stats.print_stats()
    stats.dump_stats(filename='../benchmarks/lambda_func_eval.prof')

    with cProfile.Profile() as pr2:
        sim_obj.model2.get_constant_matrices()
        for _ in tqdm(range(200)):
            sim_obj.model2.get_solution_torch_test(t=t, cm=sim_obj.contact_matrix, daily_vac=daily_vac.float())
    stats_eff = pstats.Stats(pr2)
    stats_eff.print_stats()
    stats_eff.dump_stats(filename='../benchmarks/matrix_repr_eval.prof')