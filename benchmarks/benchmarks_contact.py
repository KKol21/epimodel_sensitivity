import gc

from emsa_examples.contact_sensitivity.simulation_contact import SimulationContact
from emsa_examples.utils.dataloader_16_ag import DataLoader


import time


def benchmark_cpu(n=1):
    total_time = 0
    for _ in range(n):
        data = DataLoader()
        sim = SimulationContact(data)
        start_time = time.time()
        sim.run_sampling()
        end_time = time.time()
        total_time += (end_time - start_time)
    print(f"CPU average time: {total_time / n} seconds")
    return total_time

import torch
import gc
def benchmark_gpu(n=1):
    x = torch.zeros(size=(1000, 100, 100), device="cuda")
    total_time = 0
    for _ in range(n):
        start_time = time.time()
        data = DataLoader(device="cuda")
        sim = SimulationContact(data)
        sim.run_sampling()
        end_time = time.time()
        total_time += (end_time - start_time)
    print(f"GPU average time: {total_time / n} seconds")
    return total_time

#cpu = benchmark_cpu()
gpu = benchmark_gpu()
#print("cpu: ", cpu)
print("gpu: ", gpu)