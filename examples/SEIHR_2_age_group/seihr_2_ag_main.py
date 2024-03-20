from types import SimpleNamespace

import torch

from examples.SEIHR_2_age_group.simulation_seir import SimulationSEIR


def main():
    model_params = {"gamma": 0.2, "beta": 0.2, "alpha": 0.3}

    contact_data = torch.tensor([[1, 2], [0.5, 1]])

    age_data = torch.tensor([[1E5, 2E5]])

    data = SimpleNamespace(**{"model_params": model_params,
                              "cm": contact_data,
                              "age_data": age_data,
                              "n_age": len(age_data[0]),
                              "device": "cpu"})

    sim = SimulationSEIR(data)
    sim.run_sampling()
    sim.calculate_all_prcc()
    sim.calculate_all_p_values()
    sim.plot_all_prcc()


if __name__ == "main":
    main()
    