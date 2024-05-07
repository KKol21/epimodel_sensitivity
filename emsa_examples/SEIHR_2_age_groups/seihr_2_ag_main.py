from types import SimpleNamespace

import torch

from emsa_examples.SEIHR_2_age_groups.simulation_seihr import SimulationSEIHR


def main():
    params = {"gamma": 0.2,
              "beta": 0.2,
              "alpha": 0.3,
              "eta": [0.5, 0.5]}
    for key, value in params.items():
        params[key] = torch.tensor(value)
    contact_data = torch.tensor([[1, 2],
                                 [0.5, 1]])

    age_data = torch.tensor([1E5,
                             2E5])

    data = SimpleNamespace(**{"params": params,
                              "cm": contact_data,
                              "age_data": age_data,
                              "n_age": len(age_data),
                              "device": "cpu"})

    sim = SimulationSEIHR(data)
    sim.model.visualize_transmission_graph()
    sim.run_sampling()
    sim.calculate_all_prcc()
    sim.calculate_all_p_values()
    sim.plot_all_prcc()


if __name__ == "__main__":
    main()
