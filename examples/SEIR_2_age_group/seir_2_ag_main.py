from types import SimpleNamespace
from examples.SEIR_2_age_group.simulation_seir import SimulationSEIR

import torch

model_params = {"gamma": 0.2, "beta": 0.2, "alpha": 0.3}

contact_data = torch.tensor([[1, 2], [0.5, 1]])

age_data = torch.tensor([[1E5, 2E5]])

state_data = {
    "s": {
        "type": "susceptible",
        "n_substates": 1},
    "e": {
        "type": "infected",
        "n_substates": 1
    },
    "i": {
        "type": "infectious",
        "n_substates": 1},
    "r": {
        "type": "recovered",
        "n_substates": 1}
}

trans_data = [{
        "source": "s",
        "target": "e",
        "param": "beta",
        "distr": None,
        "type": "infection",
        "actor": "i"
    },
    {
        "source": "i",
        "target": "r",
        "param": "gamma",
        "distr": None,
        "type": "basic"},
    {
        "source": "e",
        "target": "i",
        "param": "alpha",
        "distr": None,
        "type": "basic"
    }]

data = SimpleNamespace(**{"model_params": model_params,
                          "cm": contact_data,
                          "age_data": age_data,
                          "state_data": state_data,
                          "trans_data": trans_data,
                          "n_age": len(age_data[0]),
                          "device": "cpu"})

sim = SimulationSEIR(data)
sim.run_sampling()
sim.calculate_prcc_for_simulations()
sim.calculate_all_p_values()
sim.plot_prcc_from_sim()
