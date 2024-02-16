from types import SimpleNamespace
from examples.SEIR_no_age_groups.simulation_seir import SimulationSEIR

import torch

model_params = {"gamma": 0.2, "beta": 0.2, "alpha": 0.3}

contact_data = torch.tensor([[1]])

age_data = torch.tensor([[10000]])

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

trans_data = {
    "trans_0": {
        "source": "s",
        "target": "e",
        "param": "beta",
        "distr": None,
        "type": "infection",
        "actor": "i"
    },
    "trans_1": {
        "source": "i",
        "target": "r",
        "param": "gamma",
        "distr": None,
        "type": "basic"},
    "trans_2": {
        "source": "e",
        "target": "i",
        "param": "alpha",
        "distr": None,
        "type": "basic"
    }
}

data = SimpleNamespace(**{"model_params": model_params,
                          "cm": contact_data,
                          "age_data": age_data,
                          "state_data": state_data,
                          "trans_data": trans_data,
                          "n_age": 1,
                          "device": "cpu"})

sim = SimulationSEIR(data)
sim.run_sampling()
sim.calculate_prcc_for_simulations()
sim.calculate_all_p_values()
sim.plot_prcc_for_simulations()
