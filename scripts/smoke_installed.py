"""Exercise an installed wheel from outside its source checkout, without example files."""

from importlib.metadata import metadata
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile

import numpy as np
import torch

import emsa
from emsa.generics import SimulationGeneric
from emsa.model import EpidemicModel


def main():
    location = Path(emsa.__file__).resolve()
    assert location.is_relative_to(Path(sys.prefix).resolve()), location
    assert metadata("emsa")["License-Expression"] == "MIT"
    data = SimpleNamespace(
        params={"beta": 0.4, "gamma": 0.15},
        cm=torch.tensor([[1.0]]),
        age_data=torch.tensor([1000.0]),
        n_age=1,
        device="cpu",
    )
    structure = {
        "state_data": {
            "s": {"type": "susceptible"},
            "i": {"type": "infected", "rate": "gamma"},
            "r": {"type": "recovered"},
        },
        "tms_rules": [{"source": "s", "target": "i", "actors-params": {"i": None}}],
        "trans_data": [{"source": "i", "target": "r"}],
    }
    config = {
        "target_vars": ["i_max", "r_sup"],
        "sampled_params_boundaries": {"beta": [0.3, 0.5], "gamma": [0.1, 0.2]},
        "init_vals": {"i": [10.0]},
        "n_samples": 32,
        "batch_size": 16,
        "seed": 42,
        "solver": {"method": "dopri5", "step_size": 1.0},
    }
    model = EpidemicModel(data, structure, solver_config=config["solver"])
    solution = model.get_solution(
        model.get_initial_values_from_dict(config["init_vals"]), torch.arange(61.0)
    )
    assert torch.isfinite(solution.ys).all()
    with tempfile.TemporaryDirectory() as directory:
        output_dir = Path(directory) / "unused"
        result = SimulationGeneric(
            data, structure, config, output_dir=output_dir, save_results=False
        ).run()[""]
        assert result["samples"].shape == (32, 2)
        assert not output_dir.exists()
        for target in config["target_vars"]:
            assert np.isfinite(result["targets"][target]).all()
            assert np.isfinite(result["prcc"][target]).all()
        assert result["prcc"]["i_max"][0] > 0
        assert result["prcc"]["i_max"][1] < 0
    print(f"Installed-wheel simulation and sensitivity smoke check passed: {location}")


if __name__ == "__main__":
    main()
