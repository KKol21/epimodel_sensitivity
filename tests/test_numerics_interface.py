"""Independent numerical reference and the dictionary/in-memory public interface."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.integrate import solve_ivp

from emsa.generics import SimulationGeneric
from emsa.model import EpidemicModel


@pytest.fixture
def inputs():
    path = (
        Path(__file__).resolve().parents[1]
        / "emsa_examples/SIR_no_age_groups/configs/model_struct.json"
    )
    structure = json.loads(path.read_text())
    data = SimpleNamespace(
        params={"beta": 0.4, "gamma": 0.15},
        cm=torch.tensor(1.0),
        age_data=torch.tensor([1000.0]),
        n_age=1,
        device="cpu",
    )
    config = {
        "target_vars": ["i_max", "r_sup"],
        "n_samples": 32,
        "batch_size": 16,
        "sampled_params_boundaries": {"beta": [0.3, 0.5], "gamma": [0.1, 0.2]},
        "init_vals": {"i": [10.0]},
        "seed": 42,
        "solver": {"method": "dopri5", "step_size": 1.0},
    }
    return data, structure, config


def test_trajectory_against_scipy_and_euler_convergence(inputs):
    data, structure, _ = inputs
    model = EpidemicModel(data, structure)
    times = torch.arange(0, 61, dtype=torch.float32)
    y0 = model.get_initial_values_from_dict({"i": [10.0]})

    def sir(t, y):
        infections = 0.4 * y[0] * y[1] / 1000
        recoveries = 0.15 * y[1]
        return [-infections, infections - recoveries, recoveries]

    reference = solve_ivp(
        sir, [0, 60], y0.numpy(), t_eval=times.numpy(), rtol=1e-10, atol=1e-10
    ).y.T
    adaptive = model.get_solution(y0, times, solver_config={"method": "dopri5"}).ys[0].numpy()
    np.testing.assert_allclose(adaptive, reference, rtol=2e-5, atol=0.002)
    errors = []
    for step in [1.0, 0.5, 0.25]:
        values = model.get_solution(y0, times, solver_config={"step_size": step}).ys[0].numpy()
        errors.append(np.max(np.abs(values - reference)))
    assert errors[1] < 0.6 * errors[0]
    assert errors[2] < 0.6 * errors[1]


def test_unstable_euler_fails_and_smaller_step_resolves_it(inputs):
    data, structure, _ = inputs
    data.params.update(beta=1.5, gamma=0.2)
    model = EpidemicModel(data, structure)
    y0 = model.get_initial_values_from_dict({"i": [900.0]})
    times = torch.arange(0.0, 11.0)
    with pytest.raises(RuntimeError, match="negative populations"):
        model.get_solution(y0, times)
    values = model.get_solution(y0, times, solver_config={"step_size": 0.1}).ys
    assert (values >= 0).all()


def test_solver_failure_and_nonfinite_output_are_reported(inputs):
    data, structure, _ = inputs
    model = EpidemicModel(data, structure)
    y0 = model.get_initial_values_from_dict({"i": [10.0]})
    with pytest.raises(RuntimeError, match="status codes"):
        model.get_solution(y0, torch.arange(11.0), solver_config={"max_steps": 1})
    with pytest.raises(RuntimeError, match="non-finite|status codes"):
        model.get_solution(y0, torch.arange(3.0), odefun=lambda t, y: y * float("nan"))


def test_seeded_in_memory_run_and_optional_export(inputs, tmp_path):
    data, structure, config = inputs
    output_dir = tmp_path / "results"
    simulation = SimulationGeneric(
        data, structure, config, output_dir=output_dir, save_results=False
    )
    first = simulation.run()[""]
    assert not output_dir.exists()
    assert first["samples"].shape == (32, 2)
    assert set(first["prcc"]) == {"i_max", "r_sup"}
    assert first["prcc"]["i_max"][0] > 0
    assert first["prcc"]["i_max"][1] < 0
    second = simulation.run()[""]
    np.testing.assert_array_equal(first["samples"], second["samples"])
    for target in config["target_vars"]:
        np.testing.assert_array_equal(first["targets"][target], second["targets"][target])
    simulation.save_results = True
    exported = simulation.run()[""]
    assert (output_dir / "lhs/lhs.csv").is_file()
    assert (output_dir / "simulations/simulations_i_max.csv").is_file()
    assert all("__" not in path.name for path in output_dir.rglob("*"))
    metadata = json.loads((output_dir / "metadata.json").read_text())
    assert metadata["seed"] == 42
    assert metadata["solver"]["method"] == "dopri5"
    np.testing.assert_allclose(
        np.loadtxt(output_dir / "prcc/prcc_i_max.csv"), exported["prcc"]["i_max"]
    )
    simulation.calculate_all_prcc()
    simulation.calculate_all_p_values()
    np.testing.assert_allclose(
        np.loadtxt(output_dir / "p_values/p_values_i_max.csv"),
        exported["p_values"]["i_max"],
    )
