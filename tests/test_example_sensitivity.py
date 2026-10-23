"""Check the actual example sensitivity models against explicit compartment equations."""

import json
from copy import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.integrate import solve_ivp

from emsa.generics import GenericModel
from emsa.model import R0Generator
from emsa.sensitivity.target_calc.r0_calculator_lhs import R0CalculatorLHS
from emsa_examples.contact_sensitivity.sampler_contact import SamplerContact
from emsa_examples.contact_sensitivity.sensitivity_model_contact import ContactModel
from emsa_examples.utils.dataloader_16_ag import DataLoader
from emsa_examples.vaccinated_sensitivity.sampler_vaccinated import SamplerVaccinated
from emsa_examples.vaccinated_sensitivity.sensitivity_model_vaccinated import VaccinatedModel
from tests.mock_models import MockContactModel, MockSEIHRModel, MockVaccinatedModel


@pytest.fixture
def data():
    data = DataLoader()
    data.age_data = torch.tensor([1000.0, 2000.0])
    data.params = {
        name: value[:2].clone() if torch.is_tensor(value) else value
        for name, value in data.params.items()
    }
    data.params.update(
        beta=0.2,
        eta=torch.tensor([0.1, 0.3]),
        susc=torch.tensor([0.5, 1.0]),
        total_vaccines=300.0,
        t_start=2,
        T=10,
    )
    data.cm = torch.tensor([[1.0, 0.6], [0.3, 1.2]])
    data.contact_matrices = {"home": 0.2 * data.cm}
    return data


def make_simulation(data, example, boundaries):
    path = Path(__file__).resolve().parents[1] / "emsa_examples" / example / "configs"
    with (path / "model_struct.json").open() as stream:
        model_struct = json.load(stream)
    initial_compartment = "l" if "l" in model_struct["state_data"] else "e"
    return SimpleNamespace(
        data=data,
        device=data.device,
        params=data.params,
        model_struct=model_struct,
        sampled_params_boundaries=boundaries,
        test=True,
        init_vals={initial_compartment: [10.0, 0.0]},
        cm=data.cm,
        population=data.age_data,
        susceptibles=data.age_data - torch.tensor([10.0, 0.0]),
        n_age=2,
        upper_tri_size=3,
        n_samples=16,
        batch_size=4,
    )


def snapshot(params):
    return {
        name: value.clone() if torch.is_tensor(value) else value for name, value in params.items()
    }


def assert_params_unchanged(params, before):
    assert params.keys() == before.keys()
    for name in params:
        torch.testing.assert_close(torch.as_tensor(params[name]), torch.as_tensor(before[name]))


def nonzero_state(model):
    weights = torch.arange(1, model.n_eq + 1, dtype=torch.float32).reshape(model.n_age, -1)
    return (weights / weights.sum(dim=1, keepdim=True) * model.population[:, None]).flatten()


def test_seihr_sampled_trajectories_and_repeated_scenarios(data):
    boundaries = {
        "alpha": [0.2, 0.4],
        "gamma": [0.1, 0.3],
        "eta": [[0.1, 0.1], [0.9, 0.9]],
        "susc": [[0.2, 0.2], [1.0, 1.0]],
    }
    sim = make_simulation(data, "SEIHR_2_age_groups", boundaries)
    model = GenericModel(sim)
    samples = torch.tensor(
        [
            [0.2, 0.1, 0.2, 0.4, 0.3, 0.8],
            [0.3, 0.2, 0.3, 0.5, 0.4, 0.9],
            [0.4, 0.3, 0.4, 0.6, 0.5, 1.0],
        ]
    )
    times = torch.arange(12.0)[None, :]
    initial = model.get_initial_values()[None, :]
    for beta in (0.2, 0.4):
        data.params["beta"] = beta
        before = snapshot(data.params)
        for batch in (samples[:2], samples[2:]):
            model.generate_3D_matrices(batch)
            actual = model.get_solution(
                initial.repeat(len(batch), 1), times.repeat(len(batch), 1)
            ).ys
            for index, row in enumerate(batch):
                reference_data = copy(data)
                reference_data.params = {
                    **data.params,
                    "alpha": row[0],
                    "gamma": row[1],
                    "eta": row[2:4],
                    "susc": row[4:6],
                }
                reference = MockSEIHRModel(reference_data)
                expected = model.get_sol_from_ode(initial, times, reference.odefun).ys
                torch.testing.assert_close(actual[index], expected[0], rtol=1e-5, atol=1e-3)
            assert_params_unchanged(data.params, before)
            assert model.matrix_generator.ps is data.params


def test_sampled_seihr_r0_matches_next_generation_matrix_without_mutation(data):
    boundaries = {
        "beta": [0.1, 0.4],
        "gamma": [0.1, 0.3],
        "eta": [[0.1, 0.1], [0.9, 0.9]],
        "susc": [[0.2, 0.2], [1.0, 1.0]],
    }
    sim = make_simulation(data, "SEIHR_2_age_groups", boundaries)
    samples = torch.tensor([[0.2, 0.1, 0.2, 0.4, 0.3, 0.8], [0.4, 0.3, 0.4, 0.6, 0.5, 1.0]])
    before = snapshot(data.params)
    actual = R0CalculatorLHS(sim).get_output(samples)
    expected = []
    for beta, gamma, eta0, eta1, susc0, susc1 in samples.numpy():
        susceptibility = np.array([susc0, susc1]) * (sim.susceptibles / sim.population).numpy()
        infectious_duration = (1 - np.array([eta0, eta1])) / gamma
        ngm = beta * np.diag(susceptibility) @ data.cm.numpy() @ np.diag(infectious_duration)
        expected.append(np.max(np.abs(np.linalg.eigvals(ngm))))
    np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-5)
    assert_params_unchanged(data.params, before)


def test_contact_samples_preserve_home_contacts_and_match_equations(data):
    sim = make_simulation(data, "contact_sensitivity", {})
    model = ContactModel(sim, base_r0=1.8)
    model.initialize_matrices()
    reductions = torch.tensor([[0.0, 0.0, 0.0], [0.2, 0.4, 0.6], [1.0, 1.0, 1.0]])
    contacts = model.get_contacts_from_lhs(reductions)
    torch.testing.assert_close(contacts[0], data.cm)
    torch.testing.assert_close(contacts[-1], data.contact_matrices["home"])
    for cm in contacts:
        totals = data.age_data[:, None] * cm
        torch.testing.assert_close(totals, totals.T)

    before = snapshot(data.params)
    betas = model._get_betas_from_contacts(contacts)
    initial = nonzero_state(model)[None, :]
    times = torch.arange(4.0)[None, :]
    actual = model.get_solution(initial.repeat(3, 1), times.repeat(3, 1), lhs_table=reductions).ys
    for index, (cm, beta) in enumerate(zip(contacts, betas)):
        reference_data = copy(data)
        reference_data.cm = cm
        reference_data.params = {**data.params, "beta": beta}
        reference = MockContactModel(reference_data)
        expected = model.get_sol_from_ode(initial, times, reference.odefun).ys
        torch.testing.assert_close(actual[index], expected[0], rtol=1e-5, atol=1e-3)
        r0gen = R0Generator(reference_data, sim.model_struct)
        r0 = beta * r0gen.get_eig_val(sim.susceptibles, sim.population, cm)
        torch.testing.assert_close(r0, torch.tensor(1.8), rtol=1e-5, atol=1e-6)
    assert_params_unchanged(data.params, before)


def test_contact_sampling_bounds_keep_uncalibrated_r0_above_one(data):
    sim = make_simulation(data, "contact_sensitivity", {})
    model = ContactModel(sim, base_r0=1.8)
    data.params["beta"] = model.get_beta_from_r0(1.8, sim.susceptibles)
    sampler = SamplerContact(sim, {"r0": 1.8})
    lower, upper = sampler.lhs_bounds_dict["contacts"]
    assert sampler.get_r0_from_kappa(0) < 1 < sampler.get_r0_from_kappa(1)
    assert np.all(lower == 0.2)
    assert np.all(upper > lower)
    assert sampler.get_r0_from_kappa(1 - upper[0]) > 1
    for cm in model.get_contacts_from_lhs(torch.from_numpy(sampler.get_lhs_table())):
        r0 = data.params["beta"] * R0Generator(data, sim.model_struct).get_eig_val(
            sim.susceptibles, sim.population, cm
        )
        assert r0 > 1


@pytest.mark.parametrize("time", [1.0, 2.0, 11.0, 12.0, [1.0, 2.0], [11.0, 12.0]])
def test_sampled_vaccination_matches_equations_at_campaign_boundaries(data, time):
    sim = make_simulation(data, "vaccinated_sensitivity", {})
    model = VaccinatedModel(sim)
    samples = torch.tensor([[0.8, 0.2], [0.1, 0.9]])
    initial = nonzero_state(model)[None, :].repeat(2, 1)
    before = snapshot(data.params)
    solution = model.get_solution(
        initial, torch.tensor([[0.0, 1.0]]).repeat(2, 1), lhs_table=samples
    )
    assert torch.isfinite(solution.ys).all()
    times = torch.as_tensor(time).expand(2)
    actual = model.get_vaccinated_ode(2)(times, initial)
    for index, sample in enumerate(samples):
        reference_data = copy(data)
        reference_data.params = {
            **data.params,
            "daily_vacc": sample * data.params["total_vaccines"] / data.params["T"],
        }
        expected = MockVaccinatedModel(reference_data).odefun(
            times[index].reshape(1), initial[index].unsqueeze(0)
        )
        torch.testing.assert_close(actual[index], expected, rtol=1e-5, atol=1e-3)
    torch.testing.assert_close(actual.sum(dim=1), torch.zeros(2), rtol=0, atol=1e-3)
    assert_params_unchanged(data.params, before)


def test_vaccination_with_no_eligible_population_is_finite(data):
    sim = make_simulation(data, "vaccinated_sensitivity", {})
    model = VaccinatedModel(sim)
    initial = torch.zeros(1, model.n_eq)
    initial[:, model.idx("e_0")] = data.age_data
    model.initialize_matrices()
    model.V_1 = model._get_V_1_from_lhs(torch.tensor([[0.5, 0.5]]))
    model.V_2 = model.matrix_generator.get_V_2()
    time = torch.tensor([2.0])
    actual = model.get_vaccinated_ode(1)(time, initial)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, model.get_basic_ode()(time, initial))


@pytest.mark.parametrize("total", [0.0, 2400.0, 3000.0])
def test_vaccine_allocation_preserves_budget_and_population_caps(data, total):
    sim = make_simulation(data, "vaccinated_sensitivity", {})
    sim.params["total_vaccines"] = total
    sampler = SamplerVaccinated(sim, {})
    allocations = sampler.allocate_vaccines(np.array([[0.99, 0.01], [0.01, 0.99], [1.0, 0.0]]))
    np.testing.assert_allclose(allocations.sum(axis=1), 1.0)
    assert np.all(allocations >= 0)
    assert np.all(allocations * sim.params["total_vaccines"] <= data.age_data.numpy())


def test_vaccine_allocation_rejects_an_impossible_budget(data):
    sim = make_simulation(data, "vaccinated_sensitivity", {})
    sim.params["total_vaccines"] = 3001.0
    with pytest.raises(ValueError, match="total population"):
        SamplerVaccinated(sim, {}).allocate_vaccines(np.array([[0.5, 0.5]]))


def scipy_trajectory(reference_data, reference_class, initial, times, campaign=None):
    """Integrate explicit equations independently, splitting vaccination discontinuities."""
    reference_data = copy(reference_data)
    reference_data.cm = reference_data.cm.double()
    reference_data.age_data = reference_data.age_data.double()
    reference_data.params = {
        key: value.double() if torch.is_tensor(value) else value
        for key, value in reference_data.params.items()
    }
    reference = reference_class(reference_data)
    times = times.numpy()
    cuts = [times[0]]
    if campaign:
        cuts += [t for t in campaign if times[0] < t < times[-1]]
    cuts.append(times[-1])
    values = np.empty((len(times), initial.numel()))
    current = initial.double().numpy()
    for left, right in zip(cuts[:-1], cuts[1:]):
        if campaign:
            active = campaign[0] <= (left + right) / 2 < campaign[1]
            reference.vaccination = lambda t, active=active: int(active)

        def rhs(t, y):
            return reference.odefun(torch.tensor([t]), torch.from_numpy(y)[None, :]).numpy()

        solution = solve_ivp(rhs, (left, right), current, rtol=1e-10, atol=1e-10, dense_output=True)
        assert solution.success
        included = (times >= left) & (times <= right)
        values[included] = solution.sol(times[included]).T
        current = solution.y[:, -1]
    return values


@pytest.mark.parametrize(
    "example", ["SEIHR_2_age_groups", "contact_sensitivity", "vaccinated_sensitivity"]
)
def test_example_accuracy_and_step_convergence(data, example):
    data.params["beta"] = 0.6
    boundaries = {"alpha": [0.2, 0.4], "gamma": [0.1, 0.3]} if example.startswith("SEIHR") else {}
    sim = make_simulation(data, example, boundaries)
    if example.startswith("SEIHR"):
        model = GenericModel(sim)
        samples = torch.tensor([[0.25, 0.12], [0.35, 0.18]])
        reference_class = MockSEIHRModel
    elif example == "contact_sensitivity":
        model = ContactModel(sim, base_r0=1.8)
        samples = torch.tensor([[0.2, 0.4, 0.6], [0.6, 0.2, 0.4]])
        reference_class = MockContactModel
    else:
        # Use the actual example's campaign timing, scaled to the small test population.
        data.params.update(t_start=0, T=90)
        model = VaccinatedModel(sim)
        samples = torch.tensor([[0.8, 0.2], [0.1, 0.9]])
        reference_class = MockVaccinatedModel
    model.generate_3D_matrices(samples)
    initial = model.get_initial_values()
    times = torch.arange(121.0)
    expected = []
    if example == "contact_sensitivity":
        contacts = model.get_contacts_from_lhs(samples)
        betas = model._get_betas_from_contacts(contacts)
    for index, sample in enumerate(samples):
        reference_data = copy(data)
        reference_data.params = dict(data.params)
        campaign = None
        if example.startswith("SEIHR"):
            reference_data.params.update(alpha=sample[0], gamma=sample[1])
        elif example == "contact_sensitivity":
            reference_data.cm = contacts[index]
            reference_data.params["beta"] = betas[index]
        else:
            reference_data.params["daily_vacc"] = (
                sample * data.params["total_vaccines"] / data.params["T"]
            )
            campaign = [data.params["t_start"], data.params["t_start"] + data.params["T"]]
        expected.append(scipy_trajectory(reference_data, reference_class, initial, times, campaign))
    expected = np.stack(expected)
    initial = initial.repeat(len(samples), 1)
    actual = model.get_solution(
        initial,
        times,
        lhs_table=samples,
        solver_config={"method": "dopri5", "step_size": 0.25, "rtol": 1e-6, "atol": 1e-7},
    ).ys.numpy()
    np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=0.02)
    errors = []
    for step in (1.0, 0.5, 0.25):
        actual = model.get_solution(
            initial,
            times,
            lhs_table=samples,
            solver_config={"method": "euler", "step_size": step},
        ).ys.numpy()
        errors.append(float(np.max(np.abs(actual - expected))))
    assert errors[1] < 0.7 * errors[0], errors
    assert errors[2] < 0.7 * errors[1], errors
    print(f"{example}: Euler max errors at 1, 0.5, 0.25 days: {errors}")
