from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy import stats

from emsa.generics import GenericSampler
from emsa.sensitivity.prcc import get_prcc_values
from emsa.sensitivity.target_calc.sol_based_target_calc import TargetCalc
from emsa.utils import SimulationBase


@pytest.mark.parametrize(
    "boundaries, expected",
    [
        ({"a": [0.1, 0.3], "b": [1, 3]}, [[0.1, 0.3], [1, 3]]),
        ({"a": [[0.1, 0.2], [0.3, 0.4]]}, [[0.1, 0.3], [0.2, 0.4]]),
        (
            {"a": [[0.1, 0.2], [0.3, 0.4]], "b": [[1, 2], [3, 4]]},
            [[0.1, 0.3], [0.2, 0.4], [1, 3], [2, 4]],
        ),
        (
            {"a": [[0.1, 0.2], [0.3, 0.4]], "b": [5, 6], "c": [[1, 2], [3, 4]]},
            [[0.1, 0.3], [0.2, 0.4], [5, 6], [1, 3], [2, 4]],
        ),
    ],
)
def test_lhs_columns_follow_parameter_order(boundaries, expected):
    sim = SimpleNamespace(sampled_params_boundaries=boundaries, n_samples=24, batch_size=4)
    sampler = GenericSampler(sim)
    expected = np.array(expected)
    np.testing.assert_array_equal(sampler._get_lhs_bounds(), expected)
    samples = sampler.get_lhs_table()
    assert samples.shape == (24, len(expected))
    assert np.all(samples >= expected[:, 0])
    assert np.all(samples <= expected[:, 1])


@pytest.mark.parametrize("sign", [-1, 1])
def test_one_parameter_perfect_prcc_has_correct_sign(sign):
    x = np.arange(10)
    np.testing.assert_allclose(get_prcc_values(np.c_[x, sign * x]), [sign])


def test_prcc_with_ties_matches_spearman_for_one_parameter():
    x = np.arange(10)
    y = np.repeat([0, 1], 5)
    np.testing.assert_allclose(get_prcc_values(np.c_[x, y]), [stats.spearmanr(x, y).statistic])


def test_ordinary_prcc_matches_independent_rank_residuals():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(200, 3))
    y = x @ np.array([0.8, -0.5, 0.2]) + rng.normal(size=200)
    ranked = stats.rankdata(np.c_[x, y], axis=0)
    expected = []
    for index in range(3):
        controls = np.c_[np.ones(200), np.delete(ranked[:, :3], index, axis=1)]
        x_residual = (
            ranked[:, index] - controls @ np.linalg.lstsq(controls, ranked[:, index], rcond=None)[0]
        )
        y_residual = (
            ranked[:, -1] - controls @ np.linalg.lstsq(controls, ranked[:, -1], rcond=None)[0]
        )
        expected.append(np.corrcoef(x_residual, y_residual)[0, 1])
    np.testing.assert_allclose(get_prcc_values(np.c_[x, y]), expected, atol=1e-12)


def test_perfect_output_with_independent_inputs_keeps_defined_coefficient():
    x = np.random.default_rng(42).normal(size=(100, 2))
    with pytest.warns(RuntimeWarning, match="undefined"):
        actual = get_prcc_values(np.c_[x, -x[:, 0]])
    np.testing.assert_allclose(actual[0], -1.0)
    assert np.isnan(actual[1])


@pytest.mark.parametrize(
    "table, message",
    [
        (np.c_[np.arange(10), np.zeros(10)], "constant"),
        (np.c_[np.zeros(10), np.arange(10)], "constant"),
        (np.c_[np.arange(10), np.arange(10), np.arange(10) ** 2], "rank-deficient"),
        (np.array([[0, 0], [1, np.nan], [2, 2]]), "finite"),
    ],
)
def test_degenerate_prcc_inputs_fail_explicitly(table, message):
    with pytest.raises(ValueError, match=message):
        get_prcc_values(table)


def test_p_values_use_sampled_parameter_count(tmp_path):
    (tmp_path / "prcc").mkdir()
    coefficients = np.array([0.4, -0.6, 1.0])
    np.savetxt(tmp_path / "prcc" / "prcc_case.csv", coefficients)
    sim = SimpleNamespace(folder_name=str(tmp_path), n_samples=40, n_age=19)
    SimulationBase.calculate_p_values(sim, "case")
    actual = np.loadtxt(tmp_path / "p_values" / "p_values_case.csv")
    dof = 40 - len(coefficients) - 1
    expected = [2 * stats.t.sf(abs(r) * np.sqrt(dof / (1 - r * r)), dof) for r in coefficients[:2]]
    np.testing.assert_allclose(actual, [*expected, 0.0])
    sim.n_samples = 4
    with pytest.raises(ValueError, match="More samples"):
        SimulationBase.calculate_p_values(sim, "case")


class AnalyticEpidemic:
    """A single peak at a sampled integer time, followed by extinction."""

    device = "cpu"
    test = True
    population = torch.tensor([100.0])
    state_data = {"i": {"type": "infected"}, "r": {"type": "recovered"}}

    def get_initial_values(self):
        return torch.tensor([1.0, 99.0])

    def generate_3D_matrices(self, samples):
        pass

    def aggregate_by_age(self, solution, comp):
        return solution[:, 0 if comp == "i" else 1]

    @staticmethod
    def infected(times, peaks):
        return torch.clamp(
            1 + torch.minimum(times, peaks) - 2 * torch.clamp(times - peaks, min=0), min=0
        )

    def get_solution(self, y0, t_eval, lhs_table):
        infected = self.infected(t_eval, lhs_table[:, :1])
        expected_initial = torch.stack((infected[:, 0], 100 - infected[:, 0]), dim=1)
        # Continuation must start at the last evaluated time, with its exact state.
        torch.testing.assert_close(y0, expected_initial)
        return SimpleNamespace(ys=torch.stack((infected, 100 - infected), dim=2))


@pytest.mark.parametrize("targets", [["i_max"], ["r_sup"], ["i_max", "r_sup"]])
@pytest.mark.parametrize("batch_size", [1, 3])
def test_target_completion_across_windows_and_batches(targets, batch_size):
    calc = TargetCalc(AnalyticEpidemic(), targets, {"tlim_ini": 3, "tlim_final": 11, "tdelta": 3})
    output = calc.get_output(torch.tensor([[1.0], [7.0], [4.0]]), batch_size)
    if "i_max" in targets:
        torch.testing.assert_close(output["i_max"], torch.tensor([2.0, 8.0, 5.0]))
    if "r_sup" in targets:
        torch.testing.assert_close(output["r_sup"], torch.full((3,), 100.0))
    assert calc.finished.all()


def test_target_evaluation_runs_when_initial_and_final_limits_match():
    calc = TargetCalc(
        AnalyticEpidemic(), ["i_max", "r_sup"], {"tlim_ini": 3, "tlim_final": 3, "tdelta": 1}
    )
    output = calc.get_output(torch.tensor([[1.0]]), 1)
    torch.testing.assert_close(output["i_max"], torch.tensor([2.0]))
    torch.testing.assert_close(output["r_sup"], torch.tensor([100.0]))


def test_unfinished_targets_raise_instead_of_returning_zeros():
    calc = TargetCalc(
        AnalyticEpidemic(), ["i_max", "r_sup"], {"tlim_ini": 3, "tlim_final": 5, "tdelta": 3}
    )
    with pytest.raises(RuntimeError, match="1 samples did not finish"):
        calc.get_output(torch.tensor([[7.0]]), 1)
