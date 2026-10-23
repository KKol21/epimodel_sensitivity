import json
import os
from types import SimpleNamespace

import pytest
import torch

from emsa.model import EpidemicModel
from emsa.utils import PROJECT_PATH
from emsa_examples.utils.dataloader_16_ag import DataLoader
from tests.mock_models import (
    MockModelBase,
    MockSIRModel,
    MockSEIRModel,
    MockSEIHRModel,
    MockContactModel,
    MockVaccinatedModel,
    EMSAMockVaccinatedModel,
)

# Constants for file paths
SIR_CONFIG_PATH = "emsa_examples/SIR_no_age_groups/configs/model_struct.json"
SEIR_CONFIG_PATH = "emsa_examples/SEIR_no_age_groups/configs/model_struct.json"
SEIHR_CONFIG_PATH = "emsa_examples/SEIHR_2_age_groups/configs/model_struct.json"
CONTACT_CONFIG_PATH = "emsa_examples/contact_sensitivity/configs/model_struct.json"
VACCINATED_CONFIG_PATH = "emsa_examples/vaccinated_sensitivity/configs/model_struct.json"


def load_model_struct(rel_path):
    """Load model structure from a JSON file."""
    with open(os.path.join(PROJECT_PATH, rel_path)) as f:
        model_struct = json.load(f)
    return model_struct


@pytest.fixture(scope="module")
def model_structs():
    """Load all model structures for the tests."""
    return {
        "sir": load_model_struct(SIR_CONFIG_PATH),
        "seir": load_model_struct(SEIR_CONFIG_PATH),
        "seihr": load_model_struct(SEIHR_CONFIG_PATH),
        "contact": load_model_struct(CONTACT_CONFIG_PATH),
        "vacc": load_model_struct(VACCINATED_CONFIG_PATH),
    }


@pytest.fixture
def sir_data():
    return SimpleNamespace(
        params={"beta": 0.3, "gamma": 0.2},
        cm=torch.tensor(1.0),
        age_data=torch.tensor([10000.0]),
        n_age=1,
        device="cpu",
    )


@pytest.fixture
def seir_data():
    """Fixture for SEIR model data."""
    data = DataLoader()
    contact_data_seir = torch.tensor(1)
    age_data_seir = torch.tensor([10000])
    return SimpleNamespace(
        params=data.params,
        cm=contact_data_seir,
        age_data=age_data_seir,
        n_age=1,
        device="cpu",
    )


@pytest.fixture
def seihr_data():
    """Fixture for SEIHR model data."""
    seihr_params = {
        "beta": 0.2,
        "alpha": 0.3,
        "gamma": 0.2,
        "gamma_h": 0.1,
        "eta": torch.tensor([0.1, 0.3]),
        "susc": torch.tensor([0.5, 1.0]),
    }
    contact_data_seihr = torch.tensor([[1, 2], [0.5, 1]])
    age_data_seihr = torch.tensor([1e5, 2e5])
    return SimpleNamespace(
        params=seihr_params,
        cm=contact_data_seihr,
        age_data=age_data_seihr,
        n_age=2,
        device="cpu",
    )


@pytest.fixture
def contact_data():
    """Fixture for contact model data."""
    return DataLoader()


@pytest.fixture
def vaccinated_data():
    """Fixture for vaccinated model data."""
    return DataLoader()


EXAMPLE_MODELS = [
    (MockSIRModel, "sir_data", "sir"),
    (MockSEIRModel, "seir_data", "seir"),
    (MockSEIHRModel, "seihr_data", "seihr"),
    (MockContactModel, "contact_data", "contact"),
    (MockVaccinatedModel, "vaccinated_data", "vacc"),
]


@pytest.mark.parametrize("model_cls, data_fixture, model_key", EXAMPLE_MODELS)
def test_models(model_cls, data_fixture, model_key, request, model_structs):
    """Parameterized test for different models."""
    data = request.getfixturevalue(data_fixture)
    _test_model(model_cls(data=data), model_structs[model_key])


def _test_model(mock_model: MockModelBase, model_struct):
    """Helper function to test a model."""
    is_vacc = isinstance(mock_model, MockVaccinatedModel)
    if is_vacc:
        model = EMSAMockVaccinatedModel(data=mock_model.data, model_struct=model_struct)
    else:
        model = EpidemicModel(data=mock_model.data, model_struct=model_struct)

    exposed_iv = torch.zeros(model.n_age)
    exposed_iv[0] = 10
    initial_compartment = "e" if "e" in model.state_data else "l"
    if initial_compartment not in model.state_data:
        initial_compartment = "i"
    iv = {initial_compartment: exposed_iv}
    y0 = torch.atleast_2d(model.get_initial_values_from_dict(iv))
    t_eval = torch.atleast_2d(torch.arange(0, 200, 1))
    model.initialize_matrices()
    emsa_sol = (
        model.get_solution(y0, t_eval).ys
        if not is_vacc
        else model.get_solution(y0, t_eval, odefun=model.get_vacc_ode()).ys
    )
    mock_sol = model.get_sol_from_ode(y0, t_eval, odefun=mock_model.odefun).ys
    torch.testing.assert_close(emsa_sol, mock_sol, rtol=1e-4, atol=0.1)


@pytest.mark.parametrize("model_cls, data_fixture, model_key", EXAMPLE_MODELS)
def test_model_derivatives(model_cls, data_fixture, model_key, request, model_structs):
    """Exercise every transition with nonzero states, including transient compartments."""
    data = request.getfixturevalue(data_fixture)
    mock_model = model_cls(data=data)
    is_vacc = isinstance(mock_model, MockVaccinatedModel)
    model_type = EMSAMockVaccinatedModel if is_vacc else EpidemicModel
    model = model_type(data=data, model_struct=model_structs[model_key])
    model.initialize_matrices()

    weights = torch.arange(1, model.n_eq + 1, dtype=torch.float32).reshape(model.n_age, -1)
    y = (weights / weights.sum(dim=1, keepdim=True) * model.population[:, None]).flatten()[None, :]
    t = torch.tensor([1.0])
    odefun = model.get_vacc_ode() if is_vacc else model.basic_ode
    actual = odefun(t, y)
    expected = mock_model.odefun(t, y).reshape_as(actual)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=1e-3)
    torch.testing.assert_close(
        actual.reshape(model.n_age, -1).sum(dim=1),
        torch.zeros(model.n_age),
        rtol=0,
        atol=0.02,
    )


if __name__ == "__main__":
    pytest.main(["-v"])
