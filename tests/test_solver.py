import json
import os
from types import SimpleNamespace

import pytest
import torch

from emsa.dataloader import PROJECT_PATH
from emsa.model.epidemic_model import EpidemicModel
from emsa_examples.utils.dataloader_16_ag import DataLoader
from tests.mock_models import MockSEIRModel, MockSEIHRModel, MockContactModel, MockVaccinatedModel


def load_model_struct(rel_path):
    with open(os.path.join(PROJECT_PATH, rel_path)) as f:
        model_struct = json.load(f)
    return model_struct


@pytest.fixture(scope="module")
def model_structs():
    seir_model_struct = load_model_struct(
        "emsa_examples/SEIR_no_age_groups/configs/model_struct.json"
    )
    seihr_model_struct = load_model_struct(
        "emsa_examples/SEIHR_2_age_groups/configs/model_struct.json"
    )
    contact_model_struct = load_model_struct(
        "emsa_examples/contact_sensitivity/configs/model_struct.json"
    )
    vacc_model_struct = load_model_struct(
        "emsa_examples/vaccinated_sensitivity/configs/model_struct.json"
    )
    return {
        "seir": seir_model_struct,
        "seihr": seihr_model_struct,
        "contact": contact_model_struct,
        "vacc": vacc_model_struct,
    }


@pytest.fixture
def seir_data():
    data = DataLoader()
    contact_data_seir = torch.tensor(1)
    age_data_seir = torch.tensor([10000])
    return SimpleNamespace(
        **{
            "params": data.params,
            "cm": contact_data_seir,
            "age_data": age_data_seir,
            "n_age": 1,
            "device": "cpu",
        }
    )


@pytest.fixture
def seihr_data():
    data = DataLoader()
    seihr_params = data.params.copy()
    seihr_params["eta"] = torch.tensor([0.1, 0.3])
    contact_data_seihr = torch.tensor([[1, 2], [0.5, 1]])
    age_data_seihr = torch.tensor([1e5, 2e5])
    return SimpleNamespace(
        **{
            "params": seihr_params,
            "cm": contact_data_seihr,
            "age_data": age_data_seihr,
            "n_age": 2,
            "device": "cpu",
        }
    )


@pytest.fixture
def contact_data():
    return DataLoader()


@pytest.fixture
def vaccinated_data():
    return DataLoader()


def test_seir_model(seir_data, model_structs):
    _test_model(MockSEIRModel(data=seir_data), model_structs["seir"])


def test_seihr_model(seihr_data, model_structs):
    _test_model(MockSEIHRModel(data=seihr_data), model_structs["seihr"])


def test_contact_model(contact_data, model_structs):
    _test_model(MockContactModel(data=contact_data), model_structs["contact"])


def test_vaccinated_model(vaccinated_data, model_structs):
    _test_model(MockVaccinatedModel(data=vaccinated_data), model_structs["vacc"])


def _test_model(mock_model, model_struct):
    model = EpidemicModel(data=mock_model.data, model_struct=model_struct)
    exposed_iv = torch.zeros(model.n_age)
    exposed_iv[0] = 10
    iv = {"e": exposed_iv} if "l" not in model.state_data else {"l": exposed_iv}
    y0 = torch.atleast_2d(model.get_initial_values_from_dict(iv))
    t_eval = torch.atleast_2d(torch.arange(0, 200, 1))
    model.initialize_matrices()
    emsa_sol = model.get_solution(y0, t_eval).ys
    mock_sol = model.get_sol_from_ode(y0, t_eval, odefun=mock_model.odefun).ys
    assert (
        torch.abs(mock_sol[:, -1] - emsa_sol[:, -1]).sum().item() < 1
    ), "Model solutions do not match"


if __name__ == "__main__":
    pytest.main(["-v"])
