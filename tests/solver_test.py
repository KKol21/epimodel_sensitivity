import json
import os
from types import SimpleNamespace

import torch

from emsa.dataloader import PROJECT_PATH
from emsa.model.epidemic_model import EpidemicModel
from emsa_examples.utils.dataloader_16_ag import DataLoader
from tests.mock_models import *



def load_model_struct(rel_path):
    with open(os.path.join(PROJECT_PATH, rel_path)) as f:
        model_struct = json.load(f)
    return model_struct


seir_model_struct = load_model_struct("emsa_examples/SEIR_no_age_groups/configs/model_struct.json")
seihr_model_struct = load_model_struct("emsa_examples/SEIHR_2_age_groups/configs/model_struct.json")
contact_model_struct = load_model_struct("emsa_examples/contact_sensitivity/configs/model_struct.json")
vacc_model_struct = load_model_struct("emsa_examples/vaccinated_sensitivity/configs/model_struct.json")


def get_seir_data():
    data = DataLoader()
    contact_data_seir = torch.tensor(1)
    age_data_seir = torch.tensor([10000])
    return SimpleNamespace(**{"params": data.params,
                              "cm": contact_data_seir,
                              "age_data": age_data_seir,
                              "n_age": 1,
                              "device": "cpu"})


def get_seihr_data():
    data = DataLoader()
    seihr_params = data.params.copy()
    seihr_params["eta"] = torch.tensor([0.05, 0.15])
    contact_data_seihr = torch.tensor([[1, 2],
                                       [0.5, 1]])
    age_data_seihr = torch.tensor([1E5,
                                   2E5])
    return SimpleNamespace(**{"params": seihr_params,
                              "cm": contact_data_seihr,
                              "age_data": age_data_seihr,
                              "n_age": 2,
                              "device": "cpu"})


def test_model(mock_model: MockModelBase, model_struct: dict) -> None:
    model = EpidemicModel(data=mock_model.data, model_struct=model_struct)
    exposed_iv = torch.zeros(model.n_age)
    exposed_iv[0] = 10
    iv = {"e": exposed_iv} if "l" not in model.state_data else {"l": exposed_iv}
    y0 = torch.atleast_2d(model.get_initial_values_from_dict(iv))
    t_eval = torch.atleast_2d(torch.arange(0, 200, 1))
    model.initialize_matrices()
    emsa_sol = model.get_solution(y0, t_eval).ys
    mock_sol = model.get_sol_from_ode(y0, t_eval, odefun=mock_model.odefun).ys
    print((mock_sol[:, -1] - emsa_sol[:, -1]).sum() < 1)


def run_tests():
    seir_data = get_seir_data()
    seihr_data = get_seihr_data()
    models = [MockSEIRModel(data=seir_data),
              MockSEIHRModel(data=seihr_data),
              MockContactModel(data=DataLoader()),
              MockVaccinatedModel(data=DataLoader())]
    model_structures = [seir_model_struct,
                        seihr_model_struct,
                        contact_model_struct,
                        vacc_model_struct
                        ]
    for model, structure in zip(models, model_structures):
        test_model(mock_model=model, model_struct=structure)

run_tests()
