import json
import os
import unittest
from types import SimpleNamespace

import torch

from emsa.dataloader import PROJECT_PATH
from emsa.model.epidemic_model import EpidemicModel
from emsa_examples.utils.dataloader_16_ag import DataLoader
from tests.mock_models import MockSEIRModel, MockSEIHRModel, MockContactModel, MockVaccinatedModel


def load_model_struct(rel_path):
    with open(os.path.join(PROJECT_PATH, rel_path)) as f:
        model_struct = json.load(f)
    return model_struct


class TestEpidemicModels(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.seir_model_struct = load_model_struct("emsa_examples/SEIR_no_age_groups/configs/model_struct.json")
        cls.seihr_model_struct = load_model_struct("emsa_examples/SEIHR_2_age_groups/configs/model_struct.json")
        cls.contact_model_struct = load_model_struct("emsa_examples/contact_sensitivity/configs/model_struct.json")
        cls.vacc_model_struct = load_model_struct("emsa_examples/vaccinated_sensitivity/configs/model_struct.json")

    def setUp(self):
        self.seir_data = self.get_seir_data()
        self.seihr_data = self.get_seihr_data()
        self.contact_data = DataLoader()
        self.vaccinated_data = DataLoader()

    @staticmethod
    def get_seir_data():
        data = DataLoader()
        contact_data_seir = torch.tensor(1)
        age_data_seir = torch.tensor([10000])
        return SimpleNamespace(**{"params": data.params,
                                  "cm": contact_data_seir,
                                  "age_data": age_data_seir,
                                  "n_age": 1,
                                  "device": "cpu"})

    @staticmethod
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

    def test_seir_model(self):
        self._test_model(MockSEIRModel(data=self.seir_data), self.seir_model_struct)

    def test_seihr_model(self):
        self._test_model(MockSEIHRModel(data=self.seihr_data), self.seihr_model_struct)

    def test_contact_model(self):
        self._test_model(MockContactModel(data=self.contact_data), self.contact_model_struct)

    def test_vaccinated_model(self):
        self._test_model(MockVaccinatedModel(data=self.vaccinated_data), self.vacc_model_struct)

    def _test_model(self, mock_model, model_struct):
        model = EpidemicModel(data=mock_model.data, model_struct=model_struct)
        exposed_iv = torch.zeros(model.n_age)
        exposed_iv[0] = 10
        iv = {"e": exposed_iv} if "l" not in model.state_data else {"l": exposed_iv}
        y0 = torch.atleast_2d(model.get_initial_values_from_dict(iv))
        t_eval = torch.atleast_2d(torch.arange(0, 200, 1))
        model.initialize_matrices()
        emsa_sol = model.get_solution(y0, t_eval).ys
        mock_sol = model.get_sol_from_ode(y0, t_eval, odefun=mock_model.odefun).ys
        self.assertTrue((mock_sol[:, -1] - emsa_sol[:, -1]).sum().item() < 1, "Model solutions do not match")


if __name__ == "__main__":
    unittest.main(verbosity=1)
