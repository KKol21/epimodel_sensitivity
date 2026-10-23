import matplotlib.pyplot as plt
import pytest

from emsa_examples.SEIHR_2_age_groups import seihr_2_ag_main
from emsa_examples.SEIR_no_age_groups import seir_no_ag_main
from emsa_examples.SIR_no_age_groups import sir_no_ag_main
from emsa_examples.contact_sensitivity import contact_main
from emsa_examples.vaccinated_sensitivity import vaccinated_main
from emsa.utils import SimulationBase


@pytest.fixture(autouse=True)
def setup(tmp_path, monkeypatch):
    # Turn off interactive mode
    plt.ioff()
    # Use Agg backend (to avoid getting flooded with figures)
    plt.switch_backend("Agg")
    monkeypatch.setattr(plt, "show", lambda: None)
    load_simulation_data = SimulationBase._load_simulation_data
    load_config = SimulationBase._load_config

    def load_test_data(sim):
        load_simulation_data(sim)
        sim.folder_name = str(tmp_path)

    def load_test_config(sim, config_path):
        load_config(sim, config_path)
        sim.n_samples = min(sim.n_samples, 2000)
        sim.batch_size = min(sim.batch_size, sim.n_samples)

    monkeypatch.setattr(SimulationBase, "_load_simulation_data", load_test_data)
    monkeypatch.setattr(SimulationBase, "_load_config", load_test_config)
    yield
    plt.close("all")


def test_sir_no_ag_main():
    sir_no_ag_main.main()


def test_seir_no_ag_main():
    seir_no_ag_main.main()


def test_contact_main():
    contact_main.main()


def test_seihr_2_ag_main():
    seihr_2_ag_main.main()


def test_vaccinated_main():
    vaccinated_main.main()


if __name__ == "__main__":
    pytest.main(["-v"])
