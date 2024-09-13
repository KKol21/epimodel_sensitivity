import matplotlib.pyplot as plt
import pytest

from emsa_examples.SEIHR_2_age_groups import seihr_2_ag_main
from emsa_examples.SEIR_no_age_groups import seir_no_ag_main
from emsa_examples.SIR_no_age_groups import SIR_no_ag_main
from emsa_examples.contact_sensitivity import contact_main
from emsa_examples.vaccinated_sensitivity import vaccinated_main


@pytest.fixture(autouse=True)
def setup():
    # Turn off interactive mode
    plt.ioff()
    # Use Agg backend (does not require a display)
    plt.switch_backend("Agg")


def test_sir_no_ag_main():
    SIR_no_ag_main.main()


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
