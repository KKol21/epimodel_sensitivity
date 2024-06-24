import unittest

from emsa_examples.SEIHR_2_age_groups import seihr_2_ag_main
from emsa_examples.SEIR_no_age_groups import seir_no_ag_main
from emsa_examples.contact_sensitivity import contact_main
from emsa_examples.vaccinated_sensitivity import vaccinated_main


class TestMainFunctions(unittest.TestCase):
    def test_seir_no_ag_main(self):
        seir_no_ag_main.main()

    def test_contact_main(self):
        contact_main.main()

    def test_seihr_2_ag_main(self):
        seihr_2_ag_main.main()

    def test_vaccinated_main(self):
        vaccinated_main.main()


if __name__ == '__main__':
    unittest.main(verbosity=1)
