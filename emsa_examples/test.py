from emsa_examples.SEIHR_2_age_groups import seihr_2_ag_main
from emsa_examples.SEIR_no_age_groups import seir_no_ag_main
from emsa_examples.contact_sensitivity import contact_main
from emsa_examples.vaccinated_sensitivity import vaccinated_main


def call_main_functions():
    seir_no_ag_main.main()
    contact_main.main()
    seihr_2_ag_main.main()
    vaccinated_main.main()


call_main_functions()
