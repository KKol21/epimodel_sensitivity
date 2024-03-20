from examples.SEIR_no_age_groups import seir_no_ag_main
from examples.contact_sensitivity import contact_main
from examples.SEIHR_2_age_group import seihr_2_ag_main


def call_main_functions():
    seir_no_ag_main.main()
    contact_main.main()
    seihr_2_ag_main.main()


call_main_functions()
