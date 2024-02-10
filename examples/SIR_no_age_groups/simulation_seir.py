import itertools

import torch

from examples.SIR_no_age_groups.seir_model import SEIRModel
from examples.SIR_no_age_groups.seir_sampler import SamplerSEIR
from src.model.r0 import R0Generator
from src.simulation_base import SimulationBase


class SimulationSEIR(SimulationBase):
    def __init__(self, data):
        super().__init__(data)
        self.folder_name += "/sens_data_SEIR_no_ag"

        # Initalize model
        self.model = SEIRModel(sim_obj=self)
        self.susceptibles = self.model.get_initial_values()[self.model.idx("s_0")]

        # User-defined params
        self.susc_choices = [1.0]
        self.r0_choices = [1.8]
        self.target_var_choices = ["i_max"]  # ["i_max", "ic_max", "d_max"]
        self.simulations = list(itertools.product(self.susc_choices, self.r0_choices, self.target_var_choices))

    def run_sampling(self):
        """

        Runs the sampling-based simulation with different parameter combinations.

        This method generates Latin Hypercube Sampling (LHS) samples of vaccine distributions for each parameter
        combination. The LHS tables and simulation results are saved in separate files in the 'sens_data_contact/lhs' and
        'sens_data_contact/simulations' directories, respectively.

        """
        susceptibility = torch.ones(self.n_age).to(self.data.device)
        for susc, base_r0, target_var in self.simulations:
            susceptibility[:4] = susc
            self.params.update({"susc": susceptibility})
            r0generator = R0Generator(self.data)
            # Calculate base transmission rate
            beta = base_r0 / r0generator.get_eig_val(contact_mtx=self.cm,
                                                     susceptibles=self.susceptibles.reshape(1, -1),
                                                     population=self.population)
            self.params.update({"beta": beta})
            # Generate matrices used in model representation
            self.model.initialize_matrices()
            sim_state = {"base_r0": base_r0,
                         "susc": susc,
                         "r0generator": r0generator,
                         "target_var": target_var}
            self.model.sim_state = sim_state
            param_generator = SamplerSEIR(sim_state=sim_state,
                                          sim_obj=self)
            param_generator.run_sampling()
