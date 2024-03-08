import itertools
import json
import os
from abc import ABC, abstractmethod

import numpy as np
from scipy import stats as ss

from src.dataloader import PROJECT_PATH
from src.model.r0 import R0Generator
from src.sensitivity.prcc import get_prcc_values


class SimulationBase(ABC):
    def __init__(self, data):
        # Load data
        self.data = data
        self.device = data.device
        self.test = True

        self._load_simulation_data()

    def _load_simulation_data(self):
        self.params = self.data.model_params
        self.n_age = self.data.n_age
        self.cm = self.data.cm
        self.population = self.data.age_data.flatten()
        self.age_vector = self.population.reshape((-1, 1))
        self.folder_name = PROJECT_PATH + "\sens_data"
        self.susceptibles = None

    def _load_config(self, path):
        with open(path) as f:
            config = json.load(f)
        self.target_vars = config["target_vars"]

        self.sim_options_dict = config["sim_options"]

        self.sim_options_prod = self.process_sim_options()

        self.sampled_params = config["sampled_params"]
        self.n_samples = config["n_samples"]
        self.batch_size = config["batch_size"]
        self.test = config["test"]
        self.init_vals = config["init_vals"]

    def process_sim_options(self):
        sim_opt = self.sim_options_dict

        if len(sim_opt) == 1:
            key = next(iter(sim_opt))
            return self.flatten_list_in_dict(sim_opt, key)

        def merge_dicts(ds):
            d_out = {key: value for d in ds for key, value in d.items()}
            return d_out

        flattened_options = [self.flatten_dict(sim_opt, key) for key in sim_opt.keys()]
        options_product = list(itertools.product(*flattened_options))
        return [merge_dicts(option) for option in options_product]

    def flatten_dict(self, d, key):
        if isinstance(d[key], dict):
            return self.flatten_dict_in_dict(d, key)
        else:
            return self.flatten_list_in_dict(d, key)

    @staticmethod
    def flatten_list_in_dict(d, key):
        return [{key: value} for value in d[key]]

    @staticmethod
    def flatten_dict_in_dict(d, key):
        return [{key: {subkey: value}} for subkey, value in d[key].items()]

    @abstractmethod
    def run_sampling(self):
        pass

    def run_func_for_all_configs(self, func):
        for option, target in itertools.product(self.sim_options_prod, self.target_vars):
            filename = self.get_filename(option) + f"_{target}"
            func(filename)

    def get_filename(self, option):
        return "_".join([self.parse_param_name(option, key) for key in option.keys()])

    def parse_param_name(self, option, key):
        if isinstance(option[key], dict):
            subkey = next(iter(option[key]))
            return f"{key}-{subkey}"
        else:
            return f"{key}-{option[key]}"

    def get_beta_from_r0(self, base_r0):
        r0generator = R0Generator(self.data)
        if isinstance(base_r0, tuple):
            base_r0 = base_r0[0]
        return base_r0 / r0generator.get_eig_val(contact_mtx=self.cm,
                                                 susceptibles=self.susceptibles.reshape(1, -1),
                                                 population=self.population)

    def calculate_prcc(self, filename):
        """

        Calculates PRCC (Partial Rank Correlation Coefficient) values from saved LHS tables and simulation results.

        This method reads the saved LHS tables and simulation results for each parameter combination and calculates
        the PRCC values. The PRCC values are saved in separate files in the 'sens_data_"folder_name"/prcc' directory.

        """
        folder_name = self.folder_name
        os.makedirs(f"{folder_name}/prcc", exist_ok=True)
        filename_without_target = filename[:filename[:filename.rfind("_")].rfind("_")]
        lhs_table = np.loadtxt(f'{folder_name}/lhs/lhs_{filename_without_target}.csv', delimiter=';')
        sim_output = np.loadtxt(f'{folder_name}/simulations/simulations_{filename}.csv', delimiter=';')

        prcc = get_prcc_values(np.c_[lhs_table, sim_output.T])
        np.savetxt(fname=f'{folder_name}/prcc/prcc_{filename}.csv', X=prcc)

    def calculate_p_values(self, filename, significance=0.05):
        os.makedirs(self.folder_name + '/p_values', exist_ok=True)
        prcc = np.loadtxt(fname=f'{self.folder_name}/prcc/prcc_{filename}.csv')
        t = prcc * np.sqrt((self.n_samples - 2 - self.n_age) / (1 - prcc ** 2))
        # p-value for 2-sided test
        dof = self.n_samples - 2 - self.n_age
        p_values = 2 * (1 - ss.t.cdf(x=abs(t), df=dof))
        np.savetxt(fname=f'{self.folder_name}/p_values/p_values_{filename}.csv', X=p_values)
        is_first = True
        if len(p_values) < 30:
            for idx, p_val in enumerate(p_values):
                if p_val > significance:
                    if is_first:
                        print("\nInsignificant p-values in ", filename, " case: \n")
                        is_first = False
                    print(f"\t {idx}. p-val: ", p_val)
