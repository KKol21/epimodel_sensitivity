import itertools
import json
import os
from copy import deepcopy
from importlib.metadata import version, PackageNotFoundError
from abc import ABC, abstractmethod

import numpy as np
from scipy import stats as ss
from torch import atleast_2d

from .dataloader import PROJECT_PATH
from .plotter import generate_tornado_plot
from .results import join_result_name, load_result


class SimulationBase(ABC):
    def __init__(
        self,
        data,
        model_struct_path,
        sampling_config_path,
        *,
        output_dir=None,
        save_results=True,
    ):
        # Load data
        self.data = data
        self.device = data.device
        self.model = None
        self.results = {}
        self.save_results = save_results

        self._load_simulation_data()
        self._load_config(sampling_config_path)
        self._load_model_structure(model_struct_path)
        population = np.asarray(self.population.cpu())
        contacts = np.asarray(self.cm.cpu())
        if not np.isfinite(population).all() or np.any(population <= 0):
            raise ValueError("Age populations must be finite and positive")
        if (
            contacts.shape != (self.n_age, self.n_age)
            or not np.isfinite(contacts).all()
            or np.any(contacts < 0)
        ):
            raise ValueError("Contact matrix must be finite, nonnegative and match age populations")
        for target in self.target_vars:
            comp, _, ending = target.rpartition("_")
            if target != "r0" and (
                comp not in self.model_struct["state_data"] or ending not in ("max", "sup")
            ):
                raise ValueError(f"Unsupported target: {target}")
        if not self.target_vars:
            raise ValueError("At least one target is required")
        if output_dir is not None:
            self.folder_name = os.fspath(output_dir)

    @property
    def susceptibles(self):
        return self.model.get_initial_values()[self.model.idx("s_0")]

    def _load_simulation_data(self):
        self.params = self.data.params
        self.cm = atleast_2d(self.data.cm)
        self.population = self.data.age_data.flatten()
        self.n_age = len(self.population)
        self.folder_name = os.path.join(os.getcwd(), "sens_data")

    def _load_model_structure(self, model_struct_path):
        if isinstance(model_struct_path, dict):
            self.model_struct = deepcopy(model_struct_path)
        else:
            with open(model_struct_path) as f:
                self.model_struct = json.load(f)

    def _load_config(self, config_path):
        if isinstance(config_path, dict):
            config = deepcopy(config_path)
        else:
            with open(config_path) as f:
                config = json.load(f)
        self.config = config
        self.seed = config.get("seed")
        if self.seed is not None and (not isinstance(self.seed, int) or not 0 <= self.seed < 2**32):
            raise ValueError("seed must be an integer between 0 and 2**32 - 1")
        self.solver_config = dict(config.get("solver", {}))

        self.target_vars = config["target_vars"]

        self.variable_params_dict = config.get("variable_params", {})

        self.variable_param_combinations = self.process_variable_params()

        self.sampled_params_boundaries = config.get("sampled_params_boundaries") or {}
        self.n_samples = config["n_samples"]
        self.batch_size = config["batch_size"]
        if any(not isinstance(v, int) or v < 1 for v in (self.n_samples, self.batch_size)):
            raise ValueError("n_samples and batch_size must be positive integers")

        self.test = config.get("is_static", True)
        self.init_vals = config["init_vals"]

        self.target_calc_config = {
            "tlim_ini": config.get("tlim_ini", 300),
            "tlim_final": config.get("tlim_final", 5000),
            "tdelta": config.get("tdelta", 50),
        }
        if any(not isinstance(v, int) or v < 1 for v in self.target_calc_config.values()):
            raise ValueError("Target time limits must be positive integers")

    def run(self):
        """Sample and analyze in memory; export CSV/metadata when save_results is true.

        Returns a mapping of scenario names to samples, targets, PRCC, p-values and metadata.
        Each invocation replaces the previous in-memory results.
        """
        from emsa.sensitivity import get_prcc_values

        self.results = {}
        self.run_sampling()
        for filename, result in self.results.items():
            result["prcc"] = {}
            result["p_values"] = {}
            for target, values in result["targets"].items():
                prcc = get_prcc_values(np.c_[result["samples"], values])
                p_values = self.prcc_p_values(prcc, len(result["samples"]))
                result["prcc"][target] = prcc
                result["p_values"][target] = p_values
                if self.save_results:
                    for name, array in (("prcc", prcc), ("p_values", p_values)):
                        directory = os.path.join(self.folder_name, name)
                        os.makedirs(directory, exist_ok=True)
                        np.savetxt(
                            os.path.join(
                                directory, join_result_name(name, filename, target) + ".csv"
                            ),
                            array,
                        )
        return self.results

    @staticmethod
    def prcc_p_values(prcc, n_samples):
        prcc = np.atleast_1d(prcc)
        dof = n_samples - len(prcc) - 1
        if dof <= 0:
            raise ValueError("More samples are needed to calculate PRCC p-values")
        with np.errstate(divide="ignore", invalid="ignore"):
            t = prcc * np.sqrt(dof / np.maximum(1 - prcc**2, 0))
        return 2 * ss.t.sf(np.abs(t), df=dof)

    def get_run_metadata(self, variable_params):
        import torch

        def serializable(value):
            if torch.is_tensor(value):
                return value.detach().cpu().tolist()
            if isinstance(value, dict):
                return {key: serializable(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [serializable(item) for item in value]
            return value

        try:
            package_version = version("emsa")
        except PackageNotFoundError:
            package_version = "uninstalled"
        return {
            "package_version": package_version,
            "seed": self.seed,
            "config": deepcopy(self.config),
            "model_struct": deepcopy(self.model_struct),
            "variable_params": serializable(variable_params),
            "parameters": serializable(self.params),
            "population": serializable(self.population),
            "contact_matrix": serializable(self.cm),
            "n_samples": self.n_samples,
            "batch_size": self.batch_size,
            "solver": {
                "method": "euler",
                "step_size": 1.0,
                "rtol": 1e-5,
                "atol": 1e-6,
                "max_steps": 100000,
                "negative_tolerance": 1e-4,
                **self.solver_config,
            },
        }

    def process_variable_params(self):
        vpd = self.variable_params_dict

        if len(vpd) == 1:
            key = next(iter(vpd))
            return self.flatten_dict(vpd, key)

        flattened_vpd = [self.flatten_dict(vpd, key) for key in vpd.keys()]
        variable_params_product = list(itertools.product(*flattened_vpd))
        return [self.merge_dicts(variable_params) for variable_params in variable_params_product]

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

    @staticmethod
    def merge_dicts(ds):
        d_out = {key: value for d in ds for key, value in d.items()}
        return d_out

    @abstractmethod
    def run_sampling(self):
        pass

    def run_func_for_all_configs(self, func):
        for variable_params, target in itertools.product(
            self.variable_param_combinations, self.target_vars
        ):
            filename = join_result_name(self.get_filename(variable_params), target)
            func(filename)

    def get_filename(self, variable_params):
        return (
            "_".join(
                [self.parse_param_name(variable_params, key) for key in variable_params.keys()]
            )
            if variable_params
            else ""
        )

    @staticmethod
    def parse_param_name(variable_params: dict, key):
        if isinstance(variable_params[key], dict):
            subkey = next(iter(variable_params[key]))
            return f"{key}-{subkey}"
        else:
            return f"{key}-{variable_params[key]}"

    def calculate_prcc(self, filename: str, target: str) -> None:
        """

        Calculates PRCC (Partial Rank Correlation Coefficient) values from saved LHS tables and simulation results.

        This method reads the saved LHS tables and simulation results for each parameter combination and calculates
        the PRCC values. The results are saved as CSV files in the ``prcc`` subdirectory
        of ``self.folder_name``.

        """
        from emsa.sensitivity import get_prcc_values

        folder_name = self.folder_name
        os.makedirs(os.path.join(folder_name, "prcc"), exist_ok=True)
        lhs_table = load_result(folder_name, "lhs", filename, ndmin=2)
        sim_output = load_result(folder_name, "simulations", filename, target)

        prcc = get_prcc_values(np.c_[lhs_table, sim_output.T])

        prcc_path = os.path.join(
            folder_name, "prcc/" + join_result_name("prcc", filename, target) + ".csv"
        )
        np.savetxt(fname=prcc_path, X=prcc)

    def calculate_p_values(self, filename, significance=0.05):
        p_values_dir = os.path.join(self.folder_name, "p_values")
        os.makedirs(p_values_dir, exist_ok=True)

        prcc = np.atleast_1d(load_result(self.folder_name, "prcc", filename))
        p_values = SimulationBase.prcc_p_values(prcc, self.n_samples)

        p_values_path = os.path.join(
            self.folder_name, "p_values/" + join_result_name("p_values", filename) + ".csv"
        )
        np.savetxt(fname=p_values_path, X=p_values)

        is_first = True
        if len(p_values) < 30:
            for idx, p_val in enumerate(p_values):
                if p_val > significance:
                    if is_first:
                        print("\nInsignificant p-values in ", filename, " case: \n")
                        is_first = False
                    print(f"\t {idx}. p-val: ", p_val)

    def plot_prcc(self, filename: str, labels=None):
        """

        Args:
            filename:
            labels:

        Returns: None

        """
        spb = self.sampled_params_boundaries
        from emsa.sensitivity.sensitivity_model_base import get_params_col_idx

        def get_aged_param_labels(aged_param):
            return [f"{aged_param}_{ag}" for ag in range(self.n_age)]

        if labels is None:
            labels = []
            pci = get_params_col_idx(sampled_params_boundaries=spb)
            for param, idx in pci.items():
                param_label = (
                    get_aged_param_labels(param) if isinstance(spb[param][0], list) else param
                )
                if isinstance(param_label, list):
                    labels += param_label
                else:
                    labels.append(param_label)

        prcc_plots_dir = os.path.join(self.folder_name, "prcc_plots")
        os.makedirs(prcc_plots_dir, exist_ok=True)

        prcc = load_result(self.folder_name, "prcc", filename)
        p_val = load_result(self.folder_name, "p_values", filename)

        generate_tornado_plot(
            sim_object=self, labels=labels, prcc=prcc, p_val=p_val, filename=filename
        )

    def calculate_all_prcc(self):
        for variable_params, target in itertools.product(
            self.variable_param_combinations, self.target_vars
        ):
            filename = self.get_filename(variable_params)
            self.calculate_prcc(filename=filename, target=target)

    def calculate_all_p_values(self):
        self.run_func_for_all_configs(self.calculate_p_values)

    def plot_all_prcc(self):
        self.run_func_for_all_configs(self.plot_prcc)
