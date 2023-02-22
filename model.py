from abc import ABC, abstractmethod

import numpy as np
from scipy.integrate import odeint


class EpidemicModelBase(ABC):
    def __init__(self, model_data, compartments):
        self.population = model_data.age_data.flatten()
        self.compartments = compartments
        self.c_idx = {comp: idx for idx, comp in enumerate(self.compartments)}
        self.n_age = self.population.shape[0]

    def initialize(self):
        iv = {key: np.zeros(self.n_age) for key in self.compartments}
        return iv

    def aggregate_by_age(self, solution, idx):
        return np.sum(solution[:, idx * self.n_age:(idx + 1) * self.n_age], axis=1)

    def get_cumulative(self, solution):
        idx = self.c_idx["c"]
        return self.aggregate_by_age(solution, idx)

    def get_deaths(self, solution):
        idx = self.c_idx["d"]
        return self.aggregate_by_age(solution, idx)

    def get_solution(self, t, parameters, cm):
        initial_values = self.get_initial_values()
        return np.array(odeint(self.get_model, initial_values, t, args=(parameters, cm)))

    def get_array_from_dict(self, comp_dict):
        return np.array([comp_dict[comp] for comp in self.compartments]).flatten()

    def get_initial_values(self):
        iv = self.initialize()
        self.update_initial_values(iv=iv)
        return self.get_array_from_dict(comp_dict=iv)

    @abstractmethod
    def update_initial_values(self, iv):
        pass

    @abstractmethod
    def get_model(self, xs, ts, ps, cm):
        pass


class VaccinatedModel(EpidemicModelBase):
    def __init__(self, model_data):
        self.n_vac_states = model_data.model_parameters_data["n_vac_states"]
        compartments = ["s", "e", "i", "r"] + self.get_vac_compartments(self.n_vac_states)
        super().__init__(model_data=model_data, compartments=compartments)

    @staticmethod
    def get_vac_compartments(n_classes):
        return [f"v_{i}" for i in range(n_classes)]

    def update_initial_values(self, iv):
        iv.update({
            "s": self.population - (iv["e"] + iv["i"])
        })

    def get_model(self, xs, ts, ps, cm):
        # the same order as in self.compartments!
        vac_state_val = dict()
        val = xs.reshape(-1, self.n_age)
        for idx, comp in enumerate(self.compartments[4:], 4):
            vac_state_val[comp] = val[idx]
        s, e, i, r = val[:4]

        transmission = ps["beta_0"] * np.array(i).dot(cm)
        actual_population = self.population
        vacc = np.array(ps["t_start"] < ts < (ps["t_start"] + ps["T"])).astype(float)

        model_eq_dict = {
            "s": - ps["susc"] * (s / actual_population) * transmission
            - ps["v"] * ps["rho"] * s / (s + r) * vacc
            + ps["psi"] * vac_state_val[self.compartments[-1]],                          # S'(t)
            "e": ps["susc"] * (s / actual_population) * transmission - ps["alpha"] * e,  # E'(t)
            "i": ps["alpha"] * e - ps["gamma"] * i,                                      # I'(t)
            "r": ps["gamma"] * i                                                         # R'(t)
        }

        vac_eq_dict = dict()
        vac_eq_dict["v_0"] = ps["v"] * ps["rho"] * s / (s + r) * vacc \
                             - vac_state_val["v_0"] * ps["psi"]                   # V_0'(t)
        for idx, state in enumerate(self.compartments[5:], 1):
            prev_state = vac_state_val[f"v_{idx-1}"]
            vac_eq_dict[state] = (prev_state - vac_state_val[state]) * ps["psi"]  # V_i'(t)

        model_eq_dict.update(vac_eq_dict)
        return self.get_array_from_dict(comp_dict=model_eq_dict)
