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


class RostModelHungary(EpidemicModelBase):
    def __init__(self, model_data):
        compartments = ["s", "l1", "l2",
                        "ip", "ia1", "ia2", "ia3",
                        "is1", "is2", "is3",
                        "ih", "ic", "icr",
                        "r", "d", "c"]
        super().__init__(model_data=model_data, compartments=compartments)

    def update_initial_values(self, iv):
        iv["l1"][2] = 1  # np.array([0, 0, 0, 4, 3, 3, 1, 2, 1, 2, 2, 2, 5, 5, 0, 0])
        iv.update({
            "c": iv["ip"] + iv["ia1"] + iv["ia2"] + iv["ia3"] + iv["is1"] + iv["is2"] + iv["is3"] + iv["r"] + iv["d"]
        })
        iv.update({
            "s": self.population - (iv["c"] + iv["l1"] + iv["l2"])
        })

    def get_model(self, xs, _, ps, cm):
        # the same order as in self.compartments!
        s, l1, l2, ip, ia1, ia2, ia3, is1, is2, is3, ih, ic, icr, r, d, c = xs.reshape(-1, self.n_age)

        transmission = ps["beta"] * np.array((ip + ps["inf_a"] * (ia1 + ia2 + ia3) + (is1 + is2 + is3))).dot(cm)
        actual_population = self.population

        model_eq_dict = {
            "s": -ps["susc"] * (s / actual_population) * transmission,  # S'(t)
            "l1": ps["susc"] * (s / actual_population) * transmission - 2 * ps["alpha_l"] * l1,  # L1'(t)
            "l2": 2 * ps["alpha_l"] * l1 - 2 * ps["alpha_l"] * l2,  # L2'(t)

            "ip": 2 * ps["alpha_l"] * l2 - ps["alpha_p"] * ip,  # Ip'(t)

            "ia1": ps["p"] * ps["alpha_p"] * ip - 3 * ps["gamma_a"] * ia1,  # Ia1'(t)
            "ia2": 3 * ps["gamma_a"] * ia1 - 3 * ps["gamma_a"] * ia2,  # Ia2'(t)
            "ia3": 3 * ps["gamma_a"] * ia2 - 3 * ps["gamma_a"] * ia3,  # Ia3'(t)

            "is1": (1 - ps["p"]) * ps["alpha_p"] * ip - 3 * ps["gamma_s"] * is1,  # Is1'(t)
            "is2": 3 * ps["gamma_s"] * is1 - 3 * ps["gamma_s"] * is2,  # Is2'(t)
            "is3": 3 * ps["gamma_s"] * is2 - 3 * ps["gamma_s"] * is3,  # Is3'(t)

            "ih": ps["h"] * (1 - ps["xi"]) * 3 * ps["gamma_s"] * is3 - ps["gamma_h"] * ih,  # Ih'(t)
            "ic": ps["h"] * ps["xi"] * 3 * ps["gamma_s"] * is3 - ps["gamma_c"] * ic,  # Ic'(t)
            "icr": (1 - ps["mu"]) * ps["gamma_c"] * ic - ps["gamma_cr"] * icr,  # Icr'(t)

            "r": 3 * ps["gamma_a"] * ia3 + (1 - ps["h"]) * 3 * ps["gamma_s"] * is3
            + ps["gamma_h"] * ih + ps["gamma_cr"] * icr,  # R'(t)
            "d": ps["mu"] * ps["gamma_c"] * ic,  # D'(t)

            "c": 2 * ps["alpha_l"] * l2  # C'(t)
        }

        return self.get_array_from_dict(comp_dict=model_eq_dict)

    def get_hospitalized(self, solution):
        idx = self.c_idx["ih"]
        idx_2 = self.c_idx["icr"]
        return self.aggregate_by_age(solution, idx) + self.aggregate_by_age(solution, idx_2)

    def get_ventilated(self, solution):
        idx = self.c_idx["ic"]
        return self.aggregate_by_age(solution, idx)


class VaccinatedModel(EpidemicModelBase):
    def __init__(self, model_data):
        compartments = self.get_compartments(model_data.n_classes)
        super().__init__(model_data=model_data, compartments=compartments)

    @abstractmethod
    def get_compartments(n_classes):
        vac = [f"v_{i}" for i in range(n_classes)]
        return ["S", "E", "I", "R"] + vac

    def update_initial_values(self, iv):
        iv.update({
            "s": self.population - (iv["e"] + iv["i"])
        })

    def get_model(self, xs, _, ps, cm):
        # the same order as in self.compartments!
        s, e, i, r = xs.reshape(-1, self.n_age)

        transmission = ps["beta_0"] * np.array(i).dot(cm)
        actual_population = self.population

        vacc = np.array(ps["t_start"] < ts < (ps["t_start"] + ps["T"])).astype(float)
        
        model_eq_dict = {
            "s": -ps["susc"] * (s / actual_population) * transmission,                   # S'(t)
            "e": ps["susc"] * (s / actual_population) * transmission - ps["alpha"] * e,  # E'(t)
            "i": ps["alpha"] * e - ps["gamma"] * i,                                      # I'(t)
            "r": ps["gamma"] * i                                                         # R'(t)
        }

        return self.get_array_from_dict(comp_dict=model_eq_dict)
