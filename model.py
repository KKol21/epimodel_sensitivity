from model_base import EpidemicModelBase

import numpy as np


class VaccinatedModel(EpidemicModelBase):
    def __init__(self, model_data):
        self.n_vac_states = model_data.model_parameters_data["n_vac_states"]
        compartments = ["s", "e", "i", "r"] + self.get_vac_compartments(self.n_vac_states)
        super().__init__(model_data=model_data, compartments=compartments)

    @staticmethod
    def get_vac_compartments(n_classes):
        return [f"v_{i}" for i in range(n_classes)]

    @staticmethod
    def get_vacc_bool(ts, ps):
        return np.array(ps["t_start"] < ts < (ps["t_start"] + ps["T"])).astype(float)

    def update_initial_values(self, iv):
        iv.update({
            "s": self.population - (iv["e"] + iv["i"])
        })

    def get_model(self, xs, ts, ps, cm):
        # the same order as in self.compartments!
        vac_state_val = dict()
        vac_comp = self.get_vac_compartments(self.n_vac_states)
        val = xs.reshape(-1, self.n_age)
        for idx, comp in enumerate(vac_comp, 4):
            vac_state_val[comp] = val[idx]
        s, e, i, r = val[:4]

        transmission = ps["beta_0"] * np.array(i).dot(cm)
        actual_population = self.population
        vacc = self.get_vacc_bool(ts, ps)


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

        for idx, state in enumerate(vac_comp[0:], 1):
            prev_state = vac_state_val[f"v_{idx-1}"]
            vac_eq_dict[state] = (prev_state - vac_state_val[state]) * ps["psi"]  # V_i'(t)

        model_eq_dict.update(vac_eq_dict)
        return self.get_array_from_dict(comp_dict=model_eq_dict)
