import numpy


def get_transition_state_eq(states, val, param):
    if len(states) < 2:
        return None
    eq = dict()
    for idx, state in enumerate(states[1:], 1):
        prev_state = val[idx - 1]
        eq[state] = param * (prev_state - val[idx])
    return eq


class EquationGenerator:
    def __init__(self, n_state_val, ps, actual_population, transmission, vacc, s, r):
        self.n_state_val = n_state_val
        self.ps = ps
        self.actual_population = actual_population
        self.transmission = transmission
        self.vacc = vacc
        self.s = s
        self.r = r
        numpy.seterr(all='raise')

    @staticmethod
    def get_n_states(n_classes, comp_name):
        return [f"{comp_name}_{i}" for i in range(n_classes)]

    def get_eqns(self):
        model_eqns_dict = {**self.get_base_eqns(), **self.get_e_eqns(), **self.get_i_eqns(), **self.get_ic_eqns(),
                           **self.get_h_eqns(), **self.get_icr_eqns(), **self.get_v_eqns()}
        return model_eqns_dict

    def get_base_eqns(self):
        ps = self.ps
        n_state_val = self.n_state_val
        s = self.s
        r = self.r
        try:
            eq = {
                "s":
                - ps["susc"] * (s / self.actual_population) * self.transmission
                - ps["v"] * s / (s + r) * self.vacc
                + ps["psi"] * n_state_val["v"][-1],                                    # S'(t)
                "r":
                (1 - ps["h"]) * ps["gamma"] * n_state_val["i"][-1]
                + ps["gamma_h"] * n_state_val["h"][-1]
                + ps["gamma_cr"] * n_state_val["icr"][-1],                             # R'(t)
                "d":
                ps["mu"] * ps["gamma_c"] * n_state_val["ic"][-1]                       # D'(t)
            }
        except:
            print('asd')
        return eq

    def get_e_eqns(self):
        n_state_val = self.n_state_val
        ps = self.ps
        val = n_state_val["e"]
        e_states = self.get_n_states(ps["n_e_states"], "e")
        e_eqns = {"e_0":
                      ps["susc"] * (self.s / self.actual_population) * self.transmission
                      - ps["alpha"] * val[0]}
        e_eqns.update(get_transition_state_eq(e_states, val, ps["alpha"]))
        return e_eqns

    def get_i_eqns(self):
        n_state_val = self.n_state_val
        ps = self.ps
        e_end = n_state_val["e"][-1]
        val = n_state_val["i"]
        i_states = self.get_n_states(ps["n_i_states"], "i")
        i_eqns = {"i_0": ps["alpha"] * e_end - ps["gamma"] * val[0]}
        i_eqns.update(get_transition_state_eq(i_states, val, ps['gamma']))
        return i_eqns

    def get_h_eqns(self):
        n_state_val = self.n_state_val
        ps = self.ps
        i_end = n_state_val["i"][-1]
        val = n_state_val["h"]
        h_states = self.get_n_states(ps["n_h_states"], "h")
        h_eqns = {"h_0": (1 - ps["xi"]) * ps["h"] * ps["gamma"] * i_end - ps["gamma_h"] * val[0]}
        h_eqns.update(get_transition_state_eq(h_states, val, ps['gamma_h']))
        return h_eqns

    def get_ic_eqns(self):
        n_state_val = self.n_state_val
        ps = self.ps
        i_end = n_state_val["i"][-1]
        val = n_state_val["ic"]
        ic_states = self.get_n_states(ps["n_ic_states"], "ic")
        ic_eqns = {"ic_0": ps["xi"] * ps["h"] * ps["gamma"] * i_end - ps["gamma_c"] * val[0]}
        ic_eqns.update(get_transition_state_eq(ic_states, val, ps['gamma_c']))
        return ic_eqns

    def get_icr_eqns(self):
        n_state_val = self.n_state_val
        ps = self.ps
        ic_end = n_state_val["ic"][-1]
        val = n_state_val["icr"]
        icr_states = self.get_n_states(ps["n_icr_states"], "icr")
        icr_eqns = {"icr_0": (1 - ps["mu"]) * ps["gamma_c"] * ic_end - ps["gamma_cr"] * val[0]}
        icr_eqns.update(get_transition_state_eq(icr_states, val, ps['gamma_cr']))
        return icr_eqns

    def get_v_eqns(self):
        val = self.n_state_val["v"]
        ps = self.ps
        v_states = self.get_n_states(ps["n_v_states"], "v")
        v_eqns = {'v_0': ps["v"] * self.s / (self.s + self.r) * self.vacc - val[0] * ps["psi"]}
        v_eqns.update(get_transition_state_eq(v_states, val, ps['psi']))
        return v_eqns
