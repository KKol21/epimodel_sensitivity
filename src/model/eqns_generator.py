import torch


def get_transition_state_eq(states, param):
    if len(states) < 2:
        return None
    eqns = dict()
    for idx, state in enumerate(states[1:], 1):
        eqns[state] = lambda val: param * (val[idx - 1] - val[idx])
    return [eq for eq in eqns.values()]


class EquationGenerator:
    def __init__(self, ps, actual_population):
        self.ps = ps
        self.actual_population = actual_population
        self.n_age = actual_population.size(0)
        self.initialize_model_eqns()

    def eval_transition_state_eqns(self, eqns, val):
        result = []
        for idx, eq in enumerate(eqns[1:], 1):
            eval_val = eq(val[idx])
            if eval_val.size() != self.n_age:
                eval_val = torch.full((self.n_age, ), eval_val)
            result.append(eval_val)
        return result

    def evaluate_eqns(self, n_state_val, s, r, v_end, i_end, ic_end, transmission, vacc):
        n_state_result = tuple(self.evaluate_e_eqns(n_state_val, s, transmission) + \
                        self.eval_i_eqns(n_state_val) + \
                        self.eval_ic_eqns(i_end, n_state_val) + \
                        self.eval_icr_eqns(n_state_val, ic_end) + \
                        self.eval_v_eqns(n_state_val, s, r, vacc))
        n_state_result = torch.cat(n_state_result)
        return torch.cat((self.s_eq(s, r, v_end, transmission, vacc),
                        self.r_eq(n_state_val, i_end),
                        self.d_eq(ic_end),
                        n_state_result))

    def initialize_model_eqns(self):
        self._get_s_eq()
        self._get_r_eq()
        self._get_d_eq()
        self._get_e_eqns()
        self._get_i_eqns()
        self._get_h_eqns()
        self._get_ic_eqns()
        self._get_icr_eqns()
        self._get_v_eqns()

    @staticmethod
    def get_n_states(n_classes, comp_name):
        return [f"{comp_name}_{i}" for i in range(n_classes)]

    def _get_s_eq(self):
        ps = self.ps
        self.s_eq = lambda s, r, v_end, transmission, vacc: (- ps["susc"] * (s / self.actual_population) * transmission
                                                             - ps["v"] * s / (s + r) * vacc
                                                             + ps["psi"] * v_end)

    def _get_r_eq(self):
        ps = self.ps
        self.r_eq = lambda n_state_val, i_end: ((1 - ps["h"]) * ps["gamma"] * i_end
                                                + ps["gamma_h"] * n_state_val["h"][-1]
                                                + ps["gamma_cr"] * n_state_val["icr"][-1])

    def _get_d_eq(self):
        self.d_eq = lambda ic_end: self.ps["mu"] * self.ps["gamma_c"] * ic_end

    def _get_e_eqns(self):
        e_states = self.get_n_states(self.ps["n_e_states"], "e")
        e_eqns = [self.get_e_0_eq()] + [eq for eq in get_transition_state_eq(e_states, self.ps["alpha"])]
        self.e_eqns = e_eqns

    def get_e_0_eq(self):
        return lambda s, e_0, transmission: (self.ps["susc"] * (s / self.actual_population) * transmission
                                             - self.ps["alpha"] * e_0)

    def evaluate_e_eqns(self, n_state_val, s, transmission):
        val = n_state_val['e']
        e_0 = val[0]
        return [self.e_eqns[0](s, e_0, transmission),
                self.eval_transition_state_eqns(self.e_eqns, val)]

    def _get_i_eqns(self):
        i_states = self.get_n_states(self.ps["n_i_states"], "i")
        i_eqns = [self.get_i_0_eq()] + [eq for eq in get_transition_state_eq(i_states, self.ps['gamma'])]
        self.i_eqns = i_eqns

    def get_i_0_eq(self):
        return lambda e_end, i_0: self.ps["alpha"] * e_end - self.ps["gamma"] * i_0

    def eval_i_eqns(self, n_state_val):
        val = n_state_val['i']
        i_0 = val[0]
        e_end = n_state_val['e'][-1]
        return [self.i_eqns[0](e_end, i_0),
                self.eval_transition_state_eqns(self.i_eqns, val)]

    def _get_h_eqns(self):
        h_states = self.get_n_states(self.ps["n_h_states"], "h")
        h_eqns = [self.get_h_0_eq()] + [eq for eq in get_transition_state_eq(h_states, self.ps['gamma_h'])]
        self.h_eqns = h_eqns

    def get_h_0_eq(self):
        ps = self.ps
        return lambda i_end, h_0: (1 - ps["xi"]) * ps["h"] * ps["gamma"] * i_end - ps["gamma_h"] * h_0

    def eval_h_eqns(self, i_end, n_state_val):
        val = n_state_val['h']
        h_0 = val[0]
        return [self.h_eqns[0](i_end, h_0),
                self.eval_transition_state_eqns(self.h_eqns, val)]

    def _get_ic_eqns(self):
        ic_states = self.get_n_states(self.ps["n_ic_states"], "ic")
        ic_eqns = [self.get_ic_0_eq()] + [eq for eq in get_transition_state_eq(ic_states, self.ps['gamma_c'])]
        self.ic_eqns = ic_eqns

    def get_ic_0_eq(self):
        ps = self.ps
        return lambda i_end, ic_0: ps["xi"] * ps["h"] * ps["gamma"] * i_end - ps["gamma_c"] * ic_0

    def eval_ic_eqns(self, i_end, n_state_val):
        val = n_state_val['ic']
        ic_0 = val[0]
        return [self.ic_eqns[0](i_end, ic_0),
                self.eval_transition_state_eqns(self.ic_eqns, val)]

    def _get_icr_eqns(self):
        ps = self.ps
        icr_states = self.get_n_states(ps["n_icr_states"], "icr")
        icr_eqns = [self.get_icr_0_eq()] + [eq for eq in get_transition_state_eq(icr_states, ps['gamma_cr'])]
        self.icr_eqns = icr_eqns

    def get_icr_0_eq(self):
        ps = self.ps
        return lambda ic_end, icr_0: (1 - ps["mu"]) * ps["gamma_c"] * ic_end - ps["gamma_cr"] * icr_0

    def eval_icr_eqns(self, n_state_val, ic_end):
        val = n_state_val['icr']
        icr_0 = val[0]
        return [self.ic_eqns[0](ic_end, icr_0),
                self.eval_transition_state_eqns(self.icr_eqns, val)]

    def _get_v_eqns(self):
        ps = self.ps
        v_states = self.get_n_states(ps["n_v_states"], "v")
        v_eqns = [self.get_v_0_eq()] + [eq for eq in get_transition_state_eq(v_states, ps['psi'])]
        self.v_eqns = v_eqns

    def get_v_0_eq(self):
        ps = self.ps
        return lambda s, r, vacc, v_0: ps["v"] * s / (s + r) * vacc - v_0 * ps["psi"]

    def eval_v_eqns(self, n_state_val, s, r, vacc):
        val = n_state_val["v"]
        v_0 = val[0]
        return [self.v_eqns[0](s, r, vacc, v_0),
                self.eval_transition_state_eqns(self.v_eqns, val)]
