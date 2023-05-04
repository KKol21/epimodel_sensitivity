def get_transition_state_eq(idx, param):
    return lambda val: param * (val[idx - 1] - val[idx])


class EquationGenerator:
    def __init__(self, ps, actual_population):
        self.ps = ps
        self.actual_population = actual_population
        self.n_age = actual_population.size(0)
        self.initialize_model_eqns()

    @staticmethod
    def eval_transition_state_eqns(eqns, val):
        result = []
        for idx, eq in enumerate(eqns[1:], 1):
            result.append(eq(val))
        return result

    @staticmethod
    def get_transition_state_eqns(n_states, param):
        if n_states < 2:
            return []
        eqns = []
        for idx in range(1, n_states):
            eqns.append(get_transition_state_eq(idx, param))
        return eqns

    def evaluate_eqns(self, n_state_val, s, r, transmission, vacc):
        i_end = n_state_val['i'][-1]
        ic_end = n_state_val['ic'][-1]
        v_end = n_state_val['v'][-1]

        n_state_result = self.get_n_state_result(i_end, ic_end, n_state_val, s, r, transmission, vacc)
        return [self.s_eq(s, r, v_end, transmission, vacc)] + n_state_result \
               + [self.r_eq(n_state_val, i_end), self.d_eq(ic_end)]

    def get_n_state_result(self, i_end, ic_end, n_state_val, s, r, transmission, vacc):
        n_state_result = self.eval_e_eqns(n_state_val, s, transmission) + \
                         self.eval_i_eqns(n_state_val) + \
                         self.eval_h_eqns(n_state_val, i_end) + \
                         self.eval_ic_eqns(n_state_val, i_end) + \
                         self.eval_icr_eqns(n_state_val, ic_end) + \
                         self.eval_v_eqns(n_state_val, s, r, vacc)
        return n_state_result

    def eval_e_eqns(self, n_state_val, s, transmission):
        val = n_state_val['e']
        e_0 = val[0]
        return [self.e_eqns[0](s, e_0, transmission)] + self.eval_transition_state_eqns(self.e_eqns, val)

    def eval_i_eqns(self, n_state_val):
        val = n_state_val['i']
        i_0 = val[0]
        e_end = n_state_val['e'][-1]
        return [self.i_eqns[0](e_end, i_0)] + self.eval_transition_state_eqns(self.i_eqns, val)

    def eval_h_eqns(self, n_state_val, i_end):
        val = n_state_val['h']
        h_0 = val[0]
        return [self.h_eqns[0](i_end, h_0)] + self.eval_transition_state_eqns(self.h_eqns, val)

    def eval_ic_eqns(self, n_state_val, i_end):
        val = n_state_val['ic']
        ic_0 = val[0]
        return [self.ic_eqns[0](i_end, ic_0)] + self.eval_transition_state_eqns(self.ic_eqns, val)

    def eval_icr_eqns(self, n_state_val, ic_end):
        val = n_state_val['icr']
        icr_0 = val[0]
        return [self.icr_eqns[0](ic_end, icr_0)] + self.eval_transition_state_eqns(self.icr_eqns, val)

    def eval_v_eqns(self, n_state_val, s, r, vacc):
        val = n_state_val["v"]
        v_0 = val[0]
        return [self.v_eqns[0](s, r, vacc, v_0)] + self.eval_transition_state_eqns(self.v_eqns, val)

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
        self.e_eqns = [self.get_e_0_eq()] \
                      + self.get_transition_state_eqns(self.ps["n_e"], self.ps["alpha"])

    def get_e_0_eq(self):
        return lambda s, e_0, transmission: (self.ps["susc"] * (s / self.actual_population) * transmission
                                             - self.ps["alpha"] * e_0)

    def _get_i_eqns(self):
        self.i_eqns = [self.get_i_0_eq()] \
                      + self.get_transition_state_eqns(self.ps["n_i"], self.ps['gamma'])

    def get_i_0_eq(self):
        return lambda e_end, i_0: self.ps["alpha"] * e_end - self.ps["gamma"] * i_0

    def _get_h_eqns(self):
        self.h_eqns = [self.get_h_0_eq()] \
                      + self.get_transition_state_eqns(self.ps["n_h"], self.ps['gamma_h'])

    def get_h_0_eq(self):
        ps = self.ps
        return lambda i_end, h_0: (1 - ps["xi"]) * ps["h"] * ps["gamma"] * i_end - ps["gamma_h"] * h_0

    def _get_ic_eqns(self):
        self.ic_eqns = [self.get_ic_0_eq()] \
                       + self.get_transition_state_eqns(self.ps["n_ic"], self.ps['gamma_c'])

    def get_ic_0_eq(self):
        ps = self.ps
        return lambda i_end, ic_0: ps["xi"] * ps["h"] * ps["gamma"] * i_end - ps["gamma_c"] * ic_0

    def _get_icr_eqns(self):
        self.icr_eqns = [self.get_icr_0_eq()] \
                        + self.get_transition_state_eqns(self.ps["n_icr"], self.ps['gamma_cr'])

    def get_icr_0_eq(self):
        ps = self.ps
        return lambda ic_end, icr_0: (1 - ps["mu"]) * ps["gamma_c"] * ic_end - ps["gamma_cr"] * icr_0

    def _get_v_eqns(self):
        self.v_eqns = [self.get_v_0_eq()] \
                      + self.get_transition_state_eqns(self.ps["n_v"], self.ps['psi'])

    def get_v_0_eq(self):
        ps = self.ps
        return lambda s, r, vacc, v_0: ps["v"] * s / (s + r) * vacc - v_0 * ps["psi"]
