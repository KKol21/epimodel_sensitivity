import torch
from src.model.model import VaccinatedModel2, get_n_states


class MatrixGenerator:
    def __init__(self, model: VaccinatedModel2, cm, ps):
        self.cm = cm
        self.ps = ps
        self.s_mtx = model.n_age * model.n_comp
        self.n_state_comp = model.n_state_comp
        self.n_age = model.n_age
        self.n_comp = model.n_comp
        self.population = model.population
        self.device = model.device
        self.idx = model.idx
        self.c_idx = model.c_idx
        self._get_trans_param_dict()

    def get_A(self):
        # Multiplied with y, the resulting 1D tensor contains the rate of transmission for the susceptibles of
        # age group i at the indices of compartments s^i and e_0^i
        A = torch.zeros((self.s_mtx, self.s_mtx)).to(self.device)
        transmission_rate = self.ps["beta"] * self.ps["susc"] / self.population
        idx = self.idx

        A[idx('s'), idx('s')] = - transmission_rate
        A[idx('e_0'), idx('s')] = transmission_rate
        return A

    def get_T(self):
        T = torch.zeros((self.s_mtx, self.s_mtx)).to(self.device)
        # Multiplied with y, the resulting 1D tensor contains the sum of all contacts with infecteds of
        # age group i at indices of compartments s_i and e_i^0
        for i_state in get_n_states(self.ps["n_i"], "i"):
            T[self._get_comp_slice('s'), self._get_comp_slice(i_state)] = self.cm.T
            T[self._get_comp_slice('e_0'), self._get_comp_slice(i_state)] = self.cm.T
        return T

    def get_B(self):
        from src.model.r0 import generate_transition_matrix
        ps = self.ps
        # B is the tensor representing the first order elements of the ODE system. We begin with filling in
        # the transition blocks of the erlang distributed parameters
        B = generate_transition_matrix(self.trans_param_dict, self.ps, self.n_age, self.n_comp, self.c_idx)

        # Then do the rest of the first order terms
        idx = self.idx
        c_end = self._get_end_state
        e_end = c_end('e')
        i_end = c_end('i')
        h_end = c_end('h')
        ic_end = c_end('ic')
        icr_end = c_end('icr')
        v_end = c_end('v')

        # E   ->  I
        B[idx('i_0'), idx(e_end)] = ps["alpha"]
        # I   ->  H
        B[idx('h_0'), idx(i_end)] = (1 - ps["xi"]) * ps["h"] * ps["gamma"]
        # H   ->  R
        B[idx('r'), idx(h_end)] = ps['gamma_h']
        # I   ->  IC
        B[idx('ic_0'), idx(i_end)] = ps["xi"] * ps["h"] * ps["gamma"]
        # IC  ->  ICR
        B[idx('icr_0'), idx(ic_end)] = ps["gamma_c"] * (1 - ps["mu"])
        # ICR ->  R
        B[idx('r'), idx(icr_end)] = ps["gamma_cr"]
        # IC  ->  D
        B[idx('d'), idx(ic_end)] = ps["gamma_c"] * ps["mu"]
        # I   ->  R
        B[idx('r'), idx(i_end)] = (1 - ps['h']) * ps['gamma']
        # V   ->  S
        B[idx("s"), idx(v_end)] = ps["psi"]
        return B

    def get_V_1(self):
        V_1 = torch.zeros((self.s_mtx, self.s_mtx)).to(self.device)
        V_1[self.idx('s'), self.idx('s')] = self.ps["v"]
        V_1[self.idx('v_0'), self.idx('s')] = self.ps["v"]
        return V_1

    def get_V_2(self):
        idx = self.idx
        V_2 = torch.zeros((self.s_mtx, self.s_mtx)).to(self.device)
        V_2[~ (idx('s') + idx('v_0')), 0] = 1
        V_2[idx('s'), idx('s')] = - 1
        V_2[idx('s'), idx('r')] = - 1
        V_2[idx('v_0'), idx('s')] = 1
        V_2[idx('v_0'), idx('r')] = 1
        return V_2

    def _get_comp_slice(self, comp: str) -> slice:
        return slice(self.c_idx[comp], self.s_mtx, self.n_comp)

    def _get_end_state(self, comp: str) -> str:
        n_states = self.ps[f'n_{comp}']
        return f'{comp}_{n_states - 1}'

    def _get_trans_param_dict(self):
        ps = self.ps
        trans_param_list = [ps["alpha"], ps["gamma"], ps["gamma_h"], ps["gamma_c"], ps["gamma_cr"], ps["psi"]]
        self.trans_param_dict = {key: value for key, value in zip(self.n_state_comp, trans_param_list)}
