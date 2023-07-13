import torch
from src.model.model import VaccinatedModel
from src.model.model_base import get_n_states


def generate_transition_block(transition_param: float, n_states: int) -> torch.Tensor:
    """
    Generate a transition block for the transition matrix.

    Args:
        transition_param: The transition parameter value.
        n_states: The number of states in the block.

    Returns:
        torch.Tensor: The transition block.

    """
    trans_block = torch.zeros((n_states, n_states))
    # Outflow from states (diagonal elements)
    trans_block = trans_block.fill_diagonal_(-transition_param)
    # Inflow to states (elements under the diagonal)
    trans_block[1:, :n_states - 1] = trans_block[1:, :n_states - 1].fill_diagonal_(transition_param)
    return trans_block


def generate_transition_matrix(trans_param_dict, ps, n_age, n_comp, c_idx):
    """
    Generate the transition matrix for the model.

    Args:
        trans_param_dict: A dictionary containing the transition parameters for different compartments.
        ps: A dictionary containing model parameters.
        n_age: The number of age groups.
        n_comp: The number of compartments.
        c_idx: A dictionary containing the indices of different compartments.

    Returns:
        torch.Tensor: The transition matrix.

    """
    trans_matrix = torch.zeros((n_age * n_comp, n_age * n_comp))
    for age_group in range(n_age):
        for comp, trans_param in trans_param_dict.items():
            n_states = ps[f'n_{comp}']
            diag_idx = age_group * n_comp + c_idx[f'{comp}_0']
            block_slice = slice(diag_idx, diag_idx + n_states)
            # Fill in transition block of each transitional state
            trans_matrix[block_slice, block_slice] = generate_transition_block(trans_param, n_states)
    return trans_matrix


class MatrixGenerator:
    """
    Class responsible for generating the matrices used in the model.

    Args:
        model (VaccinatedModel): An instance of the VaccinatedModel class.
        cm: The contact matrix.
        ps: A dictionary containing model parameters.

    Attributes:
        cm: The contact matrix.
        ps: A dictionary containing model parameters.
        s_mtx: The total number of compartments in the model.
        n_state_comp: The classes with substates in the model.
        n_age: The number of age groups.
        n_comp: The number of states.
        population: The total population.
        device: The device to be used for computations.
        idx: A dictionary containing the indices of different compartments.
        c_idx: A dictionary containing the indices of different compartments' components.

    Methods:
        get_A(): Generate the matrix A.
        get_T(): Generate the matrix T.
        get_B(): Generate the matrix B.
        get_V_1(daily_vac): Generate the matrix V_1.
        get_V_2(): Generate the matrix V_2.
        _get_comp_slice(comp): Get a slice representing the indices of a given compartment.
        _get_end_state(comp): Get the string representing the last state of a given compartment.
        _get_trans_param_dict(): Get a dictionary of transition parameters for different compartments.
    """

    def __init__(self, model: VaccinatedModel, cm, ps):
        """
        Initialize the MatrixGenerator instance.

        Args:
            model (VaccinatedModel): An instance of the VaccinatedModel class.
            cm: The contact matrix.
            ps: A dictionary containing model parameters.

        """
        self.cm = cm
        self.ps = ps
        self.s_mtx = model.s_mtx
        self.n_state_comp = model.n_state_comp
        self.n_age = model.n_age
        self.n_comp = model.n_comp
        self.population = model.population
        self.device = model.device
        self.idx = model.idx
        self.c_idx = model.c_idx

    def get_A(self) -> torch.Tensor:
        # Multiplied with y, the resulting 1D tensor contains the rate of transmission for the susceptibles of
        # age group i at the indices of compartments s^i and e_0^i
        A = torch.zeros((self.s_mtx, self.s_mtx)).to(self.device)
        transmission_rate = self.ps["beta"] * self.ps["susc"] / self.population
        idx = self.idx

        A[idx('s'), idx('s')] = - transmission_rate
        A[idx('e_0'), idx('s')] = transmission_rate
        return A

    def get_T(self) -> torch.Tensor:
        T = torch.zeros((self.s_mtx, self.s_mtx)).to(self.device)
        # Multiplied with y, the resulting 1D tensor contains the sum of all contacts with infecteds of
        # age group i at indices of compartments s_i and e_i^0
        for i_state in get_n_states(self.ps["n_i"], "i"):
            T[self._get_comp_slice('s'), self._get_comp_slice(i_state)] = self.cm.T
            T[self._get_comp_slice('e_0'), self._get_comp_slice(i_state)] = self.cm.T
        return T

    def get_B(self) -> torch.Tensor:
        ps = self.ps
        # B is the tensor representing the first order elements of the ODE system. We begin by
        # filling in the transition blocks of the erlang distributed parameters
        B = generate_transition_matrix(self._get_trans_param_dict(), self.ps, self.n_age, self.n_comp, self.c_idx)

        # Then fill in the rest of the first order terms
        idx = self.idx
        c_end = self._get_end_state
        e_end = c_end('e')
        i_end = c_end('i')
        h_end = c_end('h')
        ic_end = c_end('ic')
        icr_end = c_end('icr')

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
        return B

    def get_V_1(self, daily_vac) -> torch.Tensor:
        """
        Generate the matrix V_1.

        Args:
            daily_vac: The number of daily vaccinations.

        Returns:
            torch.Tensor: The matrix V_1.

        """
        V_1 = torch.zeros((self.s_mtx, self.s_mtx)).to(self.device)
        # Tensor responsible for the nominators of the vaccination formula
        V_1[self.idx('s'), self.idx('s')] = daily_vac
        V_1[self.idx('v_0'), self.idx('s')] = daily_vac
        return V_1

    def get_V_2(self) -> torch.Tensor:
        idx = self.idx
        V_2 = torch.zeros((self.s_mtx, self.s_mtx)).to(self.device)
        # Make sure to avoid division by zero
        V_2[~ (idx('s') + idx('v_0')), 0] = 1
        # Fill in all the terms such that we will divide the terms at the indices of s^i and v^i by (s^i + r^i)
        V_2[idx('s'), idx('s')] = - 1
        V_2[idx('s'), idx('r')] = - 1
        V_2[idx('v_0'), idx('s')] = 1
        V_2[idx('v_0'), idx('r')] = 1
        return V_2

    def _get_comp_slice(self, comp: str) -> slice:
        """
        Get a slice representing the indices of a given compartment.

        Args:
            comp (str): The compartment name.

        Returns:
            slice: A slice representing the indices of the compartment.

        """
        return slice(self.c_idx[comp], self.s_mtx, self.n_comp)

    def _get_end_state(self, comp: str) -> str:
        """
        Get the string representing the last state of a given compartment.

        Args:
            comp (str): The compartment name.

        Returns:
            str: The string representing the last state of the compartment.

        """
        n_states = self.ps[f'n_{comp}']
        return f'{comp}_{n_states - 1}'

    def _get_trans_param_dict(self):
        """
        Get a dictionary of transition parameters for classes with substates.

        Returns:
            dict: A dictionary of transition parameters.

        """
        ps = self.ps
        trans_param_list = [ps["alpha"], ps["gamma"], ps["gamma_h"], ps["gamma_c"], ps["gamma_cr"]]
        return {state: trans_param for state, trans_param in zip(self.n_state_comp, trans_param_list)}
