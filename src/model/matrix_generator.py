from typing import List

import torch

from src.model.model_base import get_substates, EpidemicModelBase


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
    trans_block[:n_states - 1, 1:] = trans_block[:n_states - 1, 1:].fill_diagonal_(transition_param)
    return trans_block


def get_trans_param(state: str, trans_data: dict):
    for data in trans_data.values():
        if data["source"] == state:
            return data["param"]
    raise Exception(f"No transition parameter was provided for state {state}")


def generate_transition_matrix(states_dict: dict, trans_data: dict, parameters: dict,
                               n_age: int, n_comp: int, c_idx: dict) -> torch.Tensor:
    """
    Generate the transition matrix for the model.

    Args:
        states_dict:
        trans_data:
        parameters: A dictionary containing model parameters.
        n_age: The number of age groups.
        n_comp: The number of compartments.
        c_idx: A dictionary containing the indices of different compartments.

    """
    trans_matrix = torch.zeros((n_age * n_comp, n_age * n_comp))
    for age_group in range(n_age):
        for state, data in states_dict.items():
            n_states = data["n_substates"]
            trans_param = parameters[get_trans_param(state, trans_data)]
            diag_idx = age_group * n_comp + c_idx[f'{state}_0']
            block_slice = slice(diag_idx, diag_idx + n_states)
            # Fill in transition block of each transitional state
            trans_matrix[block_slice, block_slice] = generate_transition_block(trans_param, n_states)
    return trans_matrix


def get_infectious_states(state_data):
    infectious_states = []
    for state, data in state_data.items():
        if data["type"] == "infectious":
            infectious_states += get_substates(data["n_substates"], state)
    return infectious_states


class MatrixGenerator:
    """
    Class responsible for generating the matrices used in the model.

    Let y be size n_samples * s_mtx, each row corresponding to a different simulation. The general formula we represent
    the system of ODEs with is the following:

            y' = (y @ T_1) * (y @ T_2) + y * L,

    where T_1 and T_2 are responsible for the transmission of the disease, and L defines the linear changes.

    In addition to this, continuous vaccination can be written as

            vacc = (y @ V_1) / (y @ V_2).

    Args:
        model (EpidemicModelBase): An instance of the EpidemicModelBase class.
        cm: The contact matrix.
        ps: A dictionary containing model parameters.

    Attributes:
        cm: The contact matrix.
        ps: A dictionary containing model parameters.
        s_mtx: The total number of compartments in the model.
        n_age: The number of age groups.
        n_comp: The number of states.
        population: The total population.
        device: The device to be used for computations.
        idx: A dictionary containing the indices of different compartments.
        c_idx: A dictionary containing the indices of different compartments' components.

    Methods:
        _get_comp_slice(comp): Get a slice representing the indices of a given compartment.
        _get_end_state(comp): Get the string representing the last state of a given compartment.
        _get_trans_param_dict(): Get a dictionary of transition parameters for different compartments.
    """

    def __init__(self, model: EpidemicModelBase, cm):
        """
        Initialize the MatrixGenerator instance.

        Args:
            model (EpidemicModelBase): An instance of the EpidemicModelBase class.
            cm: The contact matrix.

        """
        self.cm = cm
        self.ps = model.ps
        self.data = model.data
        self.state_data = model.data.state_data
        self.trans_data = model.data.trans_data
        self.s_mtx = model.s_mtx
        self.n_age = model.n_age
        self.n_comp = model.n_comp
        self.population = model.population
        self.device = model.device
        self.idx = model.idx
        self.c_idx = model.c_idx
        self.inf_inflow_state = [f"{trans['target']}_0" for trans in self.trans_data.values()
                                 if trans["type"] == "infection"][0]
        self.infectious_states = get_infectious_states(self.state_data)

    def get_A(self) -> torch.Tensor:
        """
        Returns:
            Torch.Tensor: When multiplied with y, the resulting tensor contains the rate of transmission for
            the susceptibles of age group i at the indices of compartments s^i and e_0^i
        """
        ps = self.ps
        A = torch.zeros((self.s_mtx, self.s_mtx)).to(self.device)
        transmission_rate = ps["beta"] * ps["susc"] / self.population
        idx = self.idx

        A[idx('s_0'), idx('s_0')] = - transmission_rate
        A[idx('s_0'), idx(self.inf_inflow_state)] = transmission_rate
        return A

    def get_T(self, cm=None) -> torch.Tensor:
        T = torch.zeros((self.s_mtx, self.s_mtx)).to(self.device)
        if cm is None:
            cm = self.cm
        # Multiplied with y, the resulting 1D tensor contains the sum of all contacts with infecteds of
        # age group i at indices of compartments s_i and e_i^0
        for i_state in self.infectious_states:
            T[self._get_comp_slice(i_state), self._get_comp_slice('s_0')] = cm.T
            T[self._get_comp_slice(i_state), self._get_comp_slice(self.inf_inflow_state)] = cm.T
        return T

    def get_B(self) -> torch.Tensor:
        ps = self.ps
        state_data = self.state_data
        trans_data = self.trans_data
        # B is the tensor representing the first-order elements of the ODE system. We begin by
        # filling in the transition blocks of the intermediate states
        intermediate_states = {state: data for state, data in state_data.items()
                               if data["type"] not in ["susceptible",
                                                       "recovered",
                                                       "dead"]}
        B = generate_transition_matrix(states_dict=intermediate_states, trans_data=trans_data, parameters=ps,
                                       n_age=self.n_age, n_comp=self.n_comp, c_idx=self.c_idx).to(self.device)

        # Fill in the rest of the first-order terms
        idx = self.idx
        end_state = {state: f"{state}_{data['n_substates'] - 1}" for state, data in state_data.items()}
        for trans in trans_data.values():
            # Iterate over the linear transitions
            if trans["type"] == "basic":
                source = end_state[trans["source"]]
                target = f"{trans['target']}_0"
                trans_param = ps[trans["param"]]
                distr = trans["distr"]
                if distr is not None:
                    # Multiply the transition parameter by the distribution(s) given
                    trans_param *= self.mul_distr(distr)
                B[idx(source), idx(target)] = trans_param
        return B

    def mul_distr(self, distr: List[str]):
        result = 1
        for distr_param in distr:
            if distr_param[-1] == "_":
                # Multiply by 1 - distribution
                result *= 1 - self.ps[distr_param[:-1]]
            else:
                # Multiply by distribution
                result *= self.ps[distr_param]
        return result

    def get_V_1(self, daily_vac=None) -> torch.Tensor:
        if daily_vac is None:
            daily_vac = self.ps["daily_vac"]
        V_1 = torch.zeros((self.s_mtx, self.s_mtx)).to(self.device)
        # Tensor responsible for the nominators of the vaccination formula
        V_1[self.idx('s_0'), self.idx('s_0')] = daily_vac
        V_1[self.idx('s_0'), self.idx('v_0')] = daily_vac
        return V_1

    def get_V_2(self) -> torch.Tensor:
        idx = self.idx
        V_2 = torch.zeros((self.s_mtx, self.s_mtx)).to(self.device)
        # Fill in all the terms such that we will divide the terms at the indices of s^i and v^i by (s^i + r^i)
        V_2[idx('s_0'), idx('s_0')] = -1
        V_2[idx('r_0'), idx('s_0')] = -1
        V_2[idx('s_0'), idx('v_0')] = 1
        V_2[idx('r_0'), idx('v_0')] = 1
        return V_2

    def _get_comp_slice(self, comp: str) -> slice:
        """

        Args:
            comp (str): The compartment name.

        Returns:
            slice: A slice representing the indices of the compartment.

        """
        return slice(self.c_idx[comp], self.s_mtx, self.n_comp)

    def get_end_state(self, comp: str) -> str:
        return f"{comp}_{self.state_data[comp]['n_substates'] - 1}"
