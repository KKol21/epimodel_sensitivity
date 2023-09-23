from abc import ABC, abstractmethod

import torch
from src.dataloader import DataLoader


class EpidemicModelBase(ABC):
    def __init__(self, model_data: DataLoader):
        """
        Initialises Abstract base class for epidemic models.

        This class provides the base functionality for epidemic models. It contains methods to initialize the model,
        retrieve initial values, and obtain the model solution.

        Args:
            model_data (DataLoader): Model data object containing model parameters and device information.

        Returns:
            None
        """
        self.data = model_data
        self.ps = model_data.model_parameters
        self.population = model_data.age_data.flatten()
        self.compartments = self.get_compartments()
        self.n_comp = len(self.compartments)
        self.c_idx = {comp: idx for idx, comp in enumerate(self.compartments)}
        self.n_age = self.population.shape[0]
        self.s_mtx = self.n_age * self.n_comp
        self.device = model_data.device
        self.size = self.n_age * self.n_comp

    @abstractmethod
    def _get_constant_matrices(self):
        pass

    def get_compartments(self):
        compartments = []
        for name, data in self.data.state_data.items():
            compartments += get_substates(data["n_substates"], name)
        return compartments

    def get_initial_values(self):
        """
        This method retrieves the initial values for the model. It sets the initial value for the infected (e_0^3)
        compartment of the 3rd age group to 1 and subtracts 1 from the susceptible (s) compartment for the appropriate
        age group.

        Returns:
            torch.Tensor: Initial values of the model.
        """
        iv = torch.zeros(self.size).to(self.device)
        age_group = 3 * self.n_comp
        iv[age_group + self.c_idx['i_0']] = 1
        iv[self.idx('s_0')] = self.population
        iv[age_group + self.c_idx['s_0']] -= 1
        return iv

    def idx(self, state: str) -> bool:
        return torch.arange(self.size) % self.n_comp == self.c_idx[state]

    def aggregate_by_age(self, solution, comp):
        """
        This method aggregates the solution by age for a compartment by summing the solution
        values of individual substates.

        Args:
            solution (torch.Tensor): Model solution tensor.
            comp (str): Compartment name.

        Returns:
            torch.Tensor: Aggregated solution by age.
        """
        result = 0
        for state in get_substates(self.data.state_data[comp]["n_substates"], comp):
            result += solution[:, self.idx(state)].sum(axis=1)
        return result


def get_substates(n_substates, comp_name):
    """
   Returns a list of compartment names based on the number of states and class name.

   This function returns a list of compartment names based on the number of states specified and the class name.
   It is used to generate the compartments for the model.

   Args:
       n_substates (int): Number of classes for the compartment.
       comp_name (str): Compartment name.

   Returns:
       list: List of state names.
   """
    return [f"{comp_name}_{i}" for i in range(n_substates)]
