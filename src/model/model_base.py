from abc import ABC

import torch
from src.dataloader import DataLoader


class EpidemicModelBase(ABC):
    def __init__(self, model_data: DataLoader, compartments: list):
        """
        Initialises Abstract base class for epidemic models.

        This class provides the base functionality for epidemic models. It contains methods to initialize the model,
        retrieve initial values, and obtain the model solution.

        Args:
            model_data (DataLoader): Model data object containing model parameters and device information.
            compartments (list): List of compartment names.

        Returns:
            None
        """
        self.ps = model_data.model_parameters_data
        self.population = model_data.age_data.flatten()
        self.compartments = compartments
        self.n_comp = len(compartments)
        self.c_idx = {comp: idx for idx, comp in enumerate(compartments)}
        self.n_age = self.population.shape[0]
        self.device = model_data.device
        self.size = self.n_age * self.n_comp

    def get_initial_values(self):
        """
        Retrieves the initial values for the model.

        This method retrieves the initial values for the model. It sets the initial value for the exposed (e_0)
        compartment to 1 and subtracts 1 from the susceptible (s) compartment for the appropriate age group.

        Returns:
            torch.Tensor: Initial values of the model.
        """
        iv = torch.zeros(self.size)
        age_group = 3 * self.n_comp
        iv[age_group + self.c_idx['e_0']] = 1
        iv[self.idx('s')] = self.population
        iv[age_group + self.c_idx['s']] -= 1
        return iv

    def idx(self, state: str) -> bool:
        return torch.arange(self.size) % self.n_comp == self.c_idx[state]

    def _aggregate_by_age_n_state(self, solution, comp):
        """
        This method aggregates the solution by age for a compartment with substates by summing the solution
        values of individual states within the compartment.

        Args:
            solution (torch.Tensor): Model solution tensor.
            comp (str): Compartment name.

        Returns:
            torch.Tensor: Aggregated solution by age.
        """
        result = 0
        for state in get_n_states(self.ps[f'n_{comp}'], comp):
            result += solution[:, self.idx(state)].sum(axis=1)
        return result


def get_n_states(n_classes, comp_name):
    """
   Returns a list of compartment names based on the number of states and class name.

   This function returns a list of compartment names based on the number of states specified and the class name.
   It is used to generate the compartments for the model.

   Args:
       n_classes (int): Number of classes for the compartment.
       comp_name (str): Compartment name.

   Returns:
       list: List of state names.
   """
    return [f"{comp_name}_{i}" for i in range(n_classes)]
