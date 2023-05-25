from abc import ABC, abstractmethod

import torch
from torchdiffeq import odeint
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

    def get_solution(self, t, cm, daily_vac):
        """
        Retrieves the model solution for the given parameters.

        This method retrieves the solution for the model given the time points (t), contact matrix (cm),
        and daily vaccine distribution (daily_vac).

        Args:
            t (torch.Tensor): Time points.
            cm (torch.Tensor): Contact matrix.
            daily_vac (torch.Tensor): Daily vaccine distribution.

        Returns:
            torch.Tensor: Model solution.
        """
        initial_values = self.get_initial_values()
        return odeint(func=self.get_model(cm, daily_vac), y0=initial_values, t=t, method='euler')

    @abstractmethod
    def get_model(self, cm, daily_vac):
        pass

    def idx(self, state: str) -> bool:
        return torch.arange(self.size) % self.n_comp == self.c_idx[state]


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
