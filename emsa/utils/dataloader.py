from abc import ABC
import torch
from os.path import dirname, realpath

PROJECT_PATH = dirname(dirname(dirname(realpath(__file__))))


class DataLoaderBase(ABC):
    def __init__(self):
        self.project_path = PROJECT_PATH
        self.age_data = None
        self.cm = None
        self.params = None
        self.device = None

    @property
    def n_age(self):
        return self.age_data.size(0)

    def transform_matrix(self, matrix: torch.Tensor) -> torch.Tensor:
        """
        Perform symmetrization of contact matrix, by averaging its elements with the elements of
        it's transpose, then dividing each row with the size of the age group corresponding to the given row.

        After this, the C[i, j] element represents the average number of interactions a member of age
        group i has with members of age group j, while C[j, i] represents the average number of
        interactions a member of age group j has with members of age group j in a day.

        For example, in the case that the population of age group i is the double of age group j:

                                     C[j, i] = 2 * C[i, j]

        Args:
            matrix (torch.Tensor): The contact matrix.

        Returns:
            torch.Tensor: The transformed contact matrix.
        """
        # Age vector as a column vector
        age_distribution = self.age_data.reshape((-1, 1))
        # Matrix containing total number of contacts
        total_contact = matrix * age_distribution
        # Symmetrize matrix
        output = (total_contact + total_contact.T) / 2
        # Divide by age group sizes
        output /= age_distribution
        return output
