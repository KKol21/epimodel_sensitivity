import numpy as np

from emsa.sensitivity import SamplerBase
from emsa.model import R0Generator


class SamplerContact(SamplerBase):
    def __init__(self, sim_object, variable_params):
        super().__init__(sim_object, variable_params)
        # Multiplier for ensuring positive R0 assumption of analysis
        kappa = self.calculate_kappa()

        self.lhs_bounds_dict = {
            "contacts": np.array(
                [
                    np.full(fill_value=0.2, shape=self.sim_object.upper_tri_size),
                    (1 - kappa) * np.full(fill_value=1, shape=self.sim_object.upper_tri_size),
                ]
            ),
        }

    def run(self):
        lhs_table = self.get_lhs_table()
        self.get_sim_output(lhs_table=lhs_table)

    def calculate_kappa(self):
        """
        Finds the minimum kappa where R0 > 1.

        Returns:
            float: The smallest kappa value satisfying R0 > 1.
        """
        kappas = np.linspace(0, 1, 1000)
        r0_values = np.array([self.get_r0_from_kappa(k) for k in kappas])

        # Find the first index where r0_values > 1
        k = np.searchsorted(r0_values, 1, side="right")

        if k == len(kappas):  # If no valid kappa found
            raise Exception("No valid kappa was found!")

        return kappas[k]

    def get_r0_from_kappa(self, kappa: float) -> float:
        """
        Computes R0 for a given kappa.

        Args:
            kappa (float): Scaling factor for contact matrix adjustment.

        Returns:
            float: Resulting R0 value.
        """
        data = self.sim_object.data
        cm_diff = data.cm - data.contact_matrices["home"]
        cm_sim = data.contact_matrices["home"] + (1 - kappa) * cm_diff
        r0_calculator = R0Generator(data=data, model_struct=self.sim_object.model_struct)

        return r0_calculator.get_eig_val(
            susceptibles=self.sim_object.susceptibles,
            population=self.sim_object.population,
            contact_mtx=cm_sim,
        )
