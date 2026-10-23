import numpy as np

from emsa.sensitivity import SamplerBase
from emsa.model import R0Generator


class SamplerContact(SamplerBase):
    def __init__(self, sim_object, variable_params):
        super().__init__(sim_object, variable_params)
        # Keep sampled contacts supercritical before per-sample beta calibration.
        kappa = self.calculate_kappa()
        max_reduction = 1 - kappa
        if max_reduction <= 0.2:
            raise ValueError("No supercritical contact samples allow at least 20% reduction")

        self.lhs_bounds_dict = {
            "contacts": np.array(
                [
                    np.full(fill_value=0.2, shape=self.sim_object.upper_tri_size),
                    np.full(fill_value=max_reduction, shape=self.sim_object.upper_tri_size),
                ]
            ),
        }

    def run(self):
        lhs_table = self.get_lhs_table()
        return self.get_sim_output(lhs_table=lhs_table)

    def calculate_kappa(self):
        """
        Find the minimum retained fraction of non-home contacts with R0 > 1.

        Returns:
            float: The smallest kappa value satisfying R0 > 1.
        """
        if self.get_r0_from_kappa(0.0) > 1:
            return 0.0
        if self.get_r0_from_kappa(1.0) <= 1:
            raise ValueError("No supercritical contact matrix was found")

        lower, upper = 0.0, 1.0
        for _ in range(24):
            midpoint = (lower + upper) / 2
            if self.get_r0_from_kappa(midpoint) > 1:
                upper = midpoint
            else:
                lower = midpoint
        return upper

    def get_r0_from_kappa(self, kappa: float) -> float:
        """
        Computes R0 for a given kappa.

        Args:
            kappa (float): Retained fraction of non-home contacts.

        Returns:
            float: Resulting R0 value.
        """
        data = self.sim_object.data
        cm_diff = data.cm - data.contact_matrices["home"]
        cm_sim = data.contact_matrices["home"] + kappa * cm_diff
        r0_calculator = R0Generator(data=data, model_struct=self.sim_object.model_struct)

        return data.params["beta"] * r0_calculator.get_eig_val(
            susceptibles=self.sim_object.susceptibles,
            population=self.sim_object.population,
            contact_mtx=cm_sim,
        )
