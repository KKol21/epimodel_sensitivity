import numpy as np

from src.sensitivity.sampler_base import SamplerBase


class SamplerContact(SamplerBase):
    def __init__(self, sim_state: dict, sim_obj, n_samples):
        super().__init__(sim_state, sim_obj)
        self.mode = "contact"
        self.n_samples = n_samples
        self.sim_obj = sim_obj
        self.susc = sim_state["susc"]
        self.target_var = sim_state["target_var"]

        self.lhs_boundaries = {
            "lower": np.zeros(self.sim_obj.upper_tri_size),
            "upper": np.ones(self.sim_obj.upper_tri_size)
                               }
        self.batch_size = 1000

    def run_sampling(self):
        lhs_table = self._get_lhs_table()
        contact_sim = list(map(self._get_cm_entries_lockdown_from_lhs, lhs_table))

        self._get_sim_output(contact_sim)

    def _get_cm_entries_lockdown_from_lhs(self, lhs_sample: np.ndarray):
        # Get ratio matrix
        ratio_matrix = get_rectangular_matrix_from_upper_triu(rvector=lhs_sample,
                                                              matrix_size=self.sim_obj.no_ag)
        # Get modified full contact matrix
        cm_sim = (1 - ratio_matrix) * (self.sim_obj.contact_matrix - self.sim_obj.contact_home)
        cm_sim += self.sim_obj.contact_home
        # Get output
        output = self._get_betas_from_contacts(cm_sim)
        cm_total_sim = cm_sim[self.sim_obj.upper_tri_indexes]
        output = np.append(cm_total_sim, output)
        return list(output)

    def _get_variable_parameters(self):
        return f'{self.susc}-{self.base_r0}-{self.target_var}'

    def _get_betas_from_contacts(self, cm_sim):
        cm = get_contact_matrix_from_upper_triu(rvector=cm_sim,
                                                age_vector=self.sim_obj.age_vector.reshape(-1, ))
        beta_lhs = self.base_r0 / self.r0generator.get_eig_val(contact_mtx=cm,
                                                               susceptibles=self.sim_obj.susceptibles.reshape(1, -1),
                                                               population=self.sim_obj.population)[0]
        output = np.array([0, beta_lhs])
        output = np.append(output, np.zeros(self.sim_obj.no_ag))
        return output


def get_contact_matrix_from_upper_triu(rvector, age_vector):
    new_2 = get_rectangular_matrix_from_upper_triu(rvector=rvector,
                                                   matrix_size=age_vector.shape[0])
    vector = np.array(new_2 / age_vector)
    return vector


def get_rectangular_matrix_from_upper_triu(rvector, matrix_size):
    upper_tri_indexes = np.triu_indices(matrix_size)
    new_contact_mtx = np.zeros((matrix_size, matrix_size))
    new_contact_mtx[upper_tri_indexes] = rvector
    new_2 = new_contact_mtx.T
    new_2[upper_tri_indexes] = rvector
    return np.array(new_2)



