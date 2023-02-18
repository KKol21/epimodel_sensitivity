import numpy as np

from plotter import plot_solution_inc, plot_solution_ic


class AnalysisNPI:
    def __init__(self, sim, susc, base_r0, mtx_type=None):
        self.sim = sim
        self.susc = susc
        self.base_r0 = base_r0
        self.mtx_type = mtx_type

    def run(self):
        cm_list = []
        legend_list = []

        self.get_full_cm(cm_list, legend_list)
        self.get_reduced_contact(cm_list, legend_list, 7, "work", 0.5)
        self.get_reduced_contact(cm_list, legend_list, 8, "work", 0.5)
        self.get_reduced_contact(cm_list, legend_list, 7, "other", 0.5)
        self.get_reduced_contact(cm_list, legend_list, 6, "work", 0.5)
        self.get_reduced_contact(cm_list, legend_list, 5, "work", 0.5)
        self.get_reduced_contact(cm_list, legend_list, 5, "other", 0.5)
        self.get_reduced_contact(cm_list, legend_list, 6, "other", 0.5)

        t = np.arange(0, 500, 0.5)

        if self.base_r0 == 1.35 and self.susc == 1:
            # R0 = 1.35, Susc = 1, Target: R0
            plot_solution_inc(self.sim, t, self.sim.params,
                              [cm_list[i] for i in [0, 1, 2, 3, 4]], [legend_list[i] for i in [0, 1, 2, 3, 4]],
                              "_R0target_half_".join([str(self.susc), str(self.base_r0)]))

            # R0 = 1.35, Susc = 1, Target: ICU
            plot_solution_ic(self.sim, t, self.sim.params,
                             [cm_list[i] for i in [0, 5, 6, 1, 7]], [legend_list[i] for i in [0, 5, 6, 1, 7]],
                             "_ICUtarget_half_".join([str(self.susc), str(self.base_r0)]))

    def get_full_cm(self, cm_list, legend_list):
        cm = self.sim.contact_matrix
        cm_list.append(cm)
        legend_list.append("Total contact")

    def get_reduced_contact(self, cm_list, legend_list, age_group, contact_type, ratio):
        contact_matrix_spec = np.copy(self.sim.data.contact_data[contact_type])

        contact_matrix_spec[age_group, :] *= ratio
        contact_matrix_spec[:, age_group] *= ratio
        contact_matrix_spec[age_group, age_group] *= (1/ratio if ratio > 0.0 else 0.0)

        full_contact_matrix = self.sim.contact_matrix - self.sim.data.contact_data[contact_type] + contact_matrix_spec

        cm_list.append(full_contact_matrix)
        legend_list.append("{r}% {c_type} reduction of a.g. {ag}".format(r=int((1-ratio)*100),
                                                                         c_type=contact_type, ag=age_group))

    def get_fix_reduced_contact(self, cm_list, legend_list, age_group, contact_type):
        cm_spec_total = np.copy(self.sim.data.contact_data[contact_type]) * self.sim.age_vector

        all_spec_contacts = np.sum(cm_spec_total[age_group, :])

        cm_spec_total[age_group, :] -= 1000000 * cm_spec_total[age_group, :] / all_spec_contacts
        cm_spec_total[:, age_group] = cm_spec_total[age_group, :].T
        contact_matrix_spec = cm_spec_total / self.sim.age_vector

        full_contact_matrix = self.sim.contact_matrix - self.sim.data.contact_data[contact_type] + contact_matrix_spec

        cm_list.append(full_contact_matrix)
        legend_list.append("1M {c_type} reduction of a.g. {ag}".format(c_type=contact_type, ag=age_group))
