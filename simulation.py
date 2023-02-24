import numpy as np
from matplotlib import pyplot as plt

from dataloader import DataLoader
from model import VaccinatedModel
from prcc import get_prcc_values
from r0 import R0Generator
from sampler_vaccinated import SamplerVaccinated


class SimulationVaccinated:
    def __init__(self):
        # Load data
        self.data = DataLoader()

        # User-defined parameters
        self.susc_choices = [0.5, 1.0]
        self.r0_choices = [1.1, 1.35, 1.6, 2.5]

        # Define initial configs
        self._get_initial_config()

    def run(self):
        susceptibility = np.ones(self.no_ag)
        for susc in self.susc_choices:
            susceptibility[:4] = susc
            self.params.update({"susc": susceptibility})
            for base_r0 in self.r0_choices:
                r0generator = R0Generator(param=self.params)
                sim_state = {"base_r0": base_r0, "susc": susc, "r0generator": r0generator,
                             "daily_vaccines": 1000}

                param_generator = SamplerVaccinated(sim_state=sim_state, sim_obj=self)
                param_generator.run()
                self.generate_prcc_plot(params=param_generator.parameters,
                                        lhs_output=np.c_[param_generator.lhs_table,
                                                         param_generator.sim_output.T])

    def _get_initial_config(self):
        self.no_ag = self.data.contact_data["home"].shape[0]
        self.model = VaccinatedModel(model_data=self.data)
        self.population = self.model.population
        self.age_vector = self.population.reshape((-1, 1))
        self.susceptibles = self.model.get_initial_values()[self.model.c_idx["s"] *
                                                            self.no_ag:(self.model.c_idx["s"] + 1) * self.no_ag]
        self.contact_matrix = self.data.contact_data["home"] + self.data.contact_data["work"] + \
                              self.data.contact_data["school"] + self.data.contact_data["other"]
        self.contact_home = self.data.contact_data["home"]
        self.upper_tri_indexes = np.triu_indices(self.no_ag)
        self.params = self.data.model_parameters_data

    @staticmethod
    def generate_prcc_plot(params, lhs_output):
        prcc_val = np.round(get_prcc_values(lhs_output), 3)
        sorted_idx = np.abs(prcc_val).argsort()[::-1]

        prcc_val = prcc_val[sorted_idx]

        plt.title("PRCC values of vaccinated model parameters, target variable: R0", fontsize=15)

        ys = range(len(params))[::-1]
        # Plot the bars one by one
        for y, value in zip(ys, prcc_val):
            plt.broken_barh(
                [(value if value < 0 else 0, abs(value))],
                (y - 0.4, 0.8),
                facecolors=['white', 'white'],
                edgecolors=['black', 'black'],
                linewidth=1,
            )

            if value != 0:
                x = (value / 2) if np.abs(value) >= 0.15 else (- np.sign(value) * 0.1)
            else:
                x = -0.1
            plt.text(x, y, str(value), va='center', ha='center')

        plt.axvline(0, color='black')

        # Position the x-axis on the top, hide all the other spines (=axis lines)
        axes = plt.gca()  # (gca = get current axes)
        axes.spines['left'].set_visible(False)
        axes.spines['right'].set_visible(False)
        axes.spines['bottom'].set_visible(False)
        axes.xaxis.set_ticks_position('top')

        # Make the y-axis display the params
        plt.yticks(ys, params[sorted_idx])

        # Set the portion of the x- and y-axes to show
        plt.xlim(-1.1, 1.1)
        plt.ylim(-1, len(params))

        # plt.text()
        plt.show()
