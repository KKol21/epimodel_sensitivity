from simulation_seir import SimulationSeir
from plotter import generate_stacked_plots, plot_contact_matrix_as_grouped_bars


def main():
    simulation = SimulationSeir()
    simulation.run()


if __name__ == '__main__':

    main()
    # generate_stacked_plots()
    # plot_contact_matrix_as_grouped_bars()
