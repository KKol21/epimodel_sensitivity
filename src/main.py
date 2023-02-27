from simulation import SimulationVaccinated


def main():
    simulation = SimulationVaccinated()
    # simulation.run_sampling()
    simulation.calculate_prcc()
    simulation.plot_prcc()


if __name__ == '__main__':

    main()
