from src.simulation import SimulationVaccinated


def main():
    simulation = SimulationVaccinated()
    simulation.run_sampling()
    #simulation.calculate_prcc()
    simulation.calculate_p_values()
    #simulation.plot_prcc()
    simulation.plot_optimal_vaccine_distributions()


if __name__ == '__main__':
    main()
