from examples.vaccinated_sensitivity.simulation_vacc import SimulationVaccinated


def main():
    simulation = SimulationVaccinated()
    simulation.run_sampling()
    simulation.calculate_prcc()
    simulation.calculate_p_values()
    simulation.plot_prcc_with_p_values()
    #simulation.plot_optimal_vaccine_distributions()


if __name__ == '__main__':
    main()
