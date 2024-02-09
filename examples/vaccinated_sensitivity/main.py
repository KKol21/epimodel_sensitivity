from examples.vaccinated_sensitivity.simulation_vacc import SimulationVaccinated
from src.dataloader import DataLoader


def main():
    data = DataLoader()
    simulation = SimulationVaccinated(data)
    simulation.run_sampling()
    simulation.calculate_prcc_for_simulations()
    simulation.calculate_p_values()
    simulation.plot_prcc_tornado_with_p_values()
    simulation.plot_optimal_vaccine_distributions()


if __name__ == '__main__':
    main()
