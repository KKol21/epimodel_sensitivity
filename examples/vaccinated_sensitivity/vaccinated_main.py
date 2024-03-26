from examples.utils.dataloader_16_ag import DataLoader
from examples.vaccinated_sensitivity.simulation_vacc import SimulationVaccinated


def main():
    data = DataLoader()
    simulation = SimulationVaccinated(data)
    simulation.run_sampling()
    simulation.calculate_all_prcc()
    simulation.calculate_all_p_values()
    simulation.plot_prcc_tornado_with_p_values()


if __name__ == '__main__':
    main()
