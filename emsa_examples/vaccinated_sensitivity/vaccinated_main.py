from emsa_examples.utils.dataloader_16_ag import DataLoader
from emsa_examples.vaccinated_sensitivity.simulation_vacc import SimulationVaccinated


def main():
    data = DataLoader()
    sim = SimulationVaccinated(data)
    sim.model.visualize_transmission_graph()
    sim.run_sampling()
    sim.calculate_all_prcc()
    sim.calculate_all_p_values()
    sim.plot_prcc_tornado_with_p_values()


if __name__ == '__main__':
    main()
