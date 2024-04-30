from examples.contact_sensitivity.simulation_contact import SimulationContact
from examples.utils.dataloader_16_ag import DataLoader


def main():
    data = DataLoader()
    sim = SimulationContact(data)
    sim.model.visualize_transmission_graph()
    sim.run_sampling()
    sim.calculate_all_prcc()
    sim.calculate_all_p_values()
    sim.run_func_for_all_configs(sim.plot_prcc_and_p_values)


if __name__ == '__main__':
    main()
