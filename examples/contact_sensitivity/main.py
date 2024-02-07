from examples.contact_sensitivity.simulation_contact import SimulationContact
from src.dataloader import DataLoader


def main():
    data = DataLoader
    sim = SimulationContact(data)
    sim.run_sampling()
    sim.calculate_prcc()
    sim.calculate_p_values()
    #sim.plot_prcc_and_p_values()


if __name__ == '__main__':
    main()
