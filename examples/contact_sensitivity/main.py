from examples.contact_sensitivity.simulation_contact import SimulationContact


def main():
    sim = SimulationContact()
    sim.run_sampling()
    sim.calculate_prcc()
    sim.calculate_p_values()
    #sim.plot_prcc_and_p_values()


if __name__ == '__main__':
    main()
