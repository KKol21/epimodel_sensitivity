from src.simulation.simulation_vacc import SimulationVaccinated
from src.simulation.simulation_contact import SimulationContact


def main():
    sim_type = "vacc"  # vacc, contact, param
    if sim_type == "vacc":
        simulation = SimulationVaccinated()
        simulation.run_sampling()
        simulation.calculate_prcc()
        simulation.calculate_p_values()
        simulation.plot_prcc()
        simulation.plot_optimal_vaccine_distributions()
    elif sim_type == "contact":
        sim = SimulationContact()
        sim.run_sampling()
        sim.calculate_prcc()
        sim.calculate_p_values()
        sim.plot_prcc_and_p_values()


if __name__ == '__main__':
    main()
