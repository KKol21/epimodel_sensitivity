from src.simulation import SimulationVaccinated


def main():
    simulation = SimulationVaccinated()

   # simulation.run_sampling()
   # simulation.calculate_prcc()
   # simulation.plot_prcc()
    simulation.plot_optimal_vaccine_distribution()


if __name__ == '__main__':
    main()
