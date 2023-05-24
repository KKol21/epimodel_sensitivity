from src.simulation import SimulationVaccinated
from plotter import get_hmap

def main():
    simulation = SimulationVaccinated()
    get_hmap(simulation.contact_matrix)
   # simulation.plot_subopt()
   # simulation.run_sampling()
   # simulation.calculate_prcc()
   # simulation.plot_prcc()
    #simulation.plot_optimal_vaccine_distributions()


if __name__ == '__main__':
    main()
