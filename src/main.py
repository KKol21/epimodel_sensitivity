from src.simulation import SimulationVaccinated
from benchmarking import benchmark_evaluation

def main():
    simulation = SimulationVaccinated()
    benchmark_evaluation(simulation)

    simulation.run_sampling()
    simulation.calculate_prcc()
    simulation.plot_prcc()


if __name__ == '__main__':
    main()
