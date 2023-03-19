from simulation import SimulationVaccinated
import torch


def main():
    print(torch.cuda.is_available(), torch.version.cuda)
    simulation = SimulationVaccinated()
    simulation.run_sampling()
    simulation.calculate_prcc()
    simulation.plot_prcc()


if __name__ == '__main__':

    main()
