# sensitivity-epimodel-2023
This repository was created for the creation of the Bsc thesis "Sensitivity analysis of age-specific
vaccination in epidemic modelling" by K. Kovács and Zs. Vizi.

## Introduction
The purpose of this project is the sensitivity analysis of a COVID-19 model (set up from the data collected
during the early stages of the outbreak in Hungary), where the parameters sampled are the distribution of
vaccines between the 16 age groups the model divides the population into. 
The repository was built upon [sensitivity-covid19-hun](https://gitlab.com/zsvizi/sensitivity-covid19-hun).

The hyperparameters of the model, including the number of substates corresponding to the transitional
states (not susceptible/dead/recovered), all available vaccines to be split among the population, and the
start and length of the vaccination period can be configured in the file `model_parameters.json`.

For sensitivity analysis we use the following methods:
- LHS (Latin Hypercube Sampling): used for parameter sampling
- PRCC (Partial Rank Correlation Coefficient): used as a metric for sensitivity

For evaluation of the model we use the package `torchdiffeq`, and we represent the ODE system corresponding
to the model with matrix operations, in order to increase efficiency.

## Structure
Implementation of methods used for model evaluation and sensitivity analysis:
- `matrix_generator.py`: contains class `MatrixGenerator` which creates the matrices used for calculating the base
reproduction number and the representation of the model
- `model.py`: contains classes `VaccinatedModel` and `ModelEq`, the first of which is responsible for encapsulating
the necessary methods and model parameters for evaluation, and the second is a wrapper so that we can evaluate 
the model using CUDA with the help of the package `torchdiffeq` 
- `r0.py`: contains class `R0Generator` that calculates the spectral norm of the NGM with the baseline transmission 
rate factored out, enabling the calculation of said transmission rate for given R0s.
- `prcc.py`: contains functions used for PRCC calculation and creation of tornado plots from sensitivity data
- `sampler.py`: contains class `SamplerVaccinated` used for creating LHS tables

Miscellaneous files:
- `simulation.py`: contains class `Simulation`, which runs the sampling and PRCC calculation for different 
scenarios (base reproduction number, susceptibility of age groups, target variable used)
- `dataloader.py`: contains class `DataLoader` for loading data in an arranged format from the `data` folder

##
After running the simulation, the folder `sens_data` is created, containing the LHS tables, the value of the target 
variables in corresponding order to the samples, the PRCC values calculated, and the tornado plots created, all in
their respective subfolders.


