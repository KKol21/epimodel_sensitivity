# sensitivity-epimodel-2023


## Introduction
The purpose of this project is the sensitivity analysis of a COVID-19 model (set up from the data collected during the early stages of the outbreak in Hungary), where the parameters sampled are the distribution of vaccines between the 16 age groups the model divides the population into. The repository was built upon [sensitivity-covid19-hun](https://gitlab.com/zsvizi/sensitivity-covid19-hun).

The hyperparameters of the model, including the number of substates corresponding to the transitional states (not susceptible/dead/recovered), all available vaccines to be split among the population, and the start and length of the vaccination period can be configured in the file `model_parameters.json`.

For sensitivity analysis we use the following methods:
- LHS (Latin Hypercube Sampling): used for parameter sampling
- PRCC (Partial Rank Correlation Coefficient): used as a metric for sensitivity

For evaluation of the model we use the package `torchdiffeq`, and we represent the ODE system corresponding to the model with matrix operations,leading to increased efficiency.

## Structure
For running simulation:
- `simulation.py`: contains class `Simulation`, which runs the sampling and PRCC calculation for different scenarios (base reprodcution number, susceptibility of age groups, target variable used)
- `dataloader.py`: contains class `DataLoader` for loading data in an arranged format

Implementation of main methods:
- `matrix_generator.py`: contains class `MatrixGenerator` which creates the matrices used for the representation of the model
- `model.py`: contains classes `VaccinatedModel` and `ModelEq`, the first of which is responsible for encapsulating the necessary methods and model parameters for evaluation, and the second is a wrapper so that we can evaluate the model using CUDA with the help of the package `torchdiffeq` 
- `r0.py`: contains class `R0Generator` that calculates R0 for the model parameters, which we use to calculate the baseline transmission rate for given R0s, by factoring it out from the next-generation matrix 
- `prcc.py`: contains functions used for PRCC calculation and plotting of sensitivity data

Implementation of setups for experiments:
- `sampler.py`: contains 
