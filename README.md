# Sensitivity analysis of age-specific vaccination in epidemic modelling
This repository was created during the making of the Bsc thesis ["Sensitivity analysis of
age-specific vaccination in epidemic modelling"](https://drive.google.com/file/d/1ICZ_JTFdJ_zOCKJDFukfIcOrxkguWCFT/view?usp=sharing)(Hungarian) by K. Kovács, Zs. Vizi and B. Péter. It serves as a
starting point for the sensitivity analysis of age-specific vaccination strategies.

## Introduction
The purpose of this project is the sensitivity analysis of a COVID-19 model (or any compartmental model), based on
the distribution of vaccines between the age groups.

The hyperparameters of the model, including the number of substates corresponding to the transitional
states, all available vaccines to be split among the population, and the
start and length of the vaccination period can be configured in the file `model_parameters.json`.

Sensitivity analysis is performed using Latin Hypercube Sampling (LHS) for parameter sampling
and Partial Rank Correlation Coefficient (PRCC) as a metric for sensitivity.

For evaluation of the model we use the package `torchdiffeq`, and we represent the ODE system corresponding
to the model with matrix operations to increase efficiency.

## Structure
Implementation of methods used for model evaluation and sensitivity analysis:
- `matrix_generator.py`: contains class `MatrixGenerator` which creates the matrices used for calculating the base
reproduction number and the representation of the model
- `model.py`: contains classes `VaccinatedModel` and `ModelEq`, the first of which is responsible for encapsulating
the necessary methods and model parameters for evaluation, and the second is a wrapper so that we can evaluate 
the model using CUDA with the help of the package `torchdiffeq`. As of now, the evaluation on GPU is inefficient, 
the package `torchode` should be used in further efforts.
- `r0.py`: contains class `R0Generator` that calculates the spectral norm of the NGM with the baseline transmission 
rate factored out, enabling the calculation of said transmission rate for given R0s.
- `prcc.py`: contains function used for PRCC calculation
- `sampler.py`: contains class `SamplerVaccinated` used for creating LHS tables

Miscellaneous files:
- `simulation.py`: contains class `Simulation`, which encapsulates the methods used to run the sampling and PRCC
calculation for different scenarios (base reproduction number, susceptibility of age groups, target variable used), 
and the plotting of data obtained from aforementioned methods
- `dataloader.py`: contains class `DataLoader` for loading data in an arranged format from the `data` folder
- `plotter.py`: contains functions used for creating tornado plots from sensitivity data and plotting the course of the
epidemic wrt. time for given paramers

## Pipeline
The pipeline is fully compatible with CUDA and provides an efficient representation of the epidemic model used.
Although the evaluation of the model has not been parallelized yet due to time constraints, the project can
serve as an excellent starting point for evaluating compartmental epidemic models on the GPU in Python.

The following flowchart represents the pipeline established in this project: ![Flowchart](/images/flowchart.jpg)

By running the methods contained in `simulation.py`, the folder `sens_data` is created, containing the LHS tables, 
the value of the target variables in corresponding order to the samples, the PRCC values calculated, the tornado 
plots, and the plots of the epidemic in the subfolders `lhs`, `simulations`, `prcc`, 
`prcc_plots`, and `epidemic_plots`, respectively.
