# sensitivity-epimodel-2023


## Introduction
The purpose of this project is the sensitivity analysis of a COVID-19 model (set up from the data collected during the early stages of the outbreak in Hungary), where the parameters sampled are the distribution of vaccines between the 16 age groups the model uses. The repository was built upon [sensitivity-covid19-hun](https://gitlab.com/zsvizi/sensitivity-covid19-hun).

The flow of individuals between the compartments is as follows:
```mermaid
graph TD;
  A-->B;
  A-->C;
  B-->D;
  C-->D;
```
```mermaid
graph TD:
  S-->V;
  V-->S;
  S-->E;
  E-->I;
  I-->I_c;
  I-->H;
  I_c-->D;
  I_c-->I_r;
  I-->R;
  H-->R;
  I_r-->R;
```
For sensitivity analysis we use the following methods:
- LHS (Latin Hypercube Sampling): used for parameter sampling
- PRCC (Partial Rank Correlation Coefficient): used as a metric for sensitivity

For evaluation of the model we use the package 'torchdiffeq', and we represent the ODE system corresponding to the model with matrix operations, thus increasing efficiently.

## Structure
For running simulation:
- `simulation.py`: contains class `Simulation`, which runs the sampling and PRCC calculation for different scenarios
- `dataloader.py`: contains class `DataLoader` for loading data in an arranged format

Implementation of main methods:
- `matrix_generator.py`: contains class `MatrixGenerator` which creates the matrices used for the representation of the model
- `model.py`: contains classes `VaccinatedModel` and `ModelEq`, the first of which is responsible for encapsulating the necessary methods and model parameters for evaluation, and the second is a wrapper so that we can evaluate the model using CUDA with the help of the package `torchdiffeq` 
- `r0.py`: contains class `R0Generator` that calculates R0 for the model parameters, which we use to calculate the baseline transmission rate for given R0s, by factoring it out from the next-generation matrix 
- `prcc.py`: contains functions used for PRCC calculation and plotting of sensitivity data

Implementation of setups for experiments:
- `sampler.py`: contains 
