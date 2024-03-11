# Sensitivity analysis of epidemic models
This repository was created during the making of the Bsc thesis ["Sensitivity analysis of
age-specific vaccination in epidemic modelling"](https://drive.google.com/file/d/1ICZ_JTFdJ_zOCKJDFukfIcOrxkguWCFT/view?usp=sharing)(Hungarian) by K. Kovács, Zs. Vizi and B. Péter. It serves as a
starting point for the sensitivity analysis of epidemic models.

## Introduction
The purpose of this project is to provide a general framework for the sensitivity
analysis of deterministic compartmental epidemic models, as well as a tool for predictive modeling. 
Sensitivity analysis is performed using Latin Hypercube Sampling (LHS) for parameter sampling
and Partial Rank Correlation Coefficient (PRCC) as a metric for sensitivity. The pipeline is fully 
compatible with CUDA and provides an efficient representation of the epidemic model used, enabling 
parallel evaluation based on different parameter combinations and initial values.

For evaluation of the model we use the package `torchode`, and we represent the ODE system corresponding
to the model with matrix operations.

## Pipeline
The user shall provide the following:
- Model parameters
- Model structure
- Simulation config
- Population distribution
- Contact matrix

The exact format of the inputs are specified in the documentation.

The following flowchart represents the pipeline established in this project: ![Flowchart](/images/Flowchart.jpg)

The LHS tables as well as the outputs are saved in subfolders in the folder `sens_data`.
