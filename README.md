# EMSA - Epidemic Modeling, Sensitivity Analysis
![Build Status](https://img.shields.io/badge/build-passing-brightgreen) ![License](https://img.shields.io/badge/license-MIT-blue)

## Introduction
EMSA is a Python package designed to simplify and optimize sensitivity
analysis for deterministic compartmental epidemic models. By leveraging CUDA for parallel processing, EMSA allows
users to efficiently evaluate how different parameters affect model outcomes, making it a powerful tool for researchers
and public health officials. Potential applications include outbreak forecasting, intervention planning, and
understanding the impact of varying disease parameters.

Sensitivity analysis is performed using Latin Hypercube Sampling (LHS) for parameter sampling
and Partial Rank Correlation Coefficient (PRCC) as a metric for sensitivity. For evaluation of the model
we use the package `torchode`, and we represent the ODE system corresponding to the model with matrix operations,
enabling parallel evaluation of the sampled parameter configurations.


- [*Documentation*](https://epimodel-sensitivity.readthedocs.io/)

If you get stuck at some point, you think the library should have an example on _x_ or you
want to suggest some other type of improvement, please open an [issue on
github](https://github.com/KKol21/epimodel-sensitivity/issues/new).


## Installation
You can install EMSA via pip:

```sh
pip install emsa
```

Or clone the development version using git:

```sh
git clone https://github.com/KKol21/epimodel-sensitivity
```


## Pipeline Overview
EMSA requires the following inputs:
- **Model Parameters:** Key values that define the dynamics of the epidemic model.
- **Model Structure:** The compartments and transitions between them.
- **Sampling Configuration:** Initial values of the simulation, sampled parameters and their ranges, etc.
- **Age Vector:** Age distribution of the population.
- **Contact Matrix:** Interaction rates between different population groups.

These inputs are processed through the following steps:
1. **Parameter Sampling:** Latin Hypercube Sampling (LHS) generates parameter sets.
2. **Model Evaluation:** Each parameter set is evaluated using the specified epidemic model.
3. **Sensitivity Measure:** Partial Rank Correlation Coefficient (PRCC) is calculated to assess parameter sensitivity.

![Flowchart](/images/Flowchart.png)


## Documentation and Examples
Full documentation is available on [Read the Docs](https://epimodel-sensitivity.readthedocs.io/).

Explore the `examples/` directory to see how EMSA can be applied to various epidemic models,
or check out the
["Introduction to EMSA"](https://colab.research.google.com/drive/1TYhzxvmqc2MLg1ie5vhzmW0UNq8CXJos?usp=sharing)
Google Collaboratory notebook to get started. For general usage, try the
["EMSA template"](https://colab.research.google.com/drive/15qE-WYD_ZfVtYohcfkIaObj5kLQy-ro0?usp=sharing)
notebook!


## Contributing

We welcome contributions to EMSA! To contribute, please open an issue first to discuss the proposed changes. Afterwards
you can fork the repository, and create a merge request in order for your changes to be merged into the package.
