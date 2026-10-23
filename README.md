# EMSA - Epidemic Modeling, Sensitivity Analysis
[![CI](https://github.com/KKol21/epimodel_sensitivity/actions/workflows/cicd.yml/badge.svg?branch=dev)](https://github.com/KKol21/epimodel_sensitivity/actions/workflows/cicd.yml) [![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

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
github](https://github.com/KKol21/epimodel_sensitivity/issues/new).


## Installation
The current dependency set supports **Python 3.10 and 3.11**. You can install the
published package via pip:

```sh
pip install emsa
```

Or clone the development version using git:

```sh
git clone https://github.com/KKol21/epimodel_sensitivity
```

From the repository root, install the development checkout with:

```sh
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

For CPU execution on Linux or Windows, install the CPU PyTorch wheel first, then
install EMSA and the extras you need:

```sh
python -m pip install --upgrade pip
python -m pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev,notebook]"
```

The CPU index and version follow the [PyTorch installation instructions](https://pytorch.org/get-started/previous-versions/#v231).
Runtime dependencies live in `pyproject.toml`; `requirements.txt` installs the package,
`requirements-dev.txt` adds development tools, and `docs/requirements.txt` adds the
documentation tools. Run these requirements commands from the repository root.
See [RELEASING.md](RELEASING.md) for building, clean-install checks and tag-based releases.

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

Start with the executable [package walkthrough](notebooks/EMSA_demo.ipynb). It defines a
SIR model, plots a trajectory, checks numerical convergence, and runs reproducible
LHS/PRCC analysis entirely in memory. From a checkout, install with
`python -m pip install -e ".[notebook]"`, then open the notebook in JupyterLab or your IDE
and select that Python environment.

`SimulationGeneric(data, model_struct, sampling_config, save_results=False).run()`
accepts dictionaries and returns a mapping of scenario names to NumPy arrays under
`samples`, `targets`, `prcc`, `p_values`, plus `metadata`. JSON paths and the existing
sampling/analysis methods remain supported. Set `output_dir` and `save_results=True`
to export CSV files and metadata. Use a separate output directory per concurrent run;
reusing a directory replaces files with the same scenario names.
Empty scenario names add no extra separator: for example, `prcc_i_max.csv` and
`metadata.json`. Existing CSV exports with the older extra underscore remain readable.

Sampling configuration accepts `seed` and `solver`. Solver settings include `method`
(`euler` or `dopri5`), `step_size`, `rtol`, `atol`, `max_steps` and
`negative_tolerance`. Euler retains the original one-day default; choose a smaller
step for faster dynamics. For Dopri5, `step_size` caps the adaptive step size.
Solver failures, non-finite states and populations below `-negative_tolerance`
(default `1e-4` people) raise errors. These checks do not replace convergence checks
for your model, particularly for time-dependent interventions.

Explore the `emsa_examples/` directory to see how EMSA can be applied to various epidemic models,
or check out the
["Introduction to EMSA"](https://colab.research.google.com/drive/1TYhzxvmqc2MLg1ie5vhzmW0UNq8CXJos?usp=sharing)
Google Collaboratory notebook to get started. For general usage, try the
["EMSA template"](https://colab.research.google.com/drive/15qE-WYD_ZfVtYohcfkIaObj5kLQy-ro0?usp=sharing)
notebook!


## Contributing

We welcome contributions to EMSA! To contribute, please open an issue first to discuss the proposed changes. Afterwards
you can fork the repository, and create a merge request in order for your changes to be merged into the package.
