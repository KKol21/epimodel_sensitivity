EMSA: Epidemic Modeling, Sensitivity Analysis
=============================================

The purpose of this project is to provide a general framework for the sensitivity
analysis of deterministic compartmental epidemic models, as well as a tool for predictive modeling.
Sensitivity analysis is performed using Latin Hypercube Sampling (LHS) for parameter sampling
and Partial Rank Correlation Coefficient (PRCC) as a metric for sensitivity.

The pipeline is fully compatible with CUDA and provides an efficient representation of the
epidemic model used, enabling parallel evaluation based on different parameter combinations and
initial values. For evaluation of the model we use the package `torchode`, and we represent the
ODE system corresponding to the model with matrix operations.


Installation Guide
==================

This guide provides the essential steps to install EMSA on your system.

System Requirements
--------------------
- **Python:** Version 3.11 (should work from 3.9, but it may require adjusting the dependencies)
- **Operating System:** Windows, macOS, or Linux.
- **(Optional) CUDA:** 11.8+

Basic Installation
-------------------
Install EMSA using `pip`:


.. code-block::

   pip install emsa

To experiment with the latest code and explore implemented examples,
you can install the development version by cloning the repository:

.. code-block::

   git clone https://github.com/KKol21/epimodel-sensitivity



Optional: verify CUDA compatibility
-----------------------------------

After installing PyTorch, you can check if CUDA is available by running the following:

.. code-block::

   import torch
   print(torch.cuda.is_available())

For further instructions on setting up CUDA, please see the
`official PyTorch guide <https://pytorch.org/get-started/locally/>`_.
This is only relevant if you'd like to use a GPU for evaluation.


Getting started
===============

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   usage/index
   advanced_topics/index
   api/emsa
