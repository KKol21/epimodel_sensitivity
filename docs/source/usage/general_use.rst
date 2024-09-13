General usage
=============


Inputs
------

No matter which use case you choose, EMSA will always need the following inputs:
- **Model Parameters:** Key values that define the dynamics of the epidemic model.
- **Model Structure:** The compartments and transitions between them.
- **Age Vector:** Age distribution of the population.
- **Contact Matrix:** Interaction rates between different population groups.

In addition to that, if you are performing sensitivity analysis, you will need to provide a
sampling configuration, which can include data such as the initial values of the simulation,
sampled parameters and their ranges, varying hyperparameters, and target variables.


Model structure
---------------

Models can be defined using the following templates:

State template:

Transition template:

Transmission template:


Model parameters
----------------


Age vector and contact matrix
-----------------------------
