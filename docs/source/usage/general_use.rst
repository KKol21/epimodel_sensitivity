General Usage
=============

Inputs
------

No matter which use case you choose, EMSA will always need the following inputs:

- **Model Parameters:** Key values that define the dynamics of the epidemic model.
- **Model Structure:** The compartments and transitions between them.
- **Age Vector:** Age distribution of the population.
- **Contact Matrix:** Interaction rates between different population groups.

In addition, if you are performing sensitivity analysis, you will need to provide a
sampling configuration, which can include data such as the initial values of the simulation,
sampled parameters and their ranges, varying hyperparameters, and target variables.


Model Structure
---------------

To define the compartments and transitions in your epidemic model, you can refer to the
:doc:`Model structure templates <./model_struct>`. These templates will guide you through how to create
states, transitions, and transmission rules for your model configuration.

Model Parameters
----------------

Model parameters represent the rates or probabilities that govern transitions between states in the epidemic model.
They are provided as key-value pairs where the key represents the parameter name and the value is the corresponding
numeric value (e.g., a rate of recovery, transmission probability, etc.).

Example of model parameters in JSON format:

.. code-block:: json

   {
     "gamma": 0.2,   // Recovery rate
     "alpha": 0.3    // Incubation rate
   }

- **gamma**: This parameter could represent the recovery rate for an infected individual transitioning to the recovered state.
- **alpha**: This parameter might represent the rate at which individuals leave the incubation stage and become infectious.

You can define as many parameters as required by your model.

Age Vector
-----------

This is a 1D array (or list) representing the population distribution across different age groups.
Each entry in the vector corresponds to the proportion or number of individuals in a specific age group.


Contact matrix
--------------

The contact matrix is a 2D array representing interaction rates between different age groups. Symmetrization is
performed to ensure that the interaction rates between age groups are more consistent by averaging the elements
of the matrix with the elements of its transpose, and then adjusting for population size. This step ensures that
the contact matrix reflects realistic interaction rates between age groups, especially when the sizes of those age
groups differ significantly.

Symmetrization Process:
***********************

1. **Averaging with Transpose**: The contact matrix is symmetrized by averaging the element at position `C[i, j]`
with the corresponding element at `C[j, i]`. This step ensures that the interaction rate between age group `i` and
age group `j` is consistent from both perspectives.

2. **Adjustment by Population Size**: After averaging, each row of the matrix is normalized by dividing it by
the size of the age group corresponding to that row. This step accounts for differences in the population sizes
of each age group.

After these operations, the following will hold:

   .. math::

     C'[i, j] = \frac{C[i, j] N_i + C[i, j] N_j}{2N_i},

where the element `C[i, j]` represents the *average number of interactions* a member of age
group `i` has with members of age group `j`. Likewise, `C[j, i]` represents the average number of interactions
a member of age group `j` has with members of age group `i`.

