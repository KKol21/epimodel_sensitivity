General Usage
=============

Inputs
******

No matter which use case you choose, EMSA will always need the following inputs:

- **Model Parameters:** Key values that define the dynamics of the epidemic model.
- **Model Structure:** The compartments and transitions between them.
- **Age Vector:** Age distribution of the population.
- **Contact Matrix:** Interaction rates between different population groups.

In addition, if you are performing sensitivity analysis, you will need to provide a
**sampling configuration**, which can include data such as the initial values of the simulation,
sampled parameters and their ranges, varying hyperparameters, and target variables.

For further description see :doc:`Inputs <./inputs>`.


Passing the inputs to the package
=================================

Model data
----------

For the more 'static' part of the inputs (age vector, contact matrix, model parameters), we need to create an object,
which will contain these as member variables, allowing us to . For this, you can either:


1. Create a class called e.g. "DataLoader", inheriting from :ref:`DataLoaderBase <dataloader_section>`.

Example: DataLoader class with separate methods for loading different parts of model data

    .. code-block:: python

       import torch
       from emsa.utils.dataloader import DataLoaderBase

       # Custom DataLoader class
       class DataLoader(DataLoaderBase):
           def __init__(self):
               super().__init__()

               # Load age data, model parameters, and contact matrix
               self._get_age_data()
               self._get_model_parameters_data()
               self._get_contact_mtx()

               # Set the computation device to CPU
               self.device = "cpu"

           # Implementation of the methods need to load and preprocess the data

2. Load them into a dictionary and create a SimpleNamespace instance from that dictionary

Example: Model data namespace for an SIR model without age groups

    .. code-block:: python

       import torch
       from types import SimpleNamespace

       # Model parameters
       params = {"gamma": 0.2, "alpha": 0.3}

       # Contact matrix (1D tensor for simplicity)
       contact_data = torch.tensor(1)

       # Age distribution (single age group with population size of 10,000)
       age_data = torch.tensor([10000])

       # Combine data into a namespace
       data = SimpleNamespace(
           **{
               "params": params,           # Model parameters
               "cm": contact_data,         # Contact matrix
               "age_data": age_data,       # Age distribution
               "n_age": 1,                 # Number of age groups
               "device": "cpu"             # Computation device (CPU)
           }
       )


The first option is more pythonic, however if the data loading logic is simple, option 2 can suffice.


Model structure
---------------