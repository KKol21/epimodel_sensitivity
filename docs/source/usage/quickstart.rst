Quickstart
==========

Install ``emsa`` with Python 3.10 or 3.11. This CPU example needs no external data
and runs directly from the installed package.

Run a sensitivity analysis
--------------------------

.. code-block:: python

   from types import SimpleNamespace

   import torch
   from emsa.generics import SimulationGeneric

   data = SimpleNamespace(
       params={"beta": 0.4, "gamma": 0.15},
       age_data=torch.tensor([1000.0]),
       cm=torch.tensor([[1.0]]),
       n_age=1,
       device="cpu",
   )
   structure = {
       "state_data": {
           "s": {"type": "susceptible"},
           "i": {"type": "infected", "rate": "gamma"},
           "r": {"type": "recovered"},
       },
       "tms_rules": [{"source": "s", "target": "i", "actors-params": {"i": None}}],
       "trans_data": [{"source": "i", "target": "r"}],
   }
   config = {
       "target_vars": ["i_max", "r_sup"],
       "sampled_params_boundaries": {"beta": [0.3, 0.5], "gamma": [0.1, 0.2]},
       "init_vals": {"i": [10.0]},
       "n_samples": 32,
       "batch_size": 16,
       "seed": 42,
       "solver": {"method": "dopri5", "step_size": 1.0},
   }

   simulation = SimulationGeneric(data, structure, config, save_results=False)
   result = simulation.run()[""]
   print(result["prcc"]["i_max"])

``run()`` returns a dictionary keyed by scenario name; ``""`` is the default
scenario. Each result contains:

- ``samples``: a NumPy array of shape ``(n_samples, n_sampled_columns)``.
- ``targets``: one NumPy vector per requested target, with one value per sample.
- ``prcc`` and ``p_values``: one coefficient or p-value per sampled column and target.
- ``metadata``: the seed, input configuration, population, parameters and solver settings.

Repeating the same seeded configuration reproduces the samples. ``save_results=False``
keeps the run in memory. For CSV files and metadata, pass
``output_dir="sens_data/my_run", save_results=True`` to the constructor and call
``run()``. Give separate runs separate output directories to retain their results.
Use ``simulation.plot_all_prcc()`` after an exported run to create the tornado plots.

Evaluate a trajectory
---------------------

The same data and structure also work without sampling:

.. code-block:: python

   from emsa.model import EpidemicModel

   model = EpidemicModel(data, structure, solver_config=config["solver"])
   initial = model.get_initial_values_from_dict({"i": [10.0]})
   times = torch.arange(61.0)
   trajectory = model.get_solution(initial, times).ys[0]
   infected = model.aggregate_by_age(trajectory, "i")

See :doc:`model_eval` for plotting and numerical settings, :doc:`inputs` for
configuration fields, and :doc:`sens_anal` for target definitions and PRCC interpretation.

Examples and notebooks
----------------------

The repository and source archive include ``emsa_examples/`` and
``notebooks/EMSA_demo.ipynb``. The wheel installs the ``emsa`` library. To run the
repository examples and notebook, clone the repository and install the extras:

.. code-block:: console

   python -m pip install -e ".[test,notebook]"

Open the notebook in JupyterLab or your IDE and select that Python environment.
