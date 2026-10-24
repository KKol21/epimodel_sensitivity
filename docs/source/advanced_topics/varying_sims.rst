Running simulations with varying parameters
###########################################

Sampled parameters vary within one LHS analysis. Scenario parameters remain fixed
for an analysis and change between runs. A custom simulation subclass can iterate
over ``variable_param_combinations`` and pass each scenario to ``GenericSampler``
so its outputs get a separate name.

Using the SIR inputs from :doc:`../usage/quickstart`, this example compares two
fixed recovery rates while sampling only the transmission rate:

.. code-block:: python

   from copy import deepcopy
   from emsa.generics import GenericSampler, SimulationGeneric

   class ScenarioSimulation(SimulationGeneric):
       def run_sampling(self):
           baseline = dict(self.params)
           try:
               for scenario in self.variable_param_combinations:
                   self.params.update(scenario)
                   GenericSampler(self, variable_params=scenario).run()
           finally:
               self.params.clear()
               self.params.update(baseline)

   scenarios = deepcopy(config)
   scenarios["variable_params"] = {"gamma": [0.1, 0.2]}
   scenarios["sampled_params_boundaries"] = {"beta": [0.3, 0.5]}
   scenarios["target_vars"] = ["i_max"]
   simulation = ScenarioSimulation(data, structure, scenarios, save_results=False)
   results = simulation.run()
   print(results.keys())  # gamma-0.1 and gamma-0.2

Multiple entries in ``variable_params`` produce their Cartesian product.
Apply any scenario-specific transformations in ``run_sampling()``: for example,
the age-structured examples convert a requested reproduction number into beta.
The repository's ``SimulationSEIR`` and ``SimulationSEIHR`` show this pattern.
