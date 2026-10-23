Sensitivity analysis
####################


A generic description of the SA pipeline implemented by EMSA is as follows:

1. **Parameter Sampling:**  Latin Hypercube Sampling (LHS) is used to generate diverse sets of parameter values.
2. **Model Evaluation:** Each parameter set is evaluated using the specified epidemic model.
3. **Sensitivity Measure:** Partial Rank Correlation Coefficient (PRCC) is calculated to assess the sensitivity of model outputs to input parameters.
4. (Optional) **Plot results:** Figures can be generated using either EMSA's built-in plotting functionality or a custom plotting method.

EMSA workflow followed in implemented examples:

.. image:: https://i.postimg.cc/L8jM6D65/1o0a-POj-Kxq-EU7-X-Kmt-f-Pqm7-V4-BD6-VCn-Zs-GNlr-R3-I4yys-GE2o7-A.png


Use cases requiring additional code
***********************************

EMSA streamlines the sensitivity analysis by different degrees based on your use case. The simplest is when the
following conditions are fulfilled:

- **Basic Model structure:** The analysed model has a structure that can be describe by the standard EMSA template
- **Simple Sampling Process:** The sampling is from uniform distribution, and there are no additional operations done to the samples or the target values.
- **Simple Target Calculation:** The target calculation process is in line with the one implemented in TargetCalculator. See documentation for caveats.
- **Simple Model Evaluation:** For a more complex ODE solver than the one in EpidemicModelBase, you will also need to implement that for your use case.

All of these use cases are discussed in the Advanced Topics section.


Basic use case
--------------

Now we will discuss the simplest use case, which is generally when the aforementioned conditions are fulfilled. The
implementation (assuming the required inputs are available) is as follows:


.. code-block:: python


    from emsa.generics import SimulationGeneric

    data = ...
    model_struct_path = '...'
    sampling_config_path = '...'

    sim = SimulationGeneric(
        data=data,
        model_struct_path=model_struct_path,
        sampling_config_path=sampling_config_path
    )

    sim.run_sampling()             # Perform sampling
    sim.calculate_all_prcc()       # Calculate Partial Rank Correlation Coefficients (PRCC)
    sim.calculate_all_p_values()   # Calculate p-values
    sim.plot_all_prcc()            # Plot PRCC results

Running this code will execute the full simulation pipeline, including sampling, PRCC calculation, and visualization
using tornado plots.


Target calculation
------------------

The built-in ``comp_max`` stopping rule assumes a single epidemic peak: a declining
compartment total, or extinction, completes that peak target. Maxima are retained
across consecutive time windows. ``comp_sup`` is intended for cumulative compartments
such as recovered or dead; it returns their value once the total in states marked
``infected`` falls below one. A sample finishes when all its requested targets finish.
If any samples remain unfinished at ``tlim_final``, calculation raises ``RuntimeError``
instead of returning placeholder zeros. Increase the horizon or inspect the model.


Statistical results
-------------------

PRCC uses average ranks for ties. Constant or non-finite columns and dependent input
columns raise ``ValueError`` because their correlations are not identifiable. If an
output has no variation left after conditioning on the other inputs, the affected
coefficient and p-value are ``NaN``, a warning is emitted, and its tornado-plot entry
is labelled ``undefined``. Other identifiable coefficients are still returned.
P-values use ``n_samples - p - 1`` degrees of freedom, where ``p`` is the number of
sampled columns, including separate columns for age-specific parameters.
