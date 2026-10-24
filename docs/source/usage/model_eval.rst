Model evaluation
################

Assuming all the inputs necessary, as discussed in the previous section
(:doc:`General use <./general_use>`) are ready and available, the following
will guide you in evaluating your model and creating a figure from the result.


Evaluating the results
======================


    .. code-block:: python


        import matplotlib.pyplot as plt
        import torch

        from emsa.model.epidemic_model import EpidemicModel


        data = ...
        model_struct = ...

        model = EpidemicModel(data=data, model_struct=model_struct)

        # Evaluation timestamps
        t_eval = torch.linspace(1, 300, 300).to(data.device)
        # Initial values
        iv_dict = {"e": [10]}
        iv = model.get_initial_values_from_dict(iv_dict)
        susceptibles = iv[model.idx("s_0")]
        # Evaluate model
        sol = model.get_solution(y0=iv, t_eval=t_eval).ys[0, :, :]

From this we can use the methods implemented by the EpidemicModelBase class to extract values of
specific compartments, and create a figure of our simulation


Plotting the results
====================


    .. code-block:: python


        # Iterate over compartments and plot them aggregated by age
        # (Erlang distributed states are aggregated as well)
        fig, ax = plt.subplots()
        for comp in model.state_data.keys():
            # Aggregate compartments by age:
            # returns compartments values in case of a single age group
            comp_sol = model.aggregate_by_age(sol, comp)
            # Plot compartments values and annotate them
            ax.plot(t_eval, comp_sol, label=comp.upper(), linewidth=2)
        # Pyplot magic
        ax.set_xlabel("Days")
        ax.set_ylabel("Population")
        ax.legend()
        plt.show()

Numerical settings
==================

Pass ``solver_config`` when creating an ``EpidemicModel``, or use ``solver`` in
the sensitivity sampling configuration. Individual ``get_solution()`` calls can
override settings with ``solver_config={...}``.

.. code-block:: python

   solver_config = {"method": "dopri5", "step_size": 0.25, "rtol": 1e-5, "atol": 1e-6}
   model = EpidemicModel(data, model_struct, solver_config=solver_config)

``method`` defaults to ``"euler"`` with a one-day ``step_size``. For ``"dopri5"``,
``step_size`` is the maximum adaptive step. Default tolerances are ``rtol=1e-5``
and ``atol=1e-6``; ``max_steps`` defaults to 100000.

Check convergence by reducing the step and comparing trajectories or targets.
Choose a resolution suitable for intervention timing. A coarse Euler step can
produce substantial error even when populations remain nonnegative.

Solver failure, non-finite output, or a population below ``-negative_tolerance``
raises ``RuntimeError``. The default ``negative_tolerance`` is ``1e-4`` people.
Input shapes, finite values and increasing evaluation times are checked as well.
The automated release checks exercise CPU execution on Linux and Windows.
