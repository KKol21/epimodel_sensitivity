Using different numerical methods
#################################

Choose the built-in Euler or adaptive Dopri5 solver through ``solver_config``
on ``EpidemicModel``, or through ``solver`` in a sampling configuration:

.. code-block:: python

   solver_config = {"method": "dopri5", "step_size": 0.25, "rtol": 1e-5, "atol": 1e-6}

See :doc:`../usage/model_eval` for defaults, per-call overrides and convergence checks.

For custom equations, ``EpidemicModel.get_solution()`` accepts an ``odefun`` callable.
It receives times of shape ``(batch,)`` and states of shape ``(batch, equations)``
and must return derivatives with the same shape, device and dtype as the states.
``get_sol_from_ode()`` applies the configured integrator and output checks.

To use a different solver library, override ``get_solution(y0, t_eval, **kwargs)``
in your model subclass. Sensitivity target calculation expects the return value
to have a ``ys`` tensor of shape ``(batch, evaluation_times, equations)``.
