Model structure Templates
=============================

This section explains how to define the different parts of your epidemic model using JSON format.
Below are templates for defining states, transitions, and transmission rules in your model.
You can also use the `dedicated structure creator app <https://epimodel-graph-editor.vercel.app/>`_,
where you can visualize your model as well.


State Template
--------------
To define the states (compartments) of the model, use the following structure. The `type` of state
(e.g., susceptible, infected) and the `rate` of transition out of the state can be specified.
Substates represent multiple stages within a state. The optional ``n_substates``
setting gives the number of stages; the template below uses three as an example.

.. code-block:: json

   {
     "state_data": {
       "state_name": {
         "type": "state_type",
         "n_substates": 3,
         "rate": "rate_name"
       }
     }
   }

**Example:**

.. code-block:: json

    {
     "state_data": {
       "s": {
          "type": "susceptible"
        },
       "e": {
          "type": "infected",
          "rate": "alpha"
        },
       "i": {
          "type": "infected",
          "rate": "gamma"
        },
       "h": {
          "n_substates": 1,
          "rate": "gamma_h"
        },
        "r": {
          "type": "recovered"
        }
      }
    }


Transition Template
-------------------
Define how individuals move from one state to another with the transition rules. Transitions may also involve parameters that control how individuals progress between states.

.. code-block:: json

   {
     "trans_data": [
       {
         "source": "source_state",
         "target": "target_state",
         "params": ["param_name"]
       }
     ]
   }

Parameters have to match a key in the model_params dictionary.
If a param ends with "_", the value of 1 - param will be used.

**Example:**

.. code-block:: json

    {
      "trans_data": [
        {
          "source": "i",
          "target": "r"
        },
        {
          "source": "e",
          "target": "i",
          "params": [
            "eta_"
          ]
        },
        {
          "source": "e",
          "target": "h",
          "params": [
            "eta"
          ]
        },
        {
          "source": "h",
          "target": "r"
        }
      ]
    }

Transmission Template
---------------------
This section describes how individuals move from the susceptible state to an infected state. Define which
infected states contribute to transmission and any parameters influencing the transmission process.

.. code-block:: json

   {
     "tms_rules": [
       {
         "source": "susceptible_state",
         "target": "infected_state",
         "actors-params": {
           "infectious_state_1": "param_1"
         },
         "susc_params": ["susc"],
         "infection_params": ["inf"]
       }
     ]
   }

**Example:**

The transmission rule for a simple SEIR model would look like this:

.. code-block:: json

  {
    "tms_rules": [
      {
        "source": "s",
        "target": "e",
        "actors-params": {
          "i": null
          }
      }
    ]
  }


Conclusion
----------

Use these templates as a guide when creating your own configuration files for the EMSA framework.
Customize the states, transitions, and transmission rules to fit the specific dynamics of the
epidemic model you're working on.
