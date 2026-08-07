Tutorials and examples
======================

This learning path introduces stationary quantum transport before moving to
QmeQ's more advanced master-equation methods.  It is intended for master's
students with limited prior exposure to quantum transport or open quantum
systems.

Work through the numbered tutorials in order.  Each notebook starts from a
physical question, makes a prediction, constructs the smallest useful model,
and checks the numerical result.

Foundations
-----------

Sequential transport with the Pauli master equation: the model, the
interactions, and the measurement that experiments actually report.

.. toctree::
   :maxdepth: 1

   Before you begin <getting_started>
   1. A first transport calculation <first_transport_calculation>
   2. Coulomb blockade and many-body states <coulomb_blockade>
   3. Bias and gate sweeps <bias_and_gate_sweeps>

Choosing and trusting an approximation
--------------------------------------

Where sequential transport stops being enough: coherences between dot states,
the energy that the charge carries, and second-order tunnelling.

.. toctree::
   :maxdepth: 1

   4. Coherence in a double dot and choosing an approximation <coherence_and_approximations>
   5. Energy and heat transport <energy_and_heat_transport>
   6. Cotunnelling and second-order methods <cotunnelling_and_second_order>
   7. Current noise from counting statistics <counting_statistics>

The notebooks are rendered directly from ``examples/tutorials/``.  Short,
self-contained programs suitable for copying into a new project remain in
``examples/scripts/``.  Electron-phonon transport is not covered by the
tutorials; see ``qmeq.BuilderElPh`` and the API documentation.

Reference and legacy material
-----------------------------

.. toctree::
   :maxdepth: 1

   State types <types>
   Symmetries <symmetries>
   Legacy tutorials <legacy>
