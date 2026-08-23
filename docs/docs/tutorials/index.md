# Tutorials and examples

This learning path introduces stationary quantum transport before moving to
QmeQ's more advanced master-equation methods. It is intended for master's
students with limited prior exposure to quantum transport or open quantum
systems.

Work through the numbered tutorials in order. Each notebook starts from a
physical question, makes a prediction, constructs the smallest useful model,
and checks the numerical result.

## Foundations

Sequential transport with the Pauli master equation: the model, the
interactions, and the measurement that experiments actually report.

- [Before you begin](getting-started.md)
- [1. A first transport calculation](../notebooks/tutorials/01_first_transport_calculation.ipynb)
- [2. Coulomb blockade and many-body states](../notebooks/tutorials/02_coulomb_blockade.ipynb)
- [3. Bias and gate sweeps](../notebooks/tutorials/03_bias_and_gate_sweeps.ipynb)

## Choosing and trusting an approximation

Where sequential transport stops being enough: coherences between dot states,
the energy that the charge carries, and second-order tunnelling.

- [4. Coherence in a double dot and choosing an approximation](../notebooks/tutorials/04_coherence_and_approximations.ipynb)
- [5. Energy and heat transport](../notebooks/tutorials/05_energy_and_heat_transport.ipynb)
- [6. Cotunnelling and second-order methods](../notebooks/tutorials/06_cotunnelling_and_second_order.ipynb)
- [7. Current noise from counting statistics](../notebooks/tutorials/07_counting_statistics.ipynb)

The notebooks are rendered directly from `examples/tutorials/`. Short,
self-contained programs suitable for copying into a new project remain in
`examples/scripts/`. Electron-phonon transport is not covered by the
tutorials; see `qmeq.BuilderElPh` and the API documentation.

## Reference and legacy material

- [State types](../notebooks/appendix/00_types.ipynb)
- [Symmetries](../notebooks/appendix/01_symmetries.ipynb)
- [Legacy tutorials](legacy.md)
