# Implementation references

This file assigns stable keys to external literature used to derive or explain
QmeQ's implementation. Source comments and docstrings should cite a key and,
where possible, the relevant equation or section. For example:

```python
# [Emary2009, Eqs. (40)-(41)] Projected-pseudoinverse cumulants.
```

The bibliographic entry for citing QmeQ itself remains in the
[README](README.md#citing-qmeq).

## Transport kernels and counting statistics

### `LeijnseWegewijs2008`

M. Leijnse and M. R. Wegewijs, “Kinetic Equations for Transport Through
Single-Molecule Transistors,” *Physical Review B* **78**, 235424 (2008).
[DOI](https://doi.org/10.1103/PhysRevB.78.235424) ·
[arXiv](https://arxiv.org/abs/0807.4027)

Used for the real-time diagrammatic notation and rules, the leading-order
kernel, and the fourth-order direct and exchange diagrams; see especially
Eqs. (48), (53), (56)-(59), and (61)-(65).

### `Emary2009`

C. Emary, “Counting Statistics of Cotunneling Electrons,” *Physical Review B*
**80**, 235306 (2009).
[DOI](https://doi.org/10.1103/PhysRevB.80.235306) ·
[arXiv](https://arxiv.org/abs/0902.3544)

Used for zero-frequency current cumulants from the counting-field kernel; see
especially Eqs. (40)-(41).

### `GergsEtAl2018`

N. M. Gergs, S. A. Bender, R. A. Duine, and D. Schuricht, “Spin Switching via
Quantum Dot Spin Valves,” *Physical Review Letters* **120**, 017701 (2018).
[DOI](https://doi.org/10.1103/PhysRevLett.120.017701)

Its Supplemental Material gives a related contour-integration treatment of
the fourth-order RTD energy integrals.

## Special functions

### `Ozaki2007`

T. Ozaki, “Continued Fraction Representation of the Fermi-Dirac Function for
Large-Scale Electronic Structure Calculations,” *Physical Review B* **75**,
035123 (2007).
[DOI](https://doi.org/10.1103/PhysRevB.75.035123)

Used for the continued-fraction pole expansion of the Fermi function.

### `KarraschMedenSchoenhammer2010`

C. Karrasch, V. Meden, and K. Schönhammer, “Finite-Temperature Linear
Conductance from the Matsubara Green's Function without Analytic Continuation
to the Real Axis,” *Physical Review B* **82**, 125114 (2010).
[DOI](https://doi.org/10.1103/PhysRevB.82.125114) ·
[arXiv](https://arxiv.org/abs/1007.3403)

Used for the simplified derivation and convergence discussion of the Ozaki
pole expansion.
