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
Eqs. (19), (49), (53), (56)-(59), and (61)-(65). Equation (49) fixes the free
molecular resolvent used when eliminating non-diagonal density-matrix elements.

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

Its Supplemental Material, Sec. I.B, gives a related contour-integration
treatment of the fourth-order RTD energy integrals: a flat reservoir band with a
hard cutoff, one integral closed analytically by the residue theorem, and the
remaining tanh-pole sum carried out numerically rather than by an Ozaki
approximation. Temperatures are per reservoir throughout, and the O(Gamma)
effective Liouvillian there carries the explicit `ln(D / 2 pi T)` term that the
wide-band forms drop. It builds on the O(Gamma^2) expressions in the
Supplemental Material of N. M. Gergs, C. B. M. Horig, M. R. Wegewijs, and
D. Schuricht, *Physical Review B* **91**, 201107(R) (2015), whose own
Supplemental Material, Sec. I B 3, states that its second-order energy
integrations are done analytically *for* `T_L = T_R = T` while noting that this
"presents no principal limitation of our method". Neither paper is itself a
closed-form unequal-temperature derivation: the 2015 main text fixes
`T_L = T_R` and the 2018 setting is a spin valve with polarized reservoirs.
What both supply is the finite-cutoff contour route that does not require
equal temperatures.

### `KirsanskasFranckieWacker2018`

G. Kiršanskas, M. Franckié, and A. Wacker, “Phenomenological position and energy
resolving Lindblad approach to quantum kinetics,” *Physical Review B* **97**,
035432 (2018).
[DOI](https://doi.org/10.1103/PhysRevB.97.035432) ·
[arXiv](https://arxiv.org/abs/1710.02762)

The construction QmeQ's Lindblad approach implements, cited as Ref. [32] of the
QmeQ paper while still in preparation. Each tunneling matrix element is dressed
with the square root of an occupation factor and one jump operator is kept per
lead rather than one per Bohr frequency, so the generator is of GKLS form
without a secular approximation and retains the nonsecular terms a Redfield
kernel has.

### `NathanRudner2020`

F. Nathan and M. S. Rudner, “Universal Lindblad equation for open quantum
systems,” *Physical Review B* **102**, 115109 (2020).
[DOI](https://doi.org/10.1103/PhysRevB.102.115109) ·
[arXiv](https://arxiv.org/abs/2004.01469)

Derives the same square-root construction as a controlled weak-coupling
approximation with an error bound, and supplies the Hermitian Lamb shift that
belongs to it. Eq. (2) gives the jump operator; Appendix D, Eqs. (D7)-(D8), give
the shift. Note that the main text's restatement of the shift, Eq. (34), carries
the second energy argument as `E_n - E_l` where Eq. (D8) has `E_l - E_n`. Only
the appendix form is Hermitian, as the paper requires below its Eq. (D1), so
`func_ule_shift` follows Eq. (D8).

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
