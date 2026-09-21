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
[arXiv](https://arxiv.org/abs/1711.03460)

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

Derives the square-root jump construction and its Hermitian shift. Use Eq. (D7)
with the corrected Eq. (D8) from `NathanRudner2021Erratum`, NOT either original
restatement. For outer states b,b' and intermediate k the arguments are
`p=E_k-E_b`, `q=E_b'-E_k`. A lower intermediate emits into empty lead states;
an upper intermediate absorbs occupied lead states. See `theory/lambshift.md`
and the direct principal-value tests. Hermiticity alone does not distinguish
an expression with both energy arguments reversed.

### `NathanRudner2021Erratum`

F. Nathan and M. S. Rudner, “Erratum: Universal Lindblad equation for open
quantum systems [Phys. Rev. B 102, 115109 (2020)],” *Physical Review B* **104**,
119901 (2021). [DOI](https://doi.org/10.1103/PhysRevB.104.119901).

Equations (1)-(2) correct the main-text indices and sign; Eq. (5) corrects
Appendix D4; Eq. (6) corrects D8. D7 is unchanged. The corrected D8 agrees
with D3 and with second-order energy denominators. Merely choosing the
original appendix over the main text does not resolve the errors.

## Analytic benchmarks

### `StoofNazarov1996`

T. H. Stoof and Yu. V. Nazarov, “Time-dependent resonant tunneling via two
discrete states,” *Physical Review B* **53**, 1050 (1996).
[DOI](https://doi.org/10.1103/PhysRevB.53.1050) ·
[arXiv](https://arxiv.org/abs/cond-mat/9506036)

Equation (8) is the stationary current through a serial double dot at large
bias, `I = T^2 Gamma_R / (T^2 (2 + Gamma_R/Gamma_L) + Gamma_R^2/4 + eps_0^2)`,
with `T` the interdot hopping and `eps_0` the detuning between the two levels.
Used as the analytic oracle of the large-bias check in
`examples/tutorials/04_coherence_and_approximations.ipynb`, which reproduces it
to machine precision at asymmetric couplings. This is the spinless form; for
the spin-resolved one see `GurvitzPrager1996`.

### `GurvitzPrager1996`

S. A. Gurvitz and Ya. S. Prager, “Microscopic derivation of rate equations for
quantum transport,” *Physical Review B* **53**, 15932 (1996).
[DOI](https://doi.org/10.1103/PhysRevB.53.15932) ·
[arXiv](https://arxiv.org/abs/cond-mat/9511026)

The same double-dot rate equations with spin components accounted for. Its
Eq. (46) carries `Gamma_R/(2 Gamma_L)` where `StoofNazarov1996` Eq. (8) carries
`Gamma_R/Gamma_L`, a difference that paper attributes to the spin sum. At
symmetric coupling the two differ as `2.5 Omega^2` against `3 Omega^2` in the
denominator. Recorded to fix which of the two QmeQ's spinless two-orbital
tutorial model reproduces; it is the Stoof-Nazarov form.

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
