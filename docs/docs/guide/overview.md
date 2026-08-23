# What QmeQ computes

## The physical problem

QmeQ computes the **stationary particle and energy currents** through a
quantum-dot device: one or more interacting single-particle levels ("the
dot"), tunnel-coupled to two or more macroscopic leads, each held at a fixed
temperature and chemical potential. This is the standard setup for quantum
transport experiments on semiconductor and molecular quantum dots. The
`qmeq/__init__.py` module docstring puts it directly: "QmeQ can calculate the
stationary state particle and energy currents using various approximate
density matrix approaches".

## The Anderson-type model

The dot is described by an **Anderson-type model**: a set of single-particle
orbitals with

- a single-particle Hamiltonian `hsingle` (on-site energies and coherent
  hopping between orbitals),
- a Coulomb interaction `coulomb` between pairs of orbitals (the "charging
  energy" that makes the problem interacting and gives rise to Coulomb
  blockade),

coupled to `nleads` fermionic reservoirs through single-particle tunneling
amplitudes `tleads`. Each lead is a reservoir in local equilibrium, set by a
chemical potential (`mulst`), a temperature (`tlst`), and, for approaches that
use a finite bandwidth, a cutoff (`dband`). (`construct_ham_coulomb`,
`qmeq/qdot.py`, and `construct_Tba`, `qmeq/leadstun.py`, build the many-body
Hamiltonian and tunneling matrix from exactly these dictionaries.)

QmeQ works in natural units $\hbar=k_\mathrm{B}=|e|=1$, so energies,
temperatures, and inverse times share units, and the reported particle
current and its noise have units of inverse time (see
`legacy_docs/source/theory/counting_statistics.rst`).

## What "approach" means

An **approach** (the `kerntype` argument to `Builder`) is the approximate
master-equation method used to obtain the dot's stationary reduced density
matrix, from which the currents follow. QmeQ implements several approaches
that differ in which terms of the exact reduced dynamics they keep, and
therefore in their validity domain, their computational cost, and their
failure modes. See [The approaches](approaches.md) for the full account —
this page only introduces the idea.

Every approach subclasses one of three base classes — `Approach`,
`ApproachElPh`, or `ApproachBase2vN` (`qmeq/approach/aprclass.py`) — and
exists in two forms where compiled extensions are built: a pure-Python
implementation and an optional Cython twin selected through `QMEQ_BACKEND`
(see the repository's `AGENTS.md`); both are meant to agree numerically.

## The package's own physics disclaimer

Quoted verbatim from the `qmeq/__init__.py` module docstring:

> All the methods in QmeQ are approximate so depending on parameter regime
> they can fail, and a good knowledge of the method is required whether to
> trust the result or not. For example, Redfield, 1vN, 2vN, and RTD approaches
> can violate positivity of the reduced density matrix and lead to currents
> flowing against the bias. We still think it is important to have a package
> where a user can duplicate existing calculations, check applicability of
> different methods, or simply discover new kind of physics using different
> approximate master equations.

The positivity warning above applies to Redfield, 1vN, 2vN, and RTD alike;
`qmeq/approach/diagnostics.py` names all four. Pauli, Lindblad, Redfield,
1vN, 2vN, RTD, and RTDnoise are all implemented, tested, importable, and
accepted `kerntype` values (`validate_kerntype` in
`qmeq/builder/validation.py`) — see [The approaches](approaches.md) for each
one's validity domain.
