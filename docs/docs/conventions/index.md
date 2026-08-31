# Conventions

Internal rules the code relies on. Each page carries the evidence for its
claims, because most of these were recovered by reading and measuring rather
than from any existing document.

## Pages

- **[Density-matrix layout](density-matrix-layout.md)** — how a Hermitian
  density matrix becomes the real vector QmeQ actually solves for, why it is
  real rather than complex, and the sign convention built into kernel
  insertion.
- **[State indexing](state-indexing.md)** — what `maptype` means, which
  selectors exist on which indexing class, and how unsupported values fail.
- **[Typing](typing.md)** — the opportunistic type-hint policy, and why some
  signatures carry a union rather than the obvious class.
- **[Docstrings](docstrings.md)** — portable source-docstring conventions and
  how they feed the generated API reference.
- **[RTD kernel matrices](rtd-kernels.md)** — the `mi` selector, why `Lnn` is
  not a Liouvillian, and where the two backends diverge.
- **[Where the time goes](where-the-time-goes.md)** — measured density and
  profiles. The kernels are dense and assembly dominates; this is not a sparse
  linear-algebra problem.

## Cross-cutting facts

A few things worth knowing before reading any of the above.

**There are two density-matrix representations, chosen per approach.** Pauli,
Lindblad, Redfield, 1vN and RTD reduce by Hermiticity and solve in a packed
*real* vector of length `si.ndm0r`. The 2vN approaches keep both orientations
of every element and solve in a *complex* vector of length `si.ndm0`
(`ApproachBase2vN` sets `dtype = complexnp`). This is a deliberate split, not
historical residue; see
[why the layout is real](density-matrix-layout.md#why-real-and-not-complex).

**A specification module exists.** `qmeq.approach.dm_layout` states the packed
real layout as nine numbered rules (L1–L9) and provides a reference
implementation. `qmeq/tests/test_dm_layout.py` pins each rule, with every test
naming the rule it covers. Prefer citing a rule to re-deriving it.
