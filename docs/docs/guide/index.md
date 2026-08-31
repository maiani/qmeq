# Guide

User-and-API-facing material: what QmeQ computes, what its approaches are and
where they can be trusted, and how to actually build and solve a system. The
[Conventions](../conventions/index.md) section next to this one is for people
changing QmeQ's internals instead; this one is for people using QmeQ.

## Pages

- **[What QmeQ computes](overview.md)** — the physical problem (an
  Anderson-type quantum dot coupled to leads), what an "approach" is, and the
  package's own physics disclaimer.
- **[The approaches](approaches.md)** — Pauli, Lindblad, Redfield, 1vN, 2vN,
  RTD, and RTDnoise: what each approximates, what it solves for, its validity
  domain, and its known failure modes.
- **[Getting started](getting-started.md)** — constructing a system with
  `Builder`, choosing `kerntype`, electron-phonon inputs, indexing and spin
  symmetry, editing a model, and reading off results after `solve()`.

The [API reference](../api/index.md) provides complete signatures and
attribute documentation; the guide focuses on choices users must make and on
safe workflows.
