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
  `Builder`, choosing `kerntype`, and reading off the result attributes after
  `solve()`.

## Status

**Started, not comprehensive.** These three pages do not yet cover everything
the legacy Sphinx manual under [`legacy_docs/`](../../../legacy_docs/) does —
notably the full `BuilderElPh`/electron-phonon path, the indexing options
(`Lin`/`sz`/`ssq`) in depth, and the counting-statistics API beyond a
pointer. Where this guide is silent, `legacy_docs/` and the docstrings in
`qmeq/` are the reference.
