# QmeQ 1.2 development plan

This file owns QmeQ's open work: what remains before 1.2.0, what has been
dropped, and what maintenance after 1.2 covers. Changes already made are in
[CHANGELOG.md](CHANGELOG.md) under `[Unreleased]`. This file is not a history:
a finished item is deleted, not ticked.

It is a coordination document. Production code, tests, fixtures and the
documentation tree state their conventions and provenance directly and never
link here or copy its item labels.

## 1. Scope

QmeQ 1.2 is the last release under the current maintainer. After it the
project passes to a new maintainer for maintenance. QmeQ takes no new features.

The 1.2 feature set is the one in 1.2.0.dev11. The remaining work finishes and
polishes that set:

- correctness fixes in shipped features;
- consolidation of RTD code paths that already exist;
- diagnostics that make a silent limitation visible;
- tests that pin behaviour so far verified only by hand;
- documentation of what each approach computes, where it fails, and how to
  check convergence; and
- packaging, publishing and the handoff.

**Rule for borderline items.** If finishing an item needs a new derivation or a
new public function, it is out of scope. Document the limitation instead.
Consolidating paths that already exist is in scope, provided no result moves
beyond the numerical floor and the path being replaced serves as the gate.

## 2. Dropped

| Item | What 1.2 ships instead |
|---|---|
| Full-coherence RTD: `L0+W1` and `L0+W1+W2` on the complete `dm0`, with coherent counting | RTD and RTDnoise eliminate same-charge coherences. `RTDCoherenceWarning` and `rtd_coherence_diagnostics`, including `clamped_coherences`, flag where the elimination fails |
| RTD energy and heat currents for complex tunnel amplitudes | `nan` with a warning. The particle current is unaffected |
| A helper that sweeps `dband` for thermal-bias RTD | A convergence recipe in the approaches guide |
| A 2vN `kpnt` convergence check | The documented `kpnt`/`niter` requirement, with a measured example, in the approaches guide |
| A public non-interacting reference solver | The NEGF solver stays test-only, in `qmeq/tests/noninteracting_negf_solver.py` |
| Higher cumulants, energy-current noise, finite-frequency noise | The first two zero-frequency particle-current cumulants |
| A compiled RTDnoise traversal or record evaluator | A Python traversal over the shared diagram records, with compiled scalar integrals. A compiled evaluator would gain at most about 2x; see `docs/docs/conventions/where-the-time-goes.md` |

The groundwork that landed for full-coherence RTD **stays**. It specifies and
tests the packed layout that every shipped approach already uses, and none of
it depends on the dropped solver. It covers:

- `qmeq.approach.dm_layout`, with rules L1-L9, `LiouvilleState` and the
  reference `DensityMatrixLayout`;
- `NO_INDEX`, `QMEQ_STRICT_INDEX` and `RtdMatrix`;
- `get_ind_dm0_bool` and `get_ind_dm0_conj`; and
- the first-order population-coherence blocks in `qmeq.approach.rtd_blocks`,
  which RTD and RTDnoise share.

The near-degeneracy clamp in the coherence elimination also stays. It is
observable through `clamped_coherences`.

## 3. Distribution and hosting

- **Name.** The package stays `qmeq`. Ask the original author for publish
  rights to the existing PyPI project, so that 1.2 reaches every existing user
  as an ordinary upgrade (E1).
- **Repository.** The repository moves from `maiani/qmeq` to the `qmeq`
  organisation, as `qmeq/qmeq`, after the next push. Every link in the tree
  already points there, and 1.2 is released from it.
- **Python.** 1.2 requires Python 3.12 or newer, the oldest version the
  current NumPy and SciPy support.
- **Conda.** 1.2 ships on the prefix.dev channel `andmai/science`. A
  conda-forge feedstock, built from the PyPI sdist, follows after 1.2.0
  (section 6).
- **Maintainers.** Simon Wozny is a maintainer. Name any others before E5, so
  that the organisation owners, `AUTHORS.md`, `README.md` and the publishing
  configuration are written once.

## 4. Open work

### D. Documentation

- **D1. Push the release tags.** The `[Unreleased]` compare link needs a `1.1`
  tag in `qmeq/qmeq`. Upstream's annotated `1.0` and `1.1` tags are fetched
  into the local repository, on commits in this history; push them
  (`git push origin 1.0 1.1`).

### E. Distribution and handoff

- **E1. Publish to PyPI as `qmeq`.**
  - Once the original author grants publish rights, configure Trusted
    Publishing for the `build_wheels.yml` publish job, with no stored token.
    Make the next maintainer an owner of the project.
  - Rewrite the install instructions in `INSTALL.md` and `README.md`, and
    remove the warning that `pip install qmeq` installs 1.1.
  - Install from PyPI into a clean environment and check
    `qmeq.get_backend_status()`.
- **E2. Confirm the Conda channel.** The channel was last confirmed at
  1.2.0.dev9, and the dev10 upload predates the OIDC publishing change. Check
  whether dev11 arrived, and confirm that the release candidate reaches
  `andmai/science`.
- **E3. Publish the built documentation** from CI on a tag, as the GitHub
  Pages site of `qmeq/qmeq`. Point the documentation links in `README.md`,
  `pyproject.toml`, `recipe/recipe.yaml` and the docs pages at it. Notebook
  execution stays in the example test jobs, not in the documentation build.
- **E4. Complete the release section of `CONTRIBUTING.md`** once E1 and E3
  land: the PyPI and documentation publishing steps, the trusted publishers
  they use, and who receives the scheduled `slow.yml` failures.
- **E5. Hand off.**
  - Add a maintenance-status paragraph to `README.md`, and the next maintainer
    to `AUTHORS.md`.
  - Give the next maintainer administration of the repository, ownership of
    the PyPI project, the Conda channel and the documentation hosting.
  - Check that the next maintainer can cut a release without the current
    maintainer's accounts.

## 5. Release gate

1.2.0 is ready when the release checklist in `CONTRIBUTING.md` holds, and in
addition:

- [ ] every item in section 4 is closed, or is a documented limitation listed
      in section 6; and
- [ ] a release candidate, `v1.2.0rc1`, has passed through every publishing
      path: the GitHub release with wheels and sdist, PyPI, the Conda channel
      and the documentation site. Pre-release tags build only `linux-64` for
      Conda, so run `release.yml` with `full-matrix: true`.

## 6. Maintenance after 1.2

**In scope:**

- fixes to shipped features, each with a test that fails without the fix;
- compatibility with new Python, NumPy, SciPy and Cython releases;
- CI and packaging upkeep; and
- `1.2.x` patch releases through the same gate.

**Out of scope:** new approaches, observables or public API.

**Planned after 1.2.0:**

- a conda-forge feedstock built from the PyPI sdist; and
- moving the repository to a GitHub organisation, then updating every link
  that a transfer does not redirect.

**Known limitations that stay.** Each is documented in
`docs/docs/guide/approaches.md`.

- RTD and RTDnoise eliminate same-charge coherences, so they are invalid for
  splittings `≲ Γ`. The diagnostics flag this case.
- RTD energy and heat currents are `nan` for complex tunnel amplitudes.
- Thermal-bias RTD needs a `dband` convergence check.
- At finite interaction 2vN carries an equilibrium current of order
  `Gamma^3`.
- RTDnoise traverses its diagrams in Python; only its scalar integrals are
  compiled.
- Counting statistics cover the first two zero-frequency particle-current
  cumulants. There is no counting for 2vN, the electron-phonon approaches or
  matrix-free solving.
