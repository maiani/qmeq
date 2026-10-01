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
- consolidation of RTD code paths that already exist (group B);
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
| A helper that sweeps `dband` for thermal-bias RTD | A documented convergence recipe (D4) |
| A 2vN `kpnt` convergence check | The documented `kpnt`/`niter` requirement, with a measured example (D4) |
| A public non-interacting reference solver | The NEGF solver stays test-only, in `qmeq/tests/noninteracting_negf_solver.py` |
| Higher cumulants, energy-current noise, finite-frequency noise | The first two zero-frequency particle-current cumulants |

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
- **Repository.** 1.2 is released from `maiani/qmeq`. After the handoff the
  repository moves to a GitHub organisation. A transfer redirects repository
  links, but not GitHub Pages URLs (E3).
- **Conda.** 1.2 ships on the prefix.dev channel `andmai/science`. A
  conda-forge feedstock, built from the PyPI sdist, follows after 1.2.0
  (section 6).
- **Still open: the next maintainer.** Name them before E5, so that
  `AUTHORS.md`, `README.md` and the publishing configuration are written once.

## 4. Open work

### B. One RTD diagram traversal, and a compiled RTDnoise

`qmeq.approach.rtd_diagrams` enumerates the RTD population diagrams as
immutable records, and `pyRTD`, `pyRTDnoise` and `RTDnoise` evaluate them.
The compiled `c_RTD.pyx` keeps its hand-written loops, held to the records by
`test_compiled_rtd_matches_the_record_based_python_rtd`.

- **B2. Decide whether to compile a record evaluator.** Measured on the record
  path for a spinful double dot (16 states, four channels):

  | approach | solve |
  |---|---|
  | `RTD`, compiled loops | 0.007 s |
  | `pyRTD` | 0.33 s |
  | `RTDnoise`, compiled scalar integrals | 0.76 s |
  | `pyRTDnoise` | 3.0 s |

  Of the 0.76 s, generating the records takes 0.12 s, and the compiled
  counting integrals take 0.29 s when called from Python: three Ozaki
  evaluations per diagram, for the value and a centred derivative. The
  remaining 0.35 s is per-record evaluation, scalar insertion into
  `Lpm_second`, the first-order and coherence blocks, and the noise solve.
  - A compiled evaluator of a lowered record table removes most of the last
    part and the call overhead. Its ceiling is about 2x, and it adds a
    compiled path to maintain.
  - The integrals bound any faster path. Compiling the enumeration as well
    would remove the generation time too, but only by giving the compiled
    side its own topology rules.
  - If a compiled evaluator is built: lower the records to typed arrays and
    test the round trip; write one evaluator, parameterised by its output,
    with no topology rules of its own; route `RTDnoise` through it once
    per-record, per-transfer, per-order, kernel, stationary-state, current and
    noise parity pass in fresh forced-backend processes, keeping
    `pyRTDnoise` all Python; and check the serial and OpenMP builds and the
    installed wheel and sdist. Compiled `RTD` stays on its own loops, since
    generating the records in Python already costs more than its whole solve.
  - If it is not built, record these measurements in
    `docs/docs/conventions/where-the-time-goes.md` and close the item.

### C. Tests

- **C1. Pin the numerical edge cases that no test covers.** The cases that
  were verified only by hand are:
  - exact and near degeneracies;
  - complex amplitudes, on every approach;
  - `remove_states`;
  - empty spin sectors;
  - very hot and very cold leads; and
  - the special functions at their limits.

  Some now have tests: the lead-temperature limits in `test_numerical_edges.py`
  and `remove_states` on many-body input. Map the suite first and add only the
  uncovered cases. Each new test asserts an invariant or an independent value,
  such as rephasing covariance for complex amplitudes or a limiting form for a
  special function. It never asserts the code's current output.

### D. Documentation

- **D1. Correct statements that contradict the code.**
  - `docs/docs/theory/counting-statistics.md` calls `kerntype='RTDnoise'` an
    alias of the pure-Python implementation. In fact `ApproachRTDnoise`
    selects compiled scalar integrals on the Cython backend, as
    `approaches.md` says.
  - Tutorial 7's validity table says that RTD off-diagonal counting
    corrections are not implemented. `off_diag_corrections=True` is supported,
    and it is the default.
  - `CHANGELOG.md` claims Conda packages for Intel macOS, but `release.yml`
    builds only `linux-64`, `linux-aarch64` and `osx-arm64`. Either build
    `osx-64` or correct the claim.
  - The URLs in `pyproject.toml` point at `gedaskir/qmeq`; point them at
    `maiani/qmeq`.
  - The changelog compare links resolve only in `gedaskir/qmeq`, because
    `maiani/qmeq` has no `1.1` tag. Push that tag to the fork at the upstream
    commit, or keep the 1.1 links pointing at upstream.
- **D2. Write the RTD validation envelope into the permanent documentation.**
  Put it on the counting-statistics page and under RTD/RTDnoise in the
  approaches guide, citing test names. It should state:
  - what is graded at `U = 0`, against the exact NEGF solver: with
    `off_diag_corrections=True`, the current and noise residuals are cubic in
    the coupling, and without the correction they are quadratic; this holds
    for real amplitudes and for generic plaquette flux; the observables are
    invariant under orbital rephasing and `2π`-periodic in the flux;
  - what is graded at `U ≠ 0`: in a deep-blockade Anderson dot, the
    elastic-cotunnelling current within 0.02% and the bidirectional-Poisson
    noise within 1%;
  - what is not graded: interacting systems outside deep blockade, splittings
    `≲ Γ` (where the elimination is invalid by construction), and the energy
    current at complex amplitudes.
- **D4. Convergence recipes in place of the dropped helpers.** In the
  approaches guide:
  - a short `dband` sweep for thermal-bias RTD and RTDnoise, including what
    "converged" means for the current and for each noise entry; and
  - the measured 2vN example: at `dband=10` and `niter=3` the current moves
    from `3.90e-05` at `kpnt=2**9` to `1.71e-05` at `kpnt=2**5`.

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
- **E3. Publish the built documentation** from CI on a tag. A GitHub Pages
  site under `maiani` stops resolving when the repository moves to the
  organisation. Either host the site at its final address from the start, or
  plan the link update as a 1.2.x change. Point the `README.md` and
  `pyproject.toml` documentation links at the published site. Notebook
  execution stays in the example test jobs, not in the documentation build.
- **E4. Write the development and release guide** as `CONTRIBUTING.md`. It
  covers:
  - editable installs, and backend and OpenMP selection;
  - regenerating the Cython output;
  - the fast and slow suites, and the documentation build;
  - artifact validation;
  - the files that must change together when a `.py`/`.pyx` pair, or the RTD
    diagram enumeration and its compiled twin in `c_RTD.pyx`, is touched;
  - the reference-data policy; and
  - the release procedure: version bump, tag, what each workflow publishes
    where, the trusted publishers, and who receives the scheduled `slow.yml`
    failures.

  `AGENTS.md` keeps only the rules specific to agents, and links to the guide
  instead of repeating it.
- **E5. Hand off.**
  - Add a maintenance-status paragraph to `README.md`, and the next maintainer
    to `AUTHORS.md`.
  - Give the next maintainer administration of the repository, ownership of
    the PyPI project, the Conda channel and the documentation hosting.
  - Check that the next maintainer can cut a release without the current
    maintainer's accounts.

## 5. Release gate

1.2.0 is ready only when all of the following hold:

- [ ] Every item in section 4 is closed, or is a documented limitation listed
      in section 6.
- [ ] A release candidate, `v1.2.0rc1`, has passed through every publishing
      path: the GitHub release with wheels and sdist, PyPI, the Conda channel
      and the documentation site. Pre-release tags build only `linux-64` for
      Conda, so run `release.yml` with `full-matrix: true`.
- [ ] The fast pure-Python and compiled suites pass across the CI matrix.
- [ ] The `--runslow` example and notebook suites pass (`slow.yml`).
- [ ] The documentation builds strictly from a clean checkout.
- [ ] The wheel and sdist contents have been inspected, and both artifacts have
      been installed outside the source tree. The installed-copy tests pass on
      both forced backends, and `qmeq.get_backend_status()` reports the
      expected implementation.
- [ ] `[Unreleased]` in `CHANGELOG.md` is one coherent `[1.2.0]` section, and
      its upgrade notes from 1.1 are complete.
- [ ] The package, documentation and tag versions agree.
- [ ] The release artifacts come from the tested revision, and are published
      only after these checks pass.

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
- Counting statistics cover the first two zero-frequency particle-current
  cumulants. There is no counting for 2vN, the electron-phonon approaches or
  matrix-free solving.
