# QmeQ roadmap

Open work only. Completed work is recorded in [CHANGELOG.md](CHANGELOG.md)
under `[Unreleased]`; this file is deliberately not a history.

Ground rules for anything below:

- This is a scientific library: **physics correctness comes before
  convenience**, and different approximations are expected to disagree with
  each other. A change that is easy to verify against the existing tests and
  reference data beats a cleverer one that is not.
- Every numerical change must be exercised in both the pure-Python and Cython
  implementations where both exist, and tolerances must be justified rather
  than widened until they pass.
- Public compatibility aliases and accepted input forms stay unless a
  deliberate breaking release says otherwise.

## P0: results a user can trust

The library's own disclaimer says these approximations can fail. Right now they
fail *silently*, which is the most expensive kind of bug in a package whose
output ends up in papers.

- [ ] Flag unphysical stationary solutions instead of returning them.
  - Redfield, 1vN, 2vN, and RTD can all violate positivity of the reduced
    density matrix. Measured `min(phi0) = -0.81` with `sum(phi0) == 1` and a
    current that looks unremarkable (`bug_report.md` issue 5).
  - Check negative populations, trace deviation, and solver conditioning; emit
    a warning *and* expose the result as a queryable diagnostic, so scripted
    sweeps can filter on it rather than parsing stderr.
  - Keep the behaviour identical between backends and approaches.
  - Cheapest available guard against publishing a wrong number; do this first.

- [ ] Warn when RTD is used outside the regime where a diagonal density matrix
      is justified.
  - `generate_row_inverse_Liouvillian` inverts a bare `1/(E_a - E_b)` clamped
    at `minE = 1e-10` and carrying no broadening. Eliminating the coherences is
    only valid when intra-sector splittings are large compared with the tunnel
    broadening; a DQD tuned near orbital degeneracy reached `0.07 Γ` with
    nothing in the output saying so (`bug_report.md` issue 4).
  - Needs a defensible Γ estimate and threshold, applied in both `RTD.py` and
    `c_RTD.pyx`. Choosing the threshold is physics, not a lint fix.

- [ ] Collect each approach's validity domain and known failure modes in one
      documented place.
  - The material exists but is scattered across tutorial 6's validity table,
    the `qmeq/__init__.py` disclaimer, and the RTD bandwidth warnings. Promote
    it into `docs/source/theory/` as one reference keyed by approach, and point
    the warnings at it.

## P1: correctness gaps in shipped features

- [ ] Make `BuilderManyBodyElPh` work, or declare it unsupported.
  - It constructs, but `solve(qdq=False, rotateq=False)` raises `IndexError` in
    `get_ind_dm0`: `si_elph` is never set up for many-body input. Reproduced on
    both backends, and unchanged by the recent initialisation-order fix, so this
    is longstanding rather than a regression.
  - Either fix the indexing path and add a regression test, or raise a clear
    `NotImplementedError` at construction instead of failing deep inside a
    solve.

- [ ] Resolve the remaining electron-phonon backend parity failure.
  - The QmeQ 1.1 reference suite now compares kernels, stationary states, and
    currents for electron-phonon Pauli, Lindblad, Redfield, and 1vN within the
    same approximation. Pauli, Redfield, and 1vN pass on both backends.
  - Compiled electron-phonon Lindblad remains a strict, conditional `xfail`
    with an inline reason. Diagnose and fix that numerical divergence without
    weakening tolerances or normalizing it into the historical fixture.

- [ ] Support the RTD energy and heat currents for complex tunnel amplitudes.
  - Both are currently filled with `nan` and a warning while the charge current
    is computed. That is a sharp edge for any model with flux or interference.

- [ ] Reconcile `RTDnoise` with the default RTD kernel.
  - `RTDnoise` refuses to run unless `off_diag_corrections=False`, so its noise
    comes from a kernel that differs from the one RTD uses by default. Either
    implement the corrections there or quantify and document the discrepancy.

- [ ] Turn the unequal-temperature RTD cutoff warning into an answer.
  - Thermal-bias results depend on `dband` at percent-to-tens-of-percent level
    and the user is simply told to rerun with larger values. A helper that
    sweeps `dband` and reports observable-level convergence would make the
    documented requirement actually followable.

- [ ] Expand numerical edge-case coverage.
  - Zero and extreme temperatures, narrow and wide bands, nearly degenerate
    states, complex amplitudes, and empty or removed state sectors; plus the
    limiting behaviour of the special functions and integration cutoffs.

## P1: distribution and support contract

- [ ] Decide how users are meant to install this fork.
  - Nothing publishes to PyPI, yet `INSTALL.md` tells users `pip install qmeq`,
    which resolves to the upstream project rather than this one. Either publish
    under a name you own (Trusted Publishing, no token) or point the
    instructions at the release assets or a git URL.

- [ ] Make wheels and source distributions self-consistent and verified.
  - Tighten `MANIFEST.in` so the sdist carries docs, tests, Cython sources, and
    examples without `docs/build` or other generated artifacts; keep examples
    out of the installed wheel.
  - Build both artifacts, inspect their file lists and sizes, run
    `twine check`, then install each into a clean environment and run import,
    metadata, and fast-test checks against the installed copy.
  - `build_wheels.yml`'s smoke test never asserts
    `get_backend_status()['active'] == 'cython'`, so a wheel that silently fell
    back to pure Python would ship unnoticed.
  - A focused CI job now installs both artifacts outside the source tree,
    asserts both requested backends, and runs the shared external-reference
    checks. Extend that gate to inspect the complete artifact inventory and run
    broader import, metadata, and fast-suite checks.

- [ ] Test the dependency floors, not just the current releases.
  - NumPy, SciPy, Cython, and the build backend are unpinned and only ever
    exercised at their newest versions. Add a lowest-supported job so the
    declared range means something, and avoid speculative upper bounds unless a
    demonstrated incompatibility justifies one.

- [ ] Verify metadata consistency before publishing.
  - Project URLs, supported versions, authorship, citation text, and the physics
    disclaimer should agree across package metadata, `README.md`, `INSTALL.md`,
    `AUTHORS.md`, and the documentation.

## P2: performance

- [ ] Establish a benchmark baseline.
  - Nothing is measured: not whether OpenMP threading helps, not where RTD and
    2vN time actually goes, not whether the 2vN Hilbert-transform chunking paid
    off. Without a baseline every "optimisation" is unfalsifiable — and the
    macOS wheels were just made serial without being able to quantify what that
    costs a user.
  - A handful of representative systems, timed reproducibly, is enough to start.

## P2: documentation and contributor workflow

- [ ] Simplify and refresh the Sphinx documentation.
  - Remove duplicate toctree entries while keeping both the pure-Python and
    Cython API reachable, and keep notebook rendering non-executing (execution
    belongs to the example test jobs).

- [ ] Document the supported development workflow.
  - Editable installs, backend and OpenMP selection, regenerating Cython
    output, running the fast and slow suites, building the docs, validating
    artifacts — and which files must change together when a `.py`/`.pyx` pair is
    touched.

## Release gate

A release is ready only when all of the following hold:

- [ ] The fast pure-Python and compiled suites pass across the CI matrix.
- [ ] The `--runslow` example suite passes (`slow.yml`).
- [ ] The documentation builds from a clean checkout with warnings as errors.
- [ ] Wheel and sdist contents have been inspected, and both artifacts tested
      after installation into clean environments.
- [ ] `CHANGELOG.md` has one coherent `[Unreleased]` section covering every
      user-visible change.
- [ ] Package, documentation, and tag versions agree.
- [ ] No P0 item above is open, or each open one is a documented, accepted
      limitation rather than a silent one.
- [ ] Release artifacts come from the tested revision and are published only
      after those checks succeed.
