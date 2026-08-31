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
  - The unfinished derivation is visible as the commented-out `gamma.imag`
    terms at `RTD.py:543,544,563,564,613,614,632,633`. Validate the energy and
    heat channels separately from the particle current.

- [ ] Turn the unequal-temperature RTD cutoff warning into an answer.
  - Thermal-bias results depend on `dband` at percent-to-tens-of-percent level
    and the user is simply told to rerun with larger values. A helper that
    sweeps `dband` and reports observable-level convergence would make the
    documented requirement actually followable.
  - The U=0 reference solver has no bandwidth cutoff and no equal-temperature
    restriction, so it can supply the converged answer the sweep should approach,
    at least in the non-interacting limit.

- [ ] Warn when the band edge silences the current.
  - With `itype=0`, moving the cutoff by `1e-9` across a transition energy is
    the difference between `current=1.66e-05` and a silent `current=0`. Warn
    once when the band excludes every transition energy: that is the whole
    class, not just the case found by hand.

- [ ] Say when `dband` is being ignored.
  - `itype=1` and `itype=3` are wide-band limits and drop the cutoff entirely:
    the current is identical to six digits from `dband=1e5` down to
    `dband=0.01`, a band far narrower than both the bias window and the level
    energies. Nothing warns. RTD forces `itype=1` and warns about the override,
    but not about the parameter it then discards.

- [ ] Cover the remaining numerical edge cases in tests.
  - Verified by hand and currently untested: exact and near degeneracies,
    complex amplitudes on every approach, `remove_states`, empty spin sectors,
    very hot and very cold leads, and the special functions at their limits.
    All behave; the point is that nothing pins them.
  - 2vN grid convergence is the open one: at `dband=10` and `niter=3` the
    current moves from `3.90e-05` at `kpnt=2**9` to `1.71e-05` at `kpnt=2**5`
    with only a generic warning. A convergence check would make `kpnt`
    followable in the way the RTD `dband` sweep above would make bandwidth
    followable.

## P1: distribution and support contract

- [ ] Cover both the pip and the Conda installation paths.
  - Conda already ships: `release.yml` builds, tests, and uploads the recipe to
    the `andmai/science` prefix.dev channel on a tag. PyPI does not, yet
    `INSTALL.md` tells users `pip install qmeq` and links its source download
    at `gedaskir/qmeq`, both of which resolve to the upstream project rather
    than this fork. Publish under a name you own (Trusted
    Publishing, no token), then have `INSTALL.md` name both paths instead of
    mentioning Conda only as the OpenMP-enabled alternative for macOS.

## P2: documentation
- [ ] Publish the built documentation.
  - Nothing deploys `docs/site/`, so reading the manual means building it
    locally or reading the Markdown in the repository, and `README.md` can only
    point at the `docs/` directory. Notebook execution stays in the example
    test jobs rather than in the documentation build either way.

- [ ] Document the supported development workflow.
  - Editable installs, backend and OpenMP selection, regenerating Cython
    output, running the fast and slow suites, building the docs, validating
    artifacts — and which files must change together when a `.py`/`.pyx` pair is
    touched.

- [x] Collect each approach's validity domain and known failure modes in one
      documented place.
  - Done: consolidated into
    [docs/docs/guide/approaches.md](docs/docs/guide/approaches.md), covering
    Pauli, Lindblad, Redfield, 1vN, 2vN, RTD, and RTDnoise — pulled from
    tutorial 6's validity table, the `qmeq/__init__.py` disclaimer, and the
    RTD bandwidth/coherence/no-broadening warnings in
    `qmeq/approach/base/RTD.py`, with every claim marked Verified, Stated, or
    Open per the site's evidence discipline.
  - Still open: the RTD warning *messages* in `RTD.py` do not yet point
    readers at the page (a code change, out of scope for a documentation
    pass) — and this line item's own completion still needs a `CHANGELOG.md`
    entry, which is left for the next edit to that file.

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
