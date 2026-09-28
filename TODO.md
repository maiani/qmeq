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

- [x] Make `BuilderManyBodyElPh` work, or declare it unsupported.
  - Done: `si_elph` is rebuilt from the many-body states, and
    `test_many_body_elph_input_matches_fock_input` requires the builder to
    reproduce `BuilderElPh` exactly for all four approaches on both backends.

- [x] Resolve the remaining electron-phonon backend parity failure.
  - Done: the compiled jump term used `conj(L[bp, a])` where
    `conj(L[bp, ap])` belongs, which broke trace preservation for coherence
    columns; with that one index fixed the compiled kernel equals the Python
    twin and the QmeQ 1.1 reference to machine precision, and the `xfail` is
    gone. `test_elph_backend_parity.py` now gates kernel-level parity and trace
    preservation for all four electron-phonon approaches.
  - The compiled `generate_fct` now also fills `tLbbp` for every same-charge
    pair, as the Python one does, so the backends agree for `'ssq'` with
    spin-dependent `velph` too.

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

- [x] Warn when the band edge silences the current.
  - Done: `check_band_coverage` in `qmeq/approach/diagnostics.py` runs from the
    shared `solve` on both backends and warns once, per lead, when a finite
    band excludes every transition that lead couples to.

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

- [ ] Explain the 2vN equilibrium current at finite interaction.
  - With the conjugation fix 2vN is gauge covariant, but at `mu_L = mu_R`
    and equal temperatures it still carries a small nonzero current: about
    2% of the biased current at `U = 2` on a spinless double dot, ~5e-8 at
    `U = 0`, independent of `dband` and `kpnt`. Halving the tunnelling
    amplitude cuts the ratio from 1.4e-2 to 7.5e-4, 2e-5 and 1.7e-6, so it is
    high order in the coupling, which fits a limitation of the 2vN truncation
    but is not shown to be one. Establish which, and document it in the
    approaches guide if it is.

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
