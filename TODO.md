# QmeQ modernization roadmap

This roadmap covers maintenance and modernization, not new physics. Changes
must preserve the documented approximations, reference results, and public
compatibility aliases unless a deliberate breaking release says otherwise.
Every numerical change should be exercised in both the pure-Python and Cython
implementations when both exist.

## P0: trustworthy builds and continuous integration

- [x] Make **backend selection** explicit and testable.
  - Provide a supported way to force a pure-Python installation/runtime and a
    supported way to require compiled extensions.
  - Expose enough backend information for tests and bug reports to state which
    implementation is active.
  - Fail clearly when compiled extensions are explicitly requested but cannot
    be built or imported; do not silently fall back in that mode.
  - Add a small backend smoke test that cannot pass while exercising the wrong
    implementation.

- [ ] Replace the tag-only wheel workflow with pull-request and branch CI.
  - Run the fast test suite on every pull request and relevant branch push.
  - Test every Python version currently claimed in `pyproject.toml` on Linux,
    and representative supported versions on macOS and Windows.
  - Give pure-Python and compiled-extension tests separate jobs.
  - Build the documentation with
    `sphinx-build -b html -W --keep-going source build/html`.
  - Run the slow example suite on a scheduled or manually dispatched job so it
    remains a release gate without delaying every small pull request.
  - Pin maintained action releases and enable dependency caching only after the
    uncached jobs are reproducible.

- [ ] Make the extension build portable and predictable.
  - Replace unconditional OpenMP compiler/linker flags with per-platform
    configuration and a documented serial fallback.
  - Verify extension builds with the toolchains used by Linux, macOS, and
    Windows wheels.
  - Ensure a compiler or OpenMP failure cannot leave a partially importable
    installation.
  - Keep `.py`, `.pyx`, and `.pxd` behavior synchronized; add parity tests for
    shared kernels rather than relying only on build success.

- [ ] Standardize the compiled backend on Cython 3.
  - Declare a supported Cython 3 minimum and test both that version and the
    current release instead of leaving the build dependency unconstrained.
  - Set language level and other compiler directives explicitly so builds do
    not depend on changing Cython defaults.
  - Audit Cython 3 exception and `nogil` diagnostics; add `noexcept` only where
    failure is genuinely impossible and does not need to propagate.
  - Regenerate every extension from the canonical `.pyx`/`.pxd` sources in a
    clean build and run the compiled and backend-parity suites.

- [ ] Define one source-build policy and make the repository match it.
  - Treat `.pyx`/`.pxd` files as the canonical extension sources: generated
    `.c` files are currently ignored and untracked.
  - Remove the dead “reuse checked-in C files” path and the custom `--cython`
    switch from `setup.py`, or explicitly reverse this policy by tracking and
    validating all generated C files.
  - Keep build requirements, `INSTALL.md`, `clean.py`, package data, and source
    distribution contents consistent with the chosen policy.
  - Test an isolated PEP 517 build rather than only editable installs.

## P1: distribution and support contract

- [ ] Make wheels and source distributions self-consistent.
  - Tighten `MANIFEST.in` to include required documentation, tests, Cython
    sources, and vendored examples without including `docs/build` or other
    generated artifacts.
  - Keep examples out of the installed wheel unless they become a supported
    runtime resource, but include them in the source archive used for release
    validation.
  - Build wheel and sdist artifacts, inspect their file lists and sizes, and
    run `twine check`.
  - Install each artifact into a clean environment and run import, metadata,
    fast-test, and example smoke checks against the installed copy.

- [ ] Turn the declared Python range into a tested support policy.
  - Test Python 3.10 through every newer version advertised by classifiers;
    remove classifiers that cannot be exercised reliably.
  - Add a lowest-supported dependency job and a current-dependency job for
    NumPy, SciPy, Cython, and the build backend.
  - Document how and when old Python and dependency versions are retired.
  - Avoid speculative upper bounds; add them only for demonstrated
    incompatibilities with an issue and regression test.

- [ ] Decide the next release line before publishing.
  - Reconcile the Python `>=3.10` floor and other compatibility changes with
    semantic versioning; the next release may need to be a major release.
  - Single-source the package, documentation, and release version instead of
    keeping `docs/source/conf.py` fixed at `1.1`.
  - Verify that project URLs, supported versions, authorship, citation text,
    and the physics disclaimer are consistent across package metadata,
    `README.md`, `INSTALL.md`, and the documentation.

## P1: runtime and public API hygiene

- [ ] Remove mutable defaults without changing call semantics.
  - Cover `BuilderBase`, `BuilderManyBody`, `BuilderElPh`,
    `BuilderManyBodyElPh`, and helper functions such as `multiarray_sort`.
  - Replace dictionary/list defaults with `None` and create fresh values inside
    the function.
  - Add tests proving that two instances or calls cannot share mutated default
    state.
  - Preserve legacy builder aliases and accepted input forms.

- [ ] Replace warning-like `print` calls with structured warnings.
  - Introduce package warning classes or use appropriate standard warning
    categories for fallback, validation, convergence, and unsupported-feature
    notices.
  - Preserve intentional output helpers such as state-printing utilities.
  - Make warning behavior consistent between Python and Cython approaches,
    including RTD and 2vN.
  - Test warning category and message content where callers may need to filter
    or capture it.

- [ ] Make optional-extension imports precise.
  - Avoid broad `ImportError` handling that can hide a broken binary extension
    or an unrelated import failure.
  - Centralize backend discovery instead of repeating fallback logic in
    builders, approaches, indexing, and special functions.
  - Report the original load failure when compiled mode is required.
  - Keep importing QmeQ quiet during a normal, intentional pure-Python run.

- [ ] Clear deprecated and fragile dependency usage.
  - Run tests with deprecations visible under the oldest and newest supported
    NumPy, SciPy, and Cython versions.
  - Remove stale compatibility comments and APIs such as the old NumPy scalar
    aliases after verifying dtype behavior.
  - Address Cython 3 exception/no-GIL diagnostics deliberately; do not add
    `noexcept` without checking the physics kernels' error behavior.
  - Add focused tests before changing numerical types, integration behavior,
    tolerances, or solver defaults.

## P1: scientific regression coverage

- [ ] Build a backend-parity test layer.
  - Run the same small reference systems through Python and Cython Pauli,
    Lindblad, Redfield, 1vN, 2vN, RTD, and electron-phonon implementations
    where both are available.
  - Compare kernels, stationary states, particle currents, energy currents, and
    convergence status with method-specific tolerances.
  - Keep comparisons within the same approximation; disagreement between
    different physical approaches is expected and is not a regression.
  - Record tolerances with a numerical justification rather than weakening
    assertions only to accommodate a platform.

- [ ] Expand numerical edge-case tests.
  - Cover zero and extreme temperatures, narrow and wide bands, nearly
    degenerate states, complex tunneling amplitudes, and empty/removed state
    sectors where supported.
  - Add limiting-behavior tests for special functions and integration cutoffs.
  - Turn diagnostic `print` blocks in tests into useful assertion messages.
  - Separate genuinely slow physics regressions from fast unit tests with
    explicit markers and documented runtime expectations.

- [ ] Add lightweight quality gates for maintenance changes.
  - Add import/compile checks and a narrowly configured linter before adopting
    any repository-wide formatter.
  - Keep generated files, notebooks, and scientific data out of broad
    mechanical rewrites.
  - Add coverage reporting to identify untested maintenance paths, without
    treating a single percentage target as evidence of physics correctness.

## P2: documentation and contributor workflow

- [ ] Simplify and refresh the Sphinx documentation.
  - Remove duplicate toctree entries while preserving pure-Python and Cython
    API access.
  - Derive documentation version information from the package.
  - Verify a clean warnings-as-errors build both with extensions unavailable
    and, where useful, with compiled APIs importable.
  - Keep notebook rendering non-executing in Sphinx; execution belongs in the
    example test jobs.

- [ ] Document the supported development workflow.
  - Add concise contributor instructions for editable installs, selecting a
    backend, regenerating Cython outputs if applicable, running fast/slow
    tests, building docs, and validating artifacts.
  - Document which files must be changed together when a Python/Cython pair is
    modified.
  - Replace the destructive, repository-specific assumptions in `clean.py`
    with explicit, reviewable cleanup targets.

## Release gate

A release is ready only when all of the following are true:

- [ ] Fast pure-Python and compiled suites pass on the supported CI matrix.
- [ ] The scheduled/manual `--runslow` example suite passes.
- [ ] Documentation builds from a clean checkout with warnings as errors.
- [ ] Wheel and sdist contents have been inspected and both artifacts tested
      after installation in clean environments.
- [ ] `CHANGELOG.md` has one coherent `[Unreleased]` section and describes all
      user-visible changes.
- [ ] Package, documentation, and tag versions agree.
- [ ] Release artifacts are produced by the tested revision and published only
      after those checks succeed.

## Completed foundation

- [x] Merge the newer `cvsvensson/qmeq` maintenance history and reconcile it
      with the local modernization work.
- [x] Move static package metadata to `pyproject.toml`, adopt automatic package
      discovery, and declare Python `>=3.10`.
- [x] Make `README.md` canonical and refresh installation guidance.
- [x] Vendor examples and notebooks, render them in the documentation, and run
      fast and slow subsets through pytest.
- [x] Make the documentation build cleanly with warnings as errors when
      optional extensions are unavailable.
- [x] Restore pure-Python special-function fallbacks and replace removed SciPy
      constants.
- [x] Stabilize the Python and Cython Bose functions with `expm1` and
      large-positive-argument guards.
