# QmeQ Changelog

## [Unreleased]

### Added

- Add a source-based Conda recipe for compiled Python 3.11-3.14 Linux x86-64,
  Linux aarch64, Intel macOS, and Apple Silicon packages, plus a tag/manual
  GitHub Actions workflow that builds each Python/platform variant
  independently, runs the fast suite against the installed artifacts, and
  uploads them to a namespaced prefix.dev channel only after every build
  succeeds. Generated C build artifacts are no longer installed as package
  data; the canonical `.pyx` and `.pxd` extension sources remain included.
  The setup-configuration regression test now skips outside a source
  checkout so the installed suite remains runnable.
- Add a narrowly scoped Ruff correctness gate for Python files and notebooks,
  available through the development dependencies and enforced in CI.
  Repository-wide automatic formatting remains intentionally disabled while
  the historical source baseline is reviewed.
- Add an unequal-temperature RTD cutoff diagnostic. RTD now warns when the
  finite ``dband`` regulator is not conservatively separated from all
  transport scales, sizes the Ozaki expansion from the widest lead rather than
  lead 0, and documents the required observable-level convergence check.
- Add `AUTHORS.md`, recording the original scientific authors, major code
  contributors, source forks, and integration work without replacing Git
  authorship.
- Add opt-in zero-frequency particle-current counting statistics, originally
  implemented by Simon Wozny in his
  [QmeQ fork](https://github.com/si8881wo/qmeq), following
  [Emary, Phys. Rev. B 80, 235306 (2009)](https://arxiv.org/abs/0902.3544).
  `countingleads` selects one or more leads sharing an aggregate counting
  field, and `current_noise` reports the first two cumulants for Pauli,
  Lindblad, Redfield, and 1vN on both Python and Cython backends. The
  pure-Python `pyRTDnoise` approach, also available as `RTDnoise`, exposes the
  full fourth-order-kernel result, its sequential result, and a consistently
  fourth-order-truncated result. See `docs/source/theory/counting_statistics.rst`
  and Simon's [example notebook](https://github.com/si8881wo/qmeq-noise-example).
- Add lead-resolved zero-frequency particle-current covariance matrices.
  `current_noise_matrix` is ordered as `countingleads` for every supported
  first-order approach and RTD; RTD also exposes the sequential companion
  `current_noise_matrix_first`. The existing `current_noise` aggregate is
  unchanged and equals the sum over the corresponding matrix entries.
- Vendor and modernize Simon Wozny's counting-statistics notebook as tutorial
  7, retaining his authorship, source link, and BSD-2-Clause license notice.
- Include the Lamb shift in the Lindblad approach. The renormalisation of the
  many-body energies by the coupling to the leads is now built as a lead-resolved
  Hamiltonian `HLS` (a new attribute of the Lindblad approach, with the same shape
  as `Tba`) and enters the master equation through the commutator
  `-1j*[HLS, phi0]`, beyond the secular approximation. Set
  `principal_part='digamma'` to include it; the backwards-compatible default
  `principal_part='omit'` leaves Lindblad results unchanged. Numerical quadrature
  is not implemented for Lindblad. The new descriptive
  `bandwidth` and `principal_part` options replace the two meanings previously
  combined in `itype`, which remains accepted as a legacy shorthand. The new
  `qmeq.specfunc.specfunc.func_lambshift` (with a compiled twin) evaluates the
  principal-value factors in the wide-band digamma approximation. See
  `qmeq.approach.base.lindblad.generate_lamb_shift` and
  `docs/source/theory/lambshift.rst` for the implemented expressions.
- Add a prioritized maintenance roadmap in `TODO.md`.
- Add a seven-notebook tutorial path in `examples/tutorials/`, each notebook
  stating a prediction, building the smallest useful model, and asserting
  physical and numerical checks: a first sequential-transport calculation,
  Coulomb blockade in the Anderson model, bias-gate stability diagrams,
  coherence in a double dot with the Pauli/Lindblad/Redfield/1vN comparison,
  energy and heat transport with thermovoltage and unit conversions, and
  cotunnelling with the RTD and 2vN approaches including a quantum-dot heat
  engine and many-body input, and zero-frequency current counting statistics.
  The path covers the material of the legacy tutorials and Simon Wozny's noise
  example.
- Add explicit backend selection through `QMEQ_BACKEND=auto|python|cython` and
  expose `qmeq.get_backend_status()` for diagnostics and test assertions.
- Vendor the tutorials, example scripts, and appendix notebooks (previously in
  the separate `qmeq-examples` repository) under `examples/`.
- Render the example notebooks in the Sphinx documentation via `nbsphinx` and
  `nbsphinx-link`, keeping the notebooks in `examples/` as the single source.
- Run the example scripts and notebooks as tests (`qmeq/tests/test_examples.py`):
  the quick examples run with the normal suite, while the long-running 2vN / RTD
  ones are marked `slow` and run only with `pytest --runslow`.
- The example scripts now save figures as PNG (instead of PDF); the generated
  figures and data files are gitignored.
- Run the pure-Python backend test suite in CI (`test_cython.yml`), not only
  the compiled backend, so a pure-Python-only regression cannot slip through
  unnoticed.

### Changed

- Move the original introductory and RTD notebooks, with their image assets, to
  `examples/legacy_tutorials/`; they are kept for reference and their material
  is now covered by the numbered tutorials.
- Batch the 2vN Hilbert transforms in memory-bounded chunks instead of
  launching one FFT pair for every density-matrix trace.
- Skip interpolation work for exactly zero tunnelling products in the compiled
  2vN iteration while preserving parity with the pure-Python implementation.
- Standardize extension generation on Cython 3 (`>=3.0,<4`), make the Python
  language level and Cython 3 exception semantics explicit, and test compiled
  builds against both Cython 3.0 and the current Cython 3 release. Backend
  parity tests cover Pauli, Lindblad, Redfield, 1vN, 2vN, and RTD.
- Use NumPy for `pi` and `exp` constants removed from the public SciPy API.
- Allow the Sphinx documentation to build without optional Cython extensions.
- Modernize packaging: move static project metadata to `pyproject.toml`,
  derive the version dynamically from `qmeq.__version__`, use automatic package
  discovery, declare a supported-Python range of `>=3.11`, and reduce `setup.py`
  to building the Cython extensions.
- Configure pytest `testpaths` in `pyproject.toml` so `pytest` discovers the
  suite from the project root.
- Add optional-dependency extras (`test`, `docs`, `dev`) so tooling can be
  installed on demand, e.g. `pip install qmeq[test]` or `pip install -e .[dev]`.
- Refresh `INSTALL.md`: replace the deprecated `python setup.py install` command
  with `pip install .`, document validating an installed build via
  `pytest --pyargs qmeq.tests`, correct the generated documentation path, and
  update stale toolchain guidance and links.
- Point the `README.md` and `INSTALL.md` tutorial/example links at the vendored
  `examples/` directory instead of the former external repository.
- Remove the dead "reuse checked-in C files" build path and the `--cython`
  `setup.py` flag; the `.pyx`/`.pxd` sources are now always cythonized. The
  removed path was unreachable in practice: generated `.c` files are
  gitignored and never present in a fresh checkout.
- Modernize `build_wheels.yml`: bump `cibuildwheel` (v1.11 -> v4.2.0) and the
  CI runner images, target `cp311`-`cp314` to match `pyproject.toml`, add a
  `macos-14` (arm64) job alongside `macos-15-intel` (Intel), and detect the
  Homebrew GCC version dynamically in `scripts/cibw_before_all_macos.sh`
  instead of pinning to `gcc-10`. Split the workflow into a `build` job
  (per-OS matrix, no elevated permissions) and a separate `publish` job that
  only runs on tag pushes and uploads the artifacts from every completed
  build to the release in one place, matching the build/publish split
  already used for the Conda packages.
- Bump `__version__` to `1.2.0.dev0` to mark ongoing modernization work past
  the released `1.1`. `docs/source/conf.py` now derives `version`/`release`
  from `qmeq.__version__` instead of a second, separately hardcoded string.
- Replace mutable default arguments (`={}`, `=[]`, `=[0]`) with `None`,
  normalized to a fresh literal inside the function body, in `BuilderBase`,
  `BuilderManyBody`, `BuilderElPh`, `BuilderManyBodyElPh`, and
  `multiarray_sort`. Call semantics and accepted input forms are unchanged.
- Rewrite `clean.py`: paths are resolved relative to the script's own
  location instead of the current working directory, generated-file discovery
  under `qmeq/` is now recursive instead of a hardcoded per-subpackage
  directory list, and `--dry-run` lists every target without deleting
  anything.
- Drop Python 3.10 support ahead of the `1.2.0` release; the floor is now
  `>=3.11`. Python 3.10 is in security-only mode and reaches end-of-life in
  October 2026, close enough that shipping `1.2.0` with it and dropping it
  again shortly after was not worth the churn. Updated everywhere the floor
  was declared: `pyproject.toml` (`requires-python`, classifiers, Ruff
  `target-version`), `recipe/recipe.yaml`, `recipe/variants.yaml`,
  `build_wheels.yml` (`CIBW_BUILD`), `release.yml`, `test.yml`, `lint.yml`,
  `INSTALL.md`, and `AGENTS.md`.
- Rename `test_cython.yml` to `test.yml` and split its single matrix job
  into `python` (pure-Python backend, run once) and `cython` (compiled
  backend, matrix over Cython versions). The old name was misleading and the
  single-job structure ran the full pure-Python suite once per Cython-version
  matrix leg for no reason, since it does not depend on Cython at all.
- Rename `publish_conda.yml` to `release.yml`.

### Fixed

- Fix the Homebrew GCC glob in `scripts/cibw_before_all_macos.sh`: `gcc-*`
  also matched companion tools like `gcc-ar-15`, `gcc-nm-15`, and
  `gcc-ranlib-15`, and `sort -V | tail -n1` could pick one of those instead
  of the actual compiler, symlinking `gcc` to e.g. `gcc-ranlib-15` and
  breaking the macOS wheel build (observed on `macos-14`/arm64 as
  `gcc --version` invoking `ranlib --version`). The glob now requires a
  digit immediately after `gcc-`.
- Set `CC=gcc` via `CIBW_ENVIRONMENT_MACOS` in `build_wheels.yml`. Symlinking
  Homebrew's GCC as `gcc` in `CIBW_BEFORE_ALL_MACOS` was not enough on its
  own: cibuildwheel's actual build step runs in a separate subprocess that
  still defaulted to Apple clang, which fails with
  `clang: error: unsupported option '-fopenmp'`.
- Add a "Determine the macOS deployment target" step to `build_wheels.yml`
  that sets `MACOSX_DEPLOYMENT_TARGET` from the host's own macOS version, fed
  into `CIBW_ENVIRONMENT_MACOS` via `${{ env.MACOSX_DEPLOYMENT_TARGET }}`.
  Homebrew's GCC bundles an OpenMP runtime (`libgomp`) built for the host it
  runs on, so `delocate-wheel` refused to repair a wheel targeting the
  cibuildwheel default (`macosx_11_0`/`macosx_10_9`), which is older.
  Computing the value inside `CIBW_ENVIRONMENT_MACOS` itself does not work:
  that variable is evaluated by cibuildwheel's own restricted shell-subset
  parser (which rejects a pipe inside `$(...)`) to build
  `CIBW_BEFORE_ALL_MACOS`'s own environment, before that script has run, so
  it cannot even depend on anything the script produces via a hand-off file.
- Replace the `macos-13` wheel-build runner with `macos-15-intel` in
  `build_wheels.yml`; the former is a retired GitHub-hosted image and the
  build job for it would queue indefinitely instead of running.
- Set `run-install: false` on the `setup-pixi` step in `publish_conda.yml`'s
  `publish` job. That job never checks out the repository, so pixi's
  automatic manifest detection defaulted to a nonexistent
  `<workspace>/pixi.toml` and `pixi install` failed before the job could
  reach the download/upload steps; the job only needs the `pixi` CLI itself.
- Add `linux-aarch64` to `[tool.pixi.workspace] platforms` in
  `pyproject.toml`. Adding the platform to `release.yml`'s build matrix
  alone was not enough: `pixi install` refused to run at all on that
  architecture with `unsupported-platform`, since the workspace itself
  never declared it as installable.
- Compare the Hilbert transforms in
  `test_get_htransf_phi1k_matches_scalar_transforms` with
  `assert_allclose(rtol=1e-13, atol=1e-15)` instead of exact
  `np.array_equal`. The batched call and the per-slice reference loop it is
  checked against can associate the underlying FFT operations differently,
  which surfaced as a last-bit failure on `linux-aarch64` (the first time the
  suite ran there) even though the two agree exactly on x86-64. The tolerance
  is a few orders of magnitude above a double-precision ULP and still rejects
  a genuine error in the batching.
- Call `_init_before_appr` from `BuilderElPh.__init__` as well, and give
  `BuilderManyBodyElPh` its own override. That builder repeats
  `BuilderBase.__init__`'s sequence rather than delegating to it, so the hook
  added for the compiled-RTD many-body fix never ran on the electron-phonon
  path: `BuilderManyBodyElPh` still applied its state indexing after the
  `Approach` object was built, and its inherited hook would have raised
  `AttributeError` had anything invoked it. Covered by
  `test_every_builder_runs_the_pre_approach_hook`.
- Remove a dead, shadowed duplicate definition of the pure-Python 2vN
  `TermsCalculator2vN.iterate` method. The surviving implementation already
  contains the initialization performed by the removed definition.
- Make the two equal-temperature RTD integral paths share the same analytic
  wide-band component, while retaining the complementary component required
  for genuinely complex tunnel products. RTD now detects such products with a
  relative phase tolerance, so roundoff-scale imaginary parts no longer switch
  individual diagrams to a different approximation or change the current.
- Normalize the example notebooks to a Python 3 kernel and fix display-math
  markup in the RTD tutorial so the documentation builds without warnings.
- Resolve ambiguous ``Approach`` cross-references between the pure-Python and
  Cython modules via ``napoleon_type_aliases`` so the documentation builds
  cleanly with warnings treated as errors (``-W``).
- Add `scipy` to `[build-system] requires` in `pyproject.toml`. `c_lapack.pyx`
  cimports `scipy.linalg.cython_lapack`, so an isolated PEP 517 build (e.g.
  `pip install git+https://...`) failed without it declared as a build-time
  dependency.
- Make the pure-Python RTD `off_diag_corrections` handling match the compiled
  backend: `ApproachPyRTD` no longer caches a separate `self.off_diag_corrections`
  snapshot at construction time; `prepare_arrays`, `clean_arrays`, and
  `generate_kern` all read `funcp.off_diag_corrections` directly, as `c_RTD.pyx`
  already did. Previously the cached copy could desync from `funcp` (e.g. after
  directly mutating the approach object), leaving allocated arrays inconsistent
  with what `generate_kern` expected and raising `TypeError`/`AttributeError`
  only on the pure-Python backend.
- Fix `BuilderManyBody` construction with the compiled RTD approach
  (`kerntype="RTD"`): the many-body state indexing (`Na`/`Ea`) previously ran
  after the `Approach` object was constructed, so `ApproachRTD.__init__` sized
  a per-thread kernel buffer (`nbr_Wdd2_copies`) from a placeholder
  `si.npauli == 1` instead of the real many-body state count. This produced
  wrong currents and, for systems with more diagonal states than the sized
  buffer, out-of-bounds writes that corrupted the heap (observed as a
  `free(): invalid size` crash on interpreter exit). `BuilderBase` now runs
  a `_init_before_appr` hook, overridden by `BuilderManyBody`, that finishes
  the many-body state setup before the `Approach` object is built. The
  previously required `kerntype="pyRTD"`-then-switch workaround (used in
  `examples/tutorials/06_cotunnelling_and_second_order.ipynb`) is no longer
  necessary and has been removed from the tutorial.
- Use `expm1` in the pure-Python and Cython Bose functions for accuracy near
  zero, and guard the electron-phonon forms against large positive arguments.
- Define all compiled special-function names through pure-Python fallbacks when
  the Cython extensions are unavailable.
- Fix string comparisons used when expanding spin-symmetric input data.
- Fix Sphinx configuration, duplicate API metadata, malformed docstring
  references, and wrapper-page headings so documentation builds cleanly.

### Removed

- Remove the outdated `README.rst`; `README.md` is now the canonical README and
  is used as the package long description.

## [1.1] - 2021-06-04

### Added

- First-order approaches to describe electron-phonon coupling inside a quantum dot
  * Pauli (classical)
  * Lindblad
  * Redfield
  * First order von Neumann (1vN)

- Approaches to describe tunneling from metallic leads
  * Second order Real Time Diagramatic (RTD) approach

- Added BuilderManyBody class for dealing with many-body state input
- Support for Fock state removal when calculating quantum dot eigenstates

### Changed

- Refactored Approach classes:
  * Introduced separate Cython class
  * Introduced KernelHandler class for more convenient dealing with master equation matrix elements

### Fixed

- Add to a coulomb matrix element correctly when before it was not defined/used

### Removed

- Python 2.7 support

## [1.0] - 2017-07-13

### Added

- Quantum dot eigenstate calculations

- Approaches to describe tunneling from metallic leads
  * Pauli (classical)
  * Lindblad
  * Redfield
  * First order von Neumann (1vN)
  * Second order von Neumann (2vN)

[unreleased]: https://github.com/gedaskir/qmeq/compare/1.1...HEAD
[1.1]: https://github.com/gedaskir/qmeq/releases/tag/1.1
[1.0]: https://github.com/gedaskir/qmeq/releases/tag/1.0
