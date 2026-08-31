# QmeQ Changelog

## [Unreleased]

### Removed

- Remove the superseded Sphinx documentation tree and its dependency, CI, and
  packaging paths. MkDocs now owns the user guide, tutorials, theory notes,
  generated API reference, and internal conventions.

### Fixed

- Stop shipping the cythonize output in the source distribution. `setup.py`
  cythonizes into `build/cython/`, whose generated `.c` files became the
  extension modules' declared sources and were carried into the sdist
  regardless of `MANIFEST.in`, accounting for 90% of the uncompressed archive.
  The sdist now prunes them and falls from 3.8 MB to under 1 MB; installs
  regenerate them from the shipped `.pyx` sources. The artifact inventory check
  gained the matching rule, and now also runs over the wheels published from
  `build_wheels.yml` rather than only a locally built one.

- Complete RTDnoise's second-order complex-amplitude diagrams from their
  inverted electron-hole and Keldysh partners. The old ``eta0 = -1`` traversal
  conjugated individual tunnel vertices instead of the complete four-vertex
  contribution, so the counted population kernel disagreed with ordinary RTD
  at generic plaquette flux and particle-current errors reverted from cubic to
  quadratic in the coupling. The value and Laplace-derivative partners now use
  the conjugate and negative-conjugate relations, respectively, while retaining
  their transfer labels. The redundant explicit partner traversal is skipped,
  reducing second-order assembly work. Generic-flux kernel agreement and
  independent non-interacting current/noise scaling gates cover the repair.
- Base the RTD eliminated-coherence warning on the Fermi-weighted sequential
  escape rates that actually damp the closest same-charge pair. The previous
  `2*pi*sum(|T|**2)` estimate omitted reservoir occupations and could warn deep
  in Coulomb blockade even when those sequential decay channels were closed.
  The warning now uses a five-to-one splitting-to-damping margin, matching the
  documented RTD validity screen instead of the former heuristic factor 10.
  `RTDCoherenceDiagnostics.gamma_upper_bound` and `upper_bound_ratio` retain
  that occupation-independent scale for conservative comparisons.
- Store the counting-resolved coherence correction's Laplace derivative in the
  channel that is not identically zero. The zero-field Schur product
  `Wdn G Wnd` is purely imaginary and its Laplace derivative purely real, so
  `1j*product_dz.imag` kept nothing: `coherence_correction_dz` was zero to
  machine precision and the correction contributed nothing to the
  non-Markovian noise. The pair is one analytic object,
  `W_corr(z) = -1j*Wdn(z) G(z) Wnd(z)`, which also matches the counting path's
  convention of real kernels and purely imaginary `_dz` arrays. Gated against
  an independent finite-`z`, transfer-resolved reference; the effect on the
  noise is `O(Gamma**3)`, which is why no existing gate saw it.
- Make the second-order Laplace differentiation step unit covariant. The
  `max(1.0, ...)` floor was absolute, so a model whose whole energy scale sat
  below `1` in the caller's units was differentiated with a step orders of
  magnitude too coarse relative to its own features. The step is now a pure
  fraction of the model's energy scale, and the energy-rescaling covariance
  test sweeps far below the removed floor as well as above it.

- Correct the RTDnoise first-order Laplace derivatives. `phi` and the Fermi
  function take the scaled argument `(E-mu)/T`, so `d/dz` carries `1/T_lead`;
  the pre-existing diagonal derivative and the new population-coherence blocks
  both omitted it. The reported non-Markovian noise was therefore not
  covariant under an overall rescaling of `E`, `mu`, `T` and `Gamma`, and
  carried a leading `O(Gamma**2)` error against the exact non-interacting
  reference at every temperature except `T = 1` -- which is where every
  existing gate ran. The pinned QmeQ-1.1 `Lpm_first_dot` bundle now confirms
  the correction exactly: current values equal the historical array divided by
  `T_lead` per lead, in the unequal-temperature scenarios too.
- Complete the analytic population-coherence Laplace derivative. `pi*f(u) +
  1j*phi(u)` is a single analytic function of the scaled energy, so its
  derivative cannot keep the `phi'` channel and drop the `pi*f'` one; the
  latter is ~0.8 times the size of the former and cancels only in the diagonal
  limit. Four of the twelve call sites also took the sign of the `phi`
  coefficient where the sign of the `pi*f` coefficient was required, which
  cancelled the charge-conserving derivative outright. The block's reduction to
  the diagonal first-order kernel at `a1 == b1` is now a test, at unequal lead
  temperatures.
- Stop rejecting population/coherence coordinate pairs whose charges differ by
  more than one electron. Such pairs are not connected by a first-order vertex,
  so their block is structurally zero; rejecting them by charge alone made
  `off_diag_corrections=True` raise `ValueError` in RTDnoise for any dot with
  more than two single-particle levels. Rejection is now conditional on a
  nonzero amplitude, and a three-level scenario covers it.

### Added

- Attach the source distribution to the GitHub release alongside the wheels.
  `build_wheels.yml` gained an `sdist` job that applies the same tag/version
  guard as the wheel jobs, builds the sdist, runs `twine check` and the
  artifact inventory over it, installs it into a clean environment outside the
  checkout, and confirms the compiled backend and the installed metadata. The
  publish job now uploads both the wheels and that sdist, so a release no
  longer offers binaries only.

- Support RTD's eliminated-coherence correction in `RTDnoise` when
  `off_diag_corrections=True`. First-order population-coherence blocks now have
  one shared traversal, an explicit-array insertion path independent of the
  historical `RtdMatrix` selector, and a lead- and transfer-resolved Schur
  composition with analytic Laplace derivatives. Summing its transfer sectors
  reproduces the ordinary RTD kernel and current, and the exact non-interacting
  reference confirms the corrected real-amplitude *current* at the retained
  order. The corresponding noise claim is **not** settled: see the open item
  below. `off_diag_corrections=False` remains the immutable historical-fixture
  mode.
- Replace RTDnoise's fixed `1e-8` Laplace-energy shifts with analytic
  derivatives for first-order coherence blocks and the bare propagator, plus a
  scale-aware centered derivative for every explicit second-order
  direct/exchange integral. A full-kernel test compares the latter against an
  independent five-point stencil. Live derivative arrays use the explicit
  `_dz` suffix; the old `*_dot` names remain only as immutable historical
  fixture keys.
- Make RTDnoise's equal-temperature counted direct/exchange integrals use the
  same analytic Appendix-D wide-band real component as stationary RTD. The
  former finite-pole counted copy broke `W(chi=0,z=0) == W`, moved the
  stationary state and pseudoinverse with `dband`, and left a spurious
  `O(Gamma**2)` noise residual. Practical-bandwidth kernel-identity,
  fourfold-bandwidth-invariance, and independent non-interacting cubic-order
  gates now cover the repair; unequal-temperature Ozaki integrals retain their
  existing cutoff-convergence requirement. Bare `RTDnoise` also dispatches its
  profiled direct/exchange value and derivative calls to numerically equivalent
  compiled wrappers when the Cython backend is active; explicit `pyRTDnoise`
  stays all-Python.
- Warn when legacy RTD's diagonal-density-matrix approximation is not
  spectrally resolved. The diagnostic estimates each same-charge coherence's
  broadening from `2*pi*sum(|T|**2)` over adjacent-charge transitions, uses a
  conservative ten-to-one splitting-to-broadening threshold, records the
  closest case as `approach.rtd_coherence_diagnostics`, and emits one
  `RTDCoherenceWarning` per approach. The check is shared by `pyRTD` and the
  compiled `RTD` backend and does not modify the kernel.
- Include the charge sector and state indices in the RTD coherence diagnostic,
  report splittings affected by the inverse-Liouvillian clamp, expose the
  RTD warning categories at the package top level, and warn when no tunnel
  broadening is present for the active same-charge states.
- Diagnose every stationary solution for physicality instead of returning it
  silently. After each master-equation solve, the approach now checks the
  reduced density matrix for negative populations, deviation of the trace from
  one, and NaN/inf entries. An unphysical result emits a `QmeqRuntimeWarning`
  (once per approach instance) and is recorded as a queryable
  `approach.stationary_diagnostics` object with a `physical` flag plus the
  minimum population, trace, trace deviation, and solver-reported conditioning
  (least-squares rank and residual where the solver provides them), so
  scripted sweeps can filter on the diagnostic rather than parse stderr. The
  check runs in shared pure-Python code called from the approach `solve`
  methods, so its behaviour is identical across approaches and backends; for
  2vN only the state after the final iteration is diagnosed, since
  intermediate iterates may be unphysical while still converging.
- Add public `QmeqWarning` and `QmeqRuntimeWarning` categories so callers can
  capture or filter all QmeQ diagnostics as a group while distinguishing
  numerical/runtime failures from input fallbacks.
- Add `qmeq.approach.dm_layout`, a written specification of the packed real
  density-matrix layout that Lindblad, Redfield, 1vN, their electron-phonon
  variants and RTD all solve in. The layout was previously defined only by its
  uses, with the offset arithmetic, inclusion test and conjugation sign
  open-coded at 19 sites across the pure-Python and Cython kernel handlers.
  The module states nine numbered rules, provides an immutable
  `LiouvilleState` view and a reference `DensityMatrixLayout` with
  pack/unpack/trace, and is accompanied by a test module in which every test
  names the rule it pins. Two conventions that had no written record are now
  specified and covered: the packed kernel implements `rho -> -i (W rho)`,
  which is why Lindblad passes `1j*fct` where 1vN passes `fct`; and the
  matrix-free handler writes its imaginary rows with the opposite sign from
  the assembled kernel, which leaves the stationary null space unchanged but
  means `dphi0_dt` is not literally the packed time derivative.
- Add `StateIndexingDM.get_ind_dm0_bool` and `get_ind_dm0_conj`, named forms of
  the `maptype=2` and `maptype=3` integer arguments, matching the accessor
  names the Cython handler already used. The integer `maptype` interface is
  unchanged.
- Add `QMEQ_STRICT_INDEX=1`, a pure-Python diagnostic that turns an insertion at
  a density-matrix element with no index into an `IndexError` naming the method
  and the offending endpoint, instead of silently skipping it. Off by default,
  and because the check sits inside a branch a correct caller never takes it
  costs nothing in the normal path. The compiled insertion methods are
  `noexcept nogil` and cannot raise, so run the strict leg with
  `QMEQ_BACKEND=python`. This makes the whole suite a standing probe for a
  missing `is_included` guard; the full suite passes under it.
- Add `RtdMatrix`, an `IntEnum` naming the eight destination arrays that RTD
  insertions select, replacing a bare trailing integer at 61 call sites — 30 in
  `RTD.py` and 31 in `c_RTD.pyx`. The compiled path cannot use a Python enum
  inside `nogil` code, so `c_kernel_handler.pxd` mirrors it as `RtdMatrixC`,
  declared `cpdef` rather than `cdef` specifically so the two copies can be
  compared member-for-member in the test suite rather than drifting silently.
- Add type hints opportunistically, for readability only: no type checker, no
  `py.typed` marker and no CI gate. `qmeq.approach.dm_layout` is fully
  annotated; `qmeq.approach.kernel_handler` and the `qmeq.indexing` lookups
  carry hints on the signatures that were hardest to read. Note that the kernel
  handlers accept `StateIndexingDM | StateIndexingDMc`, since the 2vN
  approaches pass the latter.
- Add one validated external JSON/NPZ reference-bundle infrastructure for
  historical and future numerical snapshots. The previous builder and
  electron-phonon Python dictionaries are preserved losslessly as 100 arrays
  in a `legacy` bundle whose unknown generating revision and environment are
  stated explicitly; they are not mislabeled as QmeQ 1.1. The 15
  counting-statistics arrays retain their separately recorded Simon Wozny
  source commit. Maintainer-only generators now live under
  `scripts/reference_data/`, outside the test package, and tests never generate
  expected values implicitly.
- Add an installed-artifact CI gate that builds and checks both wheel and
  source distribution, installs each outside the checkout, verifies the forced
  Python and Cython backends, and runs the external-reference suites from the
  installed package.
- Add a provenance-locked QmeQ 1.1 regression corpus generated from commit
  `96cc51076458b11f7db81a5d7d8df04c30bf8384`. External JSON/NPZ fixtures cover
  every electronic and electron-phonon method available in 1.1. The legacy RTD
  matrix covers equilibrium, real coherences with off-diagonal corrections on
  and off, complex tunnel amplitudes, unequal temperatures, many-body input,
  and the documented spin-symmetry fallback to charge indexing. It records the
  first-order, second-order, and coherence-elimination contributions; all
  population/coherence blocks; stationary states; and particle, energy, and
  heat currents. Invariant tests cover decomposition, trace preservation,
  residuals, normalization, conservation laws, equilibrium, heat-current
  consistency, and structural zeros. The manifest records model reconstruction
  data, array ordering, tolerances, and the narrow compatibility envelopes for
  the intentional post-1.1 complex-integral branch correction without changing
  the historical values. Tests never regenerate expected data implicitly, and
  the generator rejects any source revision or QmeQ version other than the
  pinned 1.1 checkout. Compiled electron-phonon Lindblad parity remains the
  suite's single strict, conditional `xfail`, with its existing P1 correctness
  gap documented inline rather than hidden by a wider tolerance.
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
  fourth-order-truncated result. See `docs/docs/theory/counting-statistics.md`
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
  `docs/docs/theory/lambshift.md` for the implemented expressions.
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
- Render the example notebooks in the documentation via `mkdocs-jupyter`,
  keeping the notebooks in `examples/` as the single source.
- Run the example scripts and notebooks as tests (`qmeq/tests/test_examples.py`):
  the quick examples run with the normal suite, while the long-running 2vN / RTD
  ones are marked `slow` and run only with `pytest --runslow`.
- The example scripts now save figures as PNG (instead of PDF); the generated
  figures and data files are gitignored.
- Run the pure-Python backend test suite in CI (`test_cython.yml`), not only
  the compiled backend, so a pure-Python-only regression cannot slip through
  unnoticed.

### Changed

- Build the documentation with `mkdocs build --strict`, so any warning fails
  the build rather than degrading a page quietly, and fix the docstrings that
  blocked it. A bare index expression such as `szlst[charge][sz]` was read as a
  Mkdocstrings shorthand cross-reference, `Phi[1](k)` was read as a Markdown
  link whose target is `k`, and five parameter continuation lines were indented
  three spaces instead of four. Both rules are recorded in the docstring
  conventions page.

- Replace warning-like standard-output prints with typed warnings across input
  validation, state indexing, solver fallback/failure paths, 2vN grid changes,
  and the Python and Cython RTD energy-current implementations. Deliberate
  state-display and build-progress output remains unchanged; RTDnoise no longer
  dumps a full failed kernel matrix to standard output.
- Generate Cython's intermediate C files under `build/cython/` instead of next
  to the canonical `.pyx` sources. Remove the obsolete per-file ignore list and
  the risk of accidentally compiling stale source-tree C output.
- Remove the unreachable `scipy.misc.factorial` compatibility fallback and use
  NumPy's public `emath` namespace for the complex logarithm. The fast suite on
  NumPy 2.5.1 and SciPy 1.18.0, plus a clean Cython 3.2.8 rebuild, emits no
  deprecation, future, or pending-deprecation warnings.
- Keep `itype` as a supported compatibility interface alongside the descriptive
  `bandwidth` and `principal_part` options; no `itype` deprecation is planned.
- Rename the RTD array `Lnn` to `Lnn_inv`, and `add_element_Lnn` to
  `add_element_Lnn_inv`. It holds the inverse of the bare coherence energy
  splitting used to eliminate coherences, not the coherence-sector Liouvillian;
  the Phase 0 reference fixtures already recorded it under the honest key
  `inverse_Lnn`, which is provenance-locked and unchanged. Note the two backends
  still differ in shape here: two-dimensional in pure Python, a bare diagonal in
  Cython.
- Stop binding the name `si` to `self.si_elph` in the pure-Python
  electron-phonon Pauli and 1vN approaches. Those modules hold two indexing
  objects of *different* classes — `si` is a `StateIndexingDM` while `si_elph`
  is a `StateIndexingDMc` — and the shared local name made the distinction
  invisible at the call site, in one case with opposite meanings in adjacent
  methods of the same class.
- Name the "no index" sentinel at 61 sites across both backends, replacing bare
  `-1` comparisons, and drop the comments that had been restating those
  comparisons in English. `NO_INDEX` records that the value must never be used
  as an index: NumPy reads it as the last row or column, and the compiled
  handler is built with `wraparound=False` and `boundscheck=False`, so there it
  is an out-of-bounds access rather than a wrap.
- Document what `maptype` actually selects in all three `get_ind_dm0` methods.
  The method returns an index for `maptype` 0 and 1 but a boolean for 2 and 3,
  which no docstring previously said; each class now carries a table of the
  selectors it supports and why the missing ones do not exist.

- Route the duplicated packed-real index arithmetic in both kernel handlers
  through one named definition. The repeated `ndm0 + i - npauli` offset is now
  a precomputed `imag_offset`, the existence test for an imaginary partner is
  the equivalent and clearer `i >= npauli`, the excluded-element sentinel has
  a name in both backends, and the `maptype` integers are gone from the
  handlers. No sign, prescription or kernel value changes; the historical
  reference corpora are reproduced unchanged on both backends.

### Performance

- Raise `MAX_CACHE`, the `lru_cache` bound on the memoised special functions,
  from 100 to 10000. The bound sets the cache hit rate and the hit rate sets the
  runtime: on a pure-Python RTD solve the aggregate hit rate goes from about
  75-81% to 96-97% and the solve is roughly 2.2 times faster. The knee is near
  1000 and 50000 adds about one percent more, so 10000 captures nearly all of
  the available gain for at most about 17 MB at a measured 207 bytes per entry.
  The bound stays finite deliberately, since distinct keys per solve grow about
  elevenfold per added orbital and parameter sweeps generate fresh float keys
  indefinitely; a regression test asserts finiteness. Memoisation is exact, so
  results are unchanged: every array in the RTD reference matrix is bitwise
  identical across the two bounds. The compiled backend uses its own
  `c_specfunc` and is unaffected. Caching `integralD` and `integralX` was
  measured and rejected -- with twelve arguments and three independent energies
  they reach a 0.3% hit rate and the key hashing costs more than the calls it
  saves.


### Fixed

- Make the QmeQ 1.1 electron-phonon kernel regression portable across LAPACK
  implementations. Raw packed-coherence kernels can differ by a diagonal
  similarity when `eigh` chooses the opposite sign for a many-body eigenvector;
  the test now requires agreement under one consistent state-level sign gauge
  instead of treating that arbitrary basis convention as physics.
- Skip the Cython build-directory probe when Cython is intentionally absent
  from a pure-Python test installation. Compiled CI jobs still exercise it.
- Raise `ValueError` instead of silently returning `None` from `get_ind_dm0`
  for an unsupported `maptype`. `None` is `np.newaxis`, so a wrong selector
  reshaped an array rather than raising and surfaced far from its cause.
- Stop `Builder.get_phi0` asking `StateIndexingDMc` for a conjugation map it
  does not have. The `maptype=3` lookup ran unconditionally and its result was
  discarded on the `StateIndexingDMc` branch, so every 2vN call relied on the
  silent `None` return above. The lookup now happens only in the branch that
  uses it, and uses the named accessor. Results are unchanged.
- Make inserting a matrix element at an uncarried endpoint a no-op in both
  kernel-handler backends. The `-1` sentinel would otherwise index the last row
  or column and corrupt an unrelated entry. Every shipped caller already guards
  with `is_included`, and a probe run of the full suite with a hard assertion
  never fired, so no existing behaviour changes.

- Adopt pytest 9's native `[tool.pytest]` TOML configuration and separate
  notebook execution from the default suite. The normal suite still exercises
  the quick Python example; notebooks run explicitly with `-m notebook` in an
  environment where Jupyter kernels may open local sockets. The scheduled
  example workflow selects the complete script/notebook group explicitly.
- Move the original introductory and RTD notebooks, with their image assets, to
  `examples/legacy_tutorials/`; they are kept for reference and their material
  is now covered by the numbered tutorials.
- Batch the 2vN Hilbert transforms in memory-bounded chunks instead of
  launching one FFT pair for every density-matrix trace.
- Skip interpolation work for exactly zero tunnelling products in the compiled
  2vN iteration while preserving parity with the pure-Python implementation.
- Standardize extension generation on Cython 3 (`>=3.0,<4`) and make the Python
  language level and Cython 3 exception semantics explicit. Backend parity tests
  cover Pauli, Lindblad, Redfield, 1vN, 2vN, and RTD.
- Use NumPy for `pi` and `exp` constants removed from the public SciPy API.
- Allow the documentation to build without optional Cython extensions.
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
- Bump `__version__` to `1.2.0.dev1` to mark ongoing modernization work past
  the released `1.1`.
- Check the tag against the package version in `build_wheels.yml`, the guard
  `release.yml` already had. Without it a tag whose name disagrees with
  `qmeq.__version__` still built, and the publish job attached wheels carrying
  the package version to a release named after the tag.
- Replace mutable default arguments (`={}`, `=[]`, `=[0]`) with `None`,
  normalized to a fresh literal inside the function body, in `BuilderBase`,
  `BuilderManyBody`, `BuilderElPh`, `BuilderManyBodyElPh`, and
  `multiarray_sort`. Call semantics and accepted input forms are unchanged.
- Rewrite `clean.py`: paths are resolved relative to the script's own
  location instead of the current working directory, generated-file discovery
  under `qmeq/` is now recursive instead of a hardcoded per-subpackage
  directory list, and `--dry-run` lists every target without deleting
  anything.
- Make OpenMP optional and compiler-aware, selected by
  `QMEQ_OPENMP=auto|on|off` (default `auto`). `setup.py` no longer guesses a
  flag from `os.name`; it probes candidate flag sets against the active
  compiler — `/openmp` for MSVC, `-fopenmp` for GCC, and
  `-Xpreprocessor -fopenmp` with an explicit `-lomp` for Apple clang, including
  variants that add the include and library directories of a prefix taken from
  `QMEQ_OPENMP_PREFIX`, `sys.prefix`, or `brew --prefix libomp`. `auto` falls
  back to a serial build with a warning, `on` makes that fallback an error, and
  `off` skips OpenMP outright. This makes `pip install .` work with an
  unmodified Apple clang toolchain, which previously failed outright.
  A serial build is fully functional: Cython lowers `prange` to an ordinary
  loop. Note that serial and threaded builds can differ in the last bits of
  reduced quantities (observed on the RTD energy current), because the number
  of per-thread accumulation buffers changes the summation order.
- Test the compiled build against all three wheel toolchains on every push:
  `test.yml`'s compiled job is now a matrix over `ubuntu-latest`,
  `windows-latest`, and `macos-14`, instead of only `ubuntu-latest`. The
  oldest-supported-Cython leg stays Linux-only.
- Sweep the whole declared Python range in CI: the pure-Python job now runs on
  3.11, 3.12, 3.13, and 3.14, matching `pyproject.toml`'s classifiers.
- Build the documentation in CI as a `mkdocs-docs` job in `test.yml`, so a
  broken internal link or a nav entry with no matching page fails the build.
- Add `slow.yml`, running the `--runslow` example suite over both backends
  weekly and on demand, so the long-running notebooks stay a release gate
  without slowing down every push.
- Guard the `setup()` call in `setup.py` with `if __name__ == '__main__'` so its
  helpers can be introspected without triggering a build. Build frontends run
  the file as `__main__`, so installs are unaffected.
- Build the macOS wheels serially and drop the Homebrew-GCC symlink script
  (`scripts/cibw_before_all_macos.sh`) along with the deployment-target
  juggling it required. See the corresponding entry under Fixed.
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

- Track the three images the legacy notebooks embed
  (`examples/legacy_tutorials/images/`). A blanket `*.png` rule, meant for the
  figures the example scripts generate, also ignored them, so they existed only
  in working trees that predated the rule: from a fresh clone the documentation
  build failed with `image file not readable`. The negations are listed per
  file, so genuinely generated figures dropped into the same directory stay
  ignored.
- Add `ipython-pygments-lexers` to the `docs` extra. The notebooks declare the
  `ipython3` Pygments lexer, which Pygments itself does not provide, so a
  docs-only install failed under `-W` with
  `Pygments lexer name 'ipython3' is not known`. It happened to work in any
  environment that also had the `test` extra installed, which is why it went
  unnoticed.
- Add `setuptools>=77` to the `test` extra. The backend tests introspect
  `setup.py` in a subprocess, and Python 3.12 and newer no longer provide
  setuptools alongside the interpreter, so those tests failed on 3.12-3.14.
  They now also skip cleanly when it is missing rather than erroring.
- Constrain the CI `setuptools` install to `>=77`, matching
  `[build-system] requires`. The compiled jobs install their build dependencies
  by hand and then use `--no-build-isolation`, but a bare
  `pip install setuptools` is a no-op when an older one is already present, so
  those jobs silently reused the runner image's version. That is new enough on
  the Ubuntu images and too old on the macOS and Windows ones, where it cannot
  parse the PEP 639 `license = "BSD-2-Clause"` metadata and fails with
  ``invalid pyproject.toml config: `project.license` ``.
- Add `test_partial_extension_set_never_imports`, covering the documented
  contract that a *partially* built extension set must fail to import rather
  than mixing compiled and pure-Python implementations -- under `auto` as well
  as `cython`, since `auto`'s quiet fallback is only meant for a cleanly absent
  extension set.
- Fix the macOS wheels, which could not be built at all and then could not be
  installed on most Intel Macs. The build depended on symlinking a Homebrew GCC
  over `gcc` (Apple clang rejects `-fopenmp`), which broke when the glob started
  matching companion tools like `gcc-ranlib-15`; and because Homebrew's GCC
  bundles a `libgomp` built for the *runner's* macOS version, `delocate-wheel`
  then forced the wheel's deployment target up to match, producing
  `macosx_15_0` Intel wheels that macOS 12-14 cannot install (with no working
  source fallback, since the sdist build hit the same clang failure). macOS
  wheels are now built with Apple clang and `QMEQ_OPENMP=off`, so they carry no
  bundled OpenMP runtime, keep cibuildwheel's low deployment target, and stay
  installable on older macOS. The kernels remain compiled; only the threading is
  lost, and the Conda packages still ship with OpenMP.
- Stop `c_RTD.pyx` from calling `omp_get_max_threads` and `omp_get_thread_num`
  directly; both now go through a `#ifdef _OPENMP` shim that reports a single
  thread. Calling them directly made a non-OpenMP build a hard link error on
  macOS and, on Linux, produced an extension with unresolved OpenMP symbols that
  imported only because SciPy had already loaded `libiomp5` into the process.
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
- Fix duplicate API metadata, malformed docstring references, and
  wrapper-page headings so documentation builds cleanly.

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
