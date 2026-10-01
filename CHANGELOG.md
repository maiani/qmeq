# QmeQ Changelog

## [Unreleased]

Upgrading from QmeQ 1.1: a script that uses only the 1.1 interface gives the
1.1 result, raises an error that says what to change, or gives a different
number because 1.1 was wrong there. The cases, each detailed below:

- **Now raises.** A Lindblad system without `principal_part` (`'omit'` with
  `itype=0` gives the 1.1 result); an unknown `kerntype`, `indexing`,
  `symmetry`, `itype` or `itype_ph`; `mfreeq=True` with RTD; a lead temperature
  that is zero, negative, or missing from a partial `tlst`; `get_ind_dm0` with
  an unsupported `maptype`. Python 3.11 or newer is required.
- **Different numbers, because 1.1 was wrong.** Compiled electron-phonon
  Lindblad (trace-violating coherence columns); any compiled calculation after
  assigning `system.mulst`, `tlst` or `dlst` (the first values were kept);
  `BuilderManyBody` with compiled RTD; `BuilderManyBodyElPh`, which crashed
  or returned wrong numbers; 2vN whenever a same-charge coherence
  is complex, which includes real-parameter models under bias (2.5% in the
  current of the 1.1 reference double dot); RTD with complex tunnel amplitudes
  (3e-3 relative at `dband=20`, 4e-5 at `dband=200`); RTD with lead-dependent
  bandwidths at unequal temperatures or with complex amplitudes (about 3e-4);
  electron-phonon rates through the Bose function (about 1e-10).
- **Different output.** What used to be printed is now a `QmeqWarning` or
  `QmeqRuntimeWarning`, and new diagnostics warn about unphysical stationary
  states and about RTD outside its regime; filter `qmeq.QmeqWarning` to
  silence them all. The RTD array `Lnn` is now `Lnn_inv`.

### Added

- **Lamb shift in the Lindblad approach.** The renormalisation of the
  many-body energies by the leads is built as a lead-resolved Hamiltonian
  `HLS` (a new approach attribute, shaped like `Tba`) and enters through the
  commutator `-1j*[HLS, phi0]`, beyond the secular approximation. Its weight
  is the one generated alongside QmeQ's own jump operators: QmeQ's Lindblad
  approach is not the secular (Davies) generator but, following
  `KirsanskasFranckieWacker2018`, dresses each tunnelling matrix element with
  the square root of an occupation factor and keeps one jump operator per
  lead, the construction `NathanRudner2020` derived as a controlled
  weak-coupling approximation. The weight is therefore the arithmetic mean of
  the principal values plus the cutoff-free correction `func_ule_shift`
  [NathanRudner2020, Eq. (D7)], with the indices of Eq. (D8) as corrected by
  [NathanRudner2021Erratum, Eq. (6)]: a lower intermediate state emits into an
  empty lead state and adds `delta_f(-x1, -x2)`, an upper one absorbs an
  occupied lead electron and adds `delta_f(y1, y2)`. It agrees with the
  Bloch-Redfield shift on every diagonal element, so one-dimensional and
  uniformly shifted charge sectors (the analytic single-level and
  spin-degenerate cases) are unaffected, and differs off the diagonal.
  `principal_part` selects it and has no default for Lindblad: `'digamma'`
  evaluates the principal values in the wide-band digamma form
  (`func_lambshift`, with a compiled twin), `'quad'` integrates them over the
  lead band (`func_lambshift_quad`) and converges to `'digamma'` at the `1/D`
  rate a test asserts, and `'omit'` reproduces QmeQ 1.1 and Appendix F of the
  QmeQ paper. See `generate_lamb_shift` and `docs/docs/theory/lambshift.md`.
- **Descriptive transport options** `bandwidth` (`'finite'`/`'infinite'`) and
  `principal_part` (`'quad'`/`'digamma'`/`'omit'`), replacing the two meanings
  combined in `itype`. `itype` stays a supported shorthand with no deprecation
  planned; for Lindblad it selects only the bandwidth. Conflicting combinations
  raise `ValueError`.
- **Zero-frequency particle-current counting statistics**, originally
  implemented by Simon Wozny in his
  [QmeQ fork](https://github.com/si8881wo/qmeq) following
  [Emary, Phys. Rev. B 80, 235306 (2009)](https://arxiv.org/abs/0902.3544).
  `countingleads` selects the counted leads; `current_noise` gives the first
  two cumulants for Pauli, Lindblad, Redfield and 1vN on both backends, and
  `current_noise_matrix` the lead-resolved covariance, ordered as
  `countingleads`, whose entries sum to `current_noise`. The `RTDnoise`
  approach (Python traversal, with compiled scalar integrals when the Cython
  backend is active; `pyRTDnoise` is all Python) gives the full fourth-order
  result, its sequential companion (`current_noise_first`,
  `current_noise_matrix_first`) and a consistently fourth-order-truncated
  result, and supports RTD's eliminated-coherence correction when
  `off_diag_corrections=True`. Its Laplace derivatives are analytic for
  first-order coherence blocks and the bare propagator and a scale-aware
  centred difference, checked against a five-point stencil, for the explicit
  second-order integrals; its equal-temperature integrals share stationary
  RTD's analytic wide-band real part, so `W(chi=0, z=0) == W`. What is
  validated against exact non-interacting transport, for currents and for the
  noise, is listed in `docs/docs/theory/counting-statistics.md`; see also
  tutorial 7.
- **RTD truncation order**, `approach.rtd_order`, on `RTD`, `pyRTD`,
  `RTDnoise` and `pyRTDnoise`, defaulting to the previous behaviour, 2. Order 1
  keeps only the two-vertex block and reproduces the golden-rule kernel
  exactly (on a single level `phi0` and all currents equal Pauli at `itype=1`
  to `0.0` on both backends), and accepts a thermal bias, since only the
  four-vertex integrals need equal lead temperatures. `off_diag_corrections`
  stays an independent switch, so enabling it at order 1 is a diagnostic
  control rather than a consistent truncation. Unimplemented or non-integer
  orders raise, and changing the order restarts the approach.
- **Diagnostics instead of silent results.**
  - Every stationary solution is checked for negative populations, a trace
    away from one, and NaN/inf entries. An unphysical result warns once per
    approach (`QmeqRuntimeWarning`) and is recorded as
    `approach.stationary_diagnostics`, with a `physical` flag, the minimum
    population, the trace and its deviation, and the solver's rank and residual
    where available. 2vN is diagnosed after its final iteration only.
  - RTD warns (`RTDCoherenceWarning`, once per approach) when its
    diagonal-density-matrix approximation is not spectrally resolved, judging
    the closest same-charge pair against the Fermi-weighted sequential escape
    rates that damp it with a five-to-one margin, and records the case, with
    its charge sector and state indices, as
    `approach.rtd_coherence_diagnostics`. `RTDNoBroadeningWarning` flags active
    same-charge states with no tunnel broadening at all.
  - RTD warns (`RTDBandwidthWarning`) at unequal temperatures when `dband` is
    not conservatively separated from the transport scales.
  - With a finite band (`itype` 0 or 2), a `QmeqWarning` names every lead
    whose band excludes all the transitions it couples to. Their rates are
    then exactly zero, so moving `dband` by `1e-9` across a transition energy
    turned a finite current into a silent `0`. Shown once per system; leads
    with no coupling at all are not flagged.
  - With `bandwidth='infinite'` (`itype` 1 or 3), a `QmeqWarning` names every
    lead whose band excludes a transition the lead couples to. The band edges
    remove no transition there, so a symmetric `dband` of `0.01` gave the same
    current as `1e5` without comment, and an asymmetric one moved the 1vN and
    Redfield principal parts by tens of percent. A band of exactly `(0, 0)`,
    the stored value when no `dband` is given, is not flagged; neither are
    RTD and RTDnoise, which keep `dband` as a regulator.
  - Under `'sz'` or `'ssq'` indexing, a `QmeqWarning` names the lead channels
    and phonon baths that break the symmetry the indexing assumes: a channel
    reaching both spins breaks S_z, and channels sharing a chemical potential,
    temperature and band that couple the two spins differently break total
    spin. The indexing then drops couplings or coherences the model has; with
    spin-dependent tunnelling, `'ssq'` moved a double-dot current by 8% while
    `'sz'` and `'charge'` agreed to machine precision. SU(2)-symmetric
    spin-resolved leads, including `symmetry='spin'`, stay silent, and Pauli is
    not flagged under `'sz'`, which does not change it.
  - `QmeqWarning` and `QmeqRuntimeWarning` are public, so all QmeQ diagnostics
    can be captured or filtered as a group; the RTD categories are exported at
    the package top level.
- **Backend selection** with `QMEQ_BACKEND=auto|python|cython`, and
  `qmeq.get_backend_status()` for diagnostics. `auto` falls back to pure Python
  only for a cleanly absent extension set; a partially built one never imports.
- **Packed density-matrix layout specification**, `qmeq.approach.dm_layout`:
  nine numbered rules for the real layout that Lindblad, Redfield, 1vN, their
  electron-phonon variants and RTD solve in, previously open-coded at 19 sites,
  with a `LiouvilleState` view, a reference `DensityMatrixLayout`, and a test
  per rule. It records two unwritten conventions: the packed kernel implements
  `rho -> -i (W rho)`, which is why Lindblad passes `1j*fct` where 1vN passes
  `fct`, and the matrix-free handler writes its imaginary rows with the
  opposite sign, so `dphi0_dt` is not literally the packed time derivative.
- **Smaller interface additions.** `StateIndexingDM.get_ind_dm0_bool` and
  `get_ind_dm0_conj`, named forms of `maptype=2` and `3`. `QMEQ_STRICT_INDEX=1`,
  a pure-Python diagnostic that turns an insertion at an unindexed
  density-matrix element into an `IndexError` (the full suite passes under
  it). `RtdMatrix`, an `IntEnum` naming the eight RTD destination arrays, with
  a compiled mirror `RtdMatrixC` compared member-for-member in the tests.
  Opportunistic type hints, for readability only. `BuilderManyBodyElPh`
  accepts `itype_ph`, as `BuilderElPh` does, instead of only by assignment
  after construction.
- **Tutorials and examples.** A seven-notebook tutorial path in
  `examples/tutorials/`, each stating a prediction and asserting physical and
  numerical checks: sequential transport, Coulomb blockade, stability
  diagrams, coherence and the choice of first-order approach, energy and heat
  transport, cotunnelling with RTD and 2vN, and counting statistics (Simon
  Wozny's notebook, vendored with his authorship and licence notice). The
  example scripts and appendix notebooks from the former `qmeq-examples`
  repository are vendored under `examples/`, with the original introductory
  and RTD notebooks kept for reference in `examples/legacy_tutorials/`,
  rendered in the
  documentation through `mkdocs-jupyter`, and run as tests
  (`qmeq/tests/test_examples.py`, long 2vN/RTD ones behind `--runslow`, notebooks
  behind `-m notebook`). Example scripts save PNG figures.
- **Reference data and test infrastructure.**
  - One validated JSON/NPZ reference-bundle format under
    `qmeq/tests/data/`, loaded through `qmeq/tests/reference_data.py`, with
    maintainer-only generators in `scripts/reference_data/`; tests never
    regenerate expected values. The previous builder and electron-phonon
    dictionaries are kept losslessly as a `legacy` bundle of unknown
    provenance, and the counting-statistics arrays keep their recorded source
    commit.
  - A provenance-locked QmeQ 1.1 regression corpus generated from commit
    `96cc51076458b11f7db81a5d7d8df04c30bf8384`, the 2024 packaging fork of
    1.1, which behaves as the PyPI
    release apart from its `expm1` Bose function. It covers every electronic
    and electron-phonon method in 1.1, including a legacy RTD matrix
    (equilibrium, real and complex coherences with off-diagonal corrections on
    and off, unequal temperatures, many-body input, the spin-symmetry fallback)
    that records every contribution and block, with invariant tests for
    decomposition, trace, conservation laws, equilibrium and structural zeros.
  - An exact non-interacting (NEGF) transport reference for grading currents
    and noise at the retained order.
  - A narrowly scoped Ruff correctness gate for Python files and notebooks.
- **Packaging and release.** A source-based Conda recipe for compiled Python
  3.11-3.14 on Linux x86-64 and aarch64 and on Intel and Apple Silicon macOS,
  published to a prefix.dev channel only after every variant builds and passes
  the fast suite. GitHub releases carry the source distribution beside the
  wheels, each checked with `twine`, the artifact inventory, and an install
  outside the checkout. An installed-artifact CI gate runs the reference suites
  from installed wheels and sdists on both forced backends. Optional extras
  `test`, `docs` and `dev`.
- `AUTHORS.md`, recording the scientific authors, major contributors, source
  forks and integration work.

### Changed

- **The Lindblad `principal_part` has no default**, in the electron-phonon
  variant too. QmeQ 1.1 had no Lamb shift and 1.2.0.dev9-dev10 switched it on
  by default, so the same script computed different things in different
  versions; since `itype` does not select the shift, any default would repeat
  that. A Lindblad system without it raises `ValueError` naming the choices.
  `bandwidth` defaults to `'infinite'` when neither it nor `itype` is given, a
  path no 1.1 script reaches. Tutorials 4 and 7 state `principal_part='omit'`,
  which is what they were written against: tutorial 4's "6-13% below 1vN" for
  Lindblad holds without the shift and not with it.
- **Reassigning `kerntype` carries over only `itype`**, as in QmeQ 1.1.
  `bandwidth` and `principal_part` resolved for one approach used to be
  inherited by the next although they mean different things there: a 1vN
  system at the default `itype=0` holds `principal_part='quad'`, so
  reassigning it to Lindblad switched on a quadrature Lamb shift. They are now
  re-derived for the new approach; a system reassigned to Lindblad has
  `principal_part` unset and `solve()` raises until it is set. Switching
  between an approach and its `py` twin keeps every option, and RTD, which
  reads `itype` directly, is unchanged.
- **Invalid input raises instead of selecting a default.** A misspelled
  `kerntype` ran Pauli, an unknown `indexing` ran `'charge'` (or `'Lin'` in
  `StateIndexing`), a misspelled `symmetry` such as `'Spin'` was ignored, and
  an out-of-range `itype` or `itype_ph` became 0, each with only a warning;
  `mfreeq=True` with RTD crashed in the kernel handler. All raise `ValueError`
  now, on construction and on assignment (`system.kerntype`, `itype`,
  `mfreeq`). Valid choices an approach cannot honour -- `'sz'` indexing for
  2vN, spin symmetry or non-charge indexing for RTD, an explicit `itype` other
  than 1 for RTD -- are still substituted with a `QmeqWarning`, as in 1.1.
- **Messages are warnings.** Warning-like prints in input validation, state
  indexing, solver fallbacks and failures, 2vN grid changes and both RTD
  energy-current implementations are typed `QmeqWarning`/`QmeqRuntimeWarning`
  warnings; RTDnoise no longer prints a failed kernel matrix. State display
  and build output are unchanged.
- **Assigned lead and bath arrays are properties.** `mulst`, `tlst`, `dlst`
  on the leads and `tlst_ph`, `dlst_ph` on the phonon baths are read like the
  constructor argument on assignment (a dictionary replaces the whole array),
  validated, and written into the stored array; a wrong shape raises
  `ValueError`. See Fixed for why.
- **Rename the RTD array `Lnn` to `Lnn_inv`** (and `add_element_Lnn` to
  `add_element_Lnn_inv`): it holds the inverse of the bare coherence splitting,
  not the coherence-sector Liouvillian. The two backends still store it in
  different shapes.
- **Python 3.11 or newer, and a modern build.** Static metadata lives in
  `pyproject.toml` with a dynamic version, automatic package discovery and
  `requires-python = ">=3.11"`; `setup.py` only builds the extensions,
  guarded by `if __name__ == '__main__'`. Extensions are generated with
  Cython 3 (`>=3.0,<4`) with explicit language level and exception semantics,
  always from the `.pyx`/`.pxd` sources, into `build/cython/`. OpenMP is
  optional and compiler-aware through `QMEQ_OPENMP=auto|on|off`, probing
  `/openmp`, `-fopenmp` and Apple clang's `-Xpreprocessor -fopenmp -lomp`, so
  `pip install .` works with an unmodified Apple toolchain; serial and threaded
  builds can differ in the last bits of reduced quantities. macOS wheels are
  built with Apple clang and without OpenMP (see Fixed); the Conda packages
  keep it. NumPy's `emath` namespace provides the complex logarithm, and the
  suite emits no deprecation warnings on NumPy 2.5 and SciPy 1.18.
- **CI.** The pure-Python suite runs on 3.11-3.14 and the compiled one on
  Linux, Windows and macOS (`test.yml`, formerly `test_cython.yml`, split into
  a single `python` job and a Cython-version matrix); the documentation builds
  strictly in CI; `slow.yml` runs the examples weekly as a compiled-backend
  gate (a pure-Python leg measured only how slow uncompiled 2vN/RTD sweeps
  are); `build_wheels.yml` uses cibuildwheel 4.2 for `cp311`-`cp314`, checks
  the tag against `qmeq.__version__`, and publishes from a separate job;
  `publish_conda.yml` is now `release.yml`. `example1c` is reported as a
  skipped example: its 81000-solve stability diagram exceeds any reasonable
  timeout, and `example1b` covers the same 2vN path.
- **Documentation** is MkDocs, built with `--strict` so any warning fails the
  build; the docstring rules this required are recorded in the conventions
  pages. Each approach's validity domain and known failure modes are collected
  in `docs/docs/guide/approaches.md`, which the RTD warnings point to.
  `INSTALL.md` names the ways to install this version -- the prefix.dev
  channel, the wheels and sdist on each GitHub release, or a source build --
  and warns that `pip install qmeq` still gives the upstream 1.1. It uses
  `pip install .` instead of the deprecated `python setup.py install`,
  documents `pytest --pyargs qmeq.tests` for an installed build, and with
  `README.md` links to the vendored `examples/`.
- **Internal clean-ups with no numerical effect**, the historical reference
  corpora reproducing unchanged on both backends: the "no index" sentinel is
  named `NO_INDEX` at 61 sites; the packed-real offset is one precomputed
  `imag_offset`; the `maptype` integers are gone from the kernel handlers and
  documented in `get_ind_dm0`; the pure-Python electron-phonon approaches no
  longer bind `si` to an object of a different class; mutable default
  arguments are `None`; `clean.py` resolves paths from its own location and
  gained `--dry-run`; pytest uses the native `[tool.pytest]` configuration;
  `pyRTD` and RTDnoise evaluate one shared enumeration of their population
  diagrams, `qmeq.approach.rtd_diagrams`, and assemble bitwise the same
  kernels.

### Removed

- The Sphinx documentation tree and its dependency, CI and packaging paths,
  superseded by MkDocs, and the outdated `README.rst`.
- The dead build path that reused checked-in C files, with its `--cython`
  flag; the unreachable `scipy.misc.factorial` fallback; the Homebrew-GCC
  symlink script for macOS wheels; and a shadowed duplicate of the pure-Python
  2vN `TermsCalculator2vN.iterate`.

### Performance

- Raise `MAX_CACHE`, the `lru_cache` bound on the memoised special functions,
  from 100 to 10000: a pure-Python RTD solve goes from a 75-81% to a 96-97% hit
  rate and runs about 2.2 times faster, for at most about 17 MB. The bound
  stays finite on purpose, since parameter sweeps generate fresh keys without
  end. Results are bitwise unchanged; the compiled backend is unaffected.
- Batch the 2vN Hilbert transforms in memory-bounded chunks instead of one FFT
  pair per density-matrix trace, and skip interpolation for exactly zero
  tunnelling products in the compiled 2vN iteration.
- Skip RTDnoise's redundant explicit partner traversal in second-order
  assembly.

### Fixed

Present in QmeQ 1.1:

- **Compiled electron-phonon Lindblad jump term.** Coherence columns paired
  `L[b, a]` with `conj(L[bp, a])` instead of `conj(L[bp, ap])`, so the
  generator did not preserve the trace: negative populations (down to -2.25
  with the default bath) and `I_L != -I_R`, while `pyLindblad` was right. The
  compiled kernel now equals the Python one to machine precision, the 1.1
  reference case that was a strict `xfail` passes, and a backend-parity and
  trace-preservation gate covers all four electron-phonon approaches. The
  legacy bundle's compiled `Lindblad22` entries recorded the error; they sit
  3e-12 from the corrected result, inside their test's tolerance, and keep a
  provenance note. The compiled `generate_fct` also built jump operators only
  for the state pairs of the `si_elph` layout, while the Python twin builds
  them for every same-charge pair; with `'sz'`/`'ssq'` indexing and a phonon
  coupling between states that layout does not pair, the backends differed by
  1e-4 in the kernel. They now agree exactly.
- **Assigning a lead or bath array was ignored by the compiled backend.** The
  compiled approaches bind views of those arrays when first prepared, and an
  assignment such as `system.mulst = values` replaced the array, so every
  later compiled solve kept the first values while the pure-Python approaches
  followed the assignment; in a single-level bias step the compiled current
  stayed at 0.0047 against 0.0168. Assignment now writes in place. `change()`
  and `add()` were never affected.
- **2vN conjugated `Phi[1]` terms.** `Phi[1](k)` is stored as a linear map
  on `Phi[0]`, and the iteration conjugated that map without transposing its
  `Phi[0]` index, using `Phi[0]_{bb'}` where `conj(Phi[0]_{bb'}) =
  Phi[0]_{b'b}` belongs. Every iteration after the first was therefore wrong
  whenever a same-charge coherence was complex in the eigenbasis, and the
  result depended on the arbitrary phases of the many-body eigenvectors (0.2%
  in the current and 7% in the energy current under a rephasing of one test
  model, not shrinking with `kpnt`). `generate_kern` already applied the
  transpose; the iteration now shares it on both backends. Populations-only
  models and `niter=1` are unchanged, and the kernel's population columns
  still reproduce QmeQ 1.1, whose 2vN stationary values record the error; a
  rephasing-covariance test gates the fix.
- **`BuilderManyBodyElPh` solves.** Its phonon state indexing `si_elph` was
  built for an empty Fock space before the many-body states were known and
  never rebuilt: Pauli, Redfield and 1vN raised `IndexError` on the Python
  backend and segfaulted on the compiled one, and compiled Lindblad returned
  wrong results without an error. `si_elph` now mirrors the many-body states,
  also after a `kerntype` reassignment, and the builder matches `BuilderElPh`
  given the same `Ea`, `Tba` and `Vbbp` exactly, on all four approaches and
  both backends.
- **`remove_states` and `use_all_states` on many-body input.** The state
  indexing of `BuilderManyBody` is created from `nsingle=0`, so its per-charge
  state lists described one empty sector: `remove_states` raised
  `IndexError`, and `use_all_states` silently kept only state 0. They now
  select from the many-body states and match the equivalent Fock-space model.
- **`BuilderManyBody` with compiled RTD** applied its many-body state indexing
  after the approach was built, so a per-thread kernel buffer was sized from a
  placeholder state count: wrong currents and, for larger systems,
  out-of-bounds writes that corrupted the heap. A `_init_before_appr` hook now
  finishes the state setup first, for the electron-phonon builders too, and
  tutorial 6 no longer needs its `pyRTD`-then-switch workaround.
- **Lead temperatures.** A zero or negative temperature made each approach
  fail its own way, some reporting `success=True` beside `nan` populations,
  and a negative one returned a confidently wrong current; a partial `tlst`
  left the unnamed leads at zero. `LeadsTunneling` now refuses them on
  construction, assignment, `add` and `change`, naming the entries; a refused
  update changes nothing, and an empty `tlst` is still accepted for systems
  that never solve.
- **RTD with complex tunnel amplitudes.** The two equal-temperature integral
  paths now share the analytic wide-band real part and keep the complementary
  part genuinely complex products need; such products are detected with a
  relative phase tolerance, so roundoff-scale imaginary parts no longer switch
  a diagram to a different approximation. The Ozaki expansion is sized from
  the widest lead rather than lead 0.
- **Approach classes as `kerntype`**, which the documentation offers, never
  worked: construction assigned `self.kerntype` before the approach existed,
  and `issubclass(value, Approach)` fails for every compiled approach. A class
  is now recognised by its `kerntype` name and validated under it; any other
  object raises `TypeError`.
- **`get_phi0` and `get_phi1` for Pauli** branched on `funcp.kerntype`, which
  the builder never sets: `get_phi0` on a Pauli coherence raised `IndexError`,
  and `get_phi1` did not return `None`.
- **`get_ind_dm0` with an unsupported `maptype`** returned `None`, which NumPy
  reads as `np.newaxis`, so a wrong selector reshaped an array far from its
  cause; it raises `ValueError`. Inserting a matrix element at an uncarried
  endpoint is a no-op in both kernel handlers instead of writing to the last
  row or column (no shipped caller did).
- **`itype=0` with a transition exactly on a band edge** passed SciPy's
  `Parameter 'wvar' must not equal integration limits` to the user, and the
  compiled path returned zeros; both now raise an error naming the energy, the
  edge and `dband`, since the principal value does not exist there.
- **pyRTD** cached `off_diag_corrections` at construction, which could desync
  from `funcp` and fail only on the pure-Python backend; it reads `funcp` as
  the compiled RTD does.
- **S_z-changing dot terms under `'sz'`/`'ssq'` indexing** failed deep in the
  Hamiltonian construction with `ValueError: 1 is not in list`, since those
  indexings build it block by block in S_z. The term is now refused up front,
  named, with `'charge'` indexing as the way out; a refused `change()` leaves
  the dot untouched.
- **`set_statesdm`** in `StateIndexingPauli` and `StateIndexingDMc` appended
  its sentinel sector to the caller's list instead of a copy, as
  `StateIndexingDM` already did, so `BuilderElPh.remove_states` left `si` and
  `si_elph` sharing one `statesdm` that grew an empty sector on every call.
- **Spin-symmetric input** compared strings with `is`, which failed for a
  `'spin'` built at runtime (from JSON or argparse).
- **The Bose functions** use `expm1` for accuracy near zero, and the
  electron-phonon forms are guarded against large arguments.

In features added since QmeQ 1.1 (development builds only):

- **RTDnoise.** Complex-amplitude second-order diagrams are completed from
  their inverted electron-hole and Keldysh partners (the old traversal
  conjugated single vertices, so particle-current errors fell back from cubic
  to quadratic in the coupling at generic flux); the first-order Laplace
  derivatives carry the `1/T_lead` of the scaled argument (the noise was not
  covariant under rescaling and wrong at every temperature but `T = 1`); the
  analytic population-coherence derivative keeps both channels of
  `pi*f + 1j*phi` with the right signs; the counting-resolved coherence
  correction's derivative is stored in the non-zero channel (it contributed
  nothing before); the free-coherence resolvent derivative uses the vertex
  blocks' Laplace orientation; the second-order differentiation step is unit
  covariant; and population/coherence pairs more than one electron apart are
  no longer rejected, which had made `off_diag_corrections=True` raise for
  dots with more than two levels. Each is gated by an independent reference;
  their effects on the noise are `O(Gamma**2)` to `O(Gamma**3)`.
- **Counting-statistics pseudoinverse.** The projected pseudoinverse behind
  every `current_noise` and `current_noise_matrix`, and RTDnoise's order
  decomposition, called `np.linalg.pinv` with its `rcond=1e-15`, while the
  nullity check certifies the stationary null space at `n*eps*sigma_max`.
  Above a few states the null singular value is roundoff between the two, and
  whether it fell above `1e-15` depended on the LAPACK path; when it did, its
  `1/sigma_null ~ 1/eps` triple survived the stationary projections at
  `O(1/sigma_max)`. In a spinful double dot under Lindblad this broke
  `S_L == S_R` and the zero row sums of the all-lead covariance matrix by up
  to 1e-3 relative, and put the two backends 2e-4 apart with the Lamb shift
  on. The pseudoinverse now excludes exactly the certified null triple, gated
  by a planted-singular-value test and an all-lead conservation test.
- **Lindblad Lamb shift.** The weight first paired QmeQ's dissipator with the
  Bloch-Redfield principal-value shift, and then had the spectral function and
  argument orientation of the correction swapped between the two intermediate
  charge sectors; both are corrected in the form described under Added.
  1.2.0.dev9-dev10 also switched the shift on by default (see Changed).
- **RTD coherence warning.** It estimated the damping from
  `2*pi*sum(|T|**2)`, which omits reservoir occupations and could warn deep in
  Coulomb blockade; it now uses the Fermi-weighted escape rates.
  `RTDCoherenceDiagnostics.gamma_upper_bound` keeps the old scale.
- **Legacy names** `qmeq.Builder_many_body` and `qmeq.Builder_elph`, which
  development builds had dropped, are restored.
- **The non-interacting NEGF test oracle** read QmeQ's `hsingle` and `tleads`
  conventions the wrong way round, which together is the time-reversed model
  (invisible to two-terminal currents, one order short in three-terminal RTD
  grading), and its `tleads` conversion conjugated the amplitudes. QmeQ's
  kernels were unaffected; the oracle's adapter is corrected and pinned by
  flux-sign and golden-rule convergence tests.
- **`Builder.get_phi0`** no longer asks `StateIndexingDMc` for a conjugation
  map it lacks.

Packaging, CI and documentation:

- The source distribution no longer ships the cythonize output (90% of the
  archive, 3.8 MB down to under 1 MB); `scipy` is a declared build
  requirement, so isolated PEP 517 builds work.
- macOS wheels could not be built, and then not installed on most Intel Macs:
  the Homebrew-GCC symlink broke, and its bundled `libgomp` forced a
  `macosx_15_0` deployment target. They are built with Apple clang and
  `QMEQ_OPENMP=off`. `c_RTD.pyx` reaches `omp_*` only through a guarded shim,
  so a non-OpenMP build no longer fails to link or imports with unresolved
  symbols.
- Retired and misconfigured CI pieces: the `macos-13` runner, `pixi install`
  in the Conda publish job, the undeclared `linux-aarch64` pixi platform, and
  a `setuptools` install that silently kept a runner's too-old version.
- The `docs` extra gains `ipython-pygments-lexers` and the `test` extra
  `setuptools>=77`; the legacy notebooks' images are tracked despite the
  `*.png` ignore rule; notebooks use a Python 3 kernel; ambiguous `Approach`
  cross-references and malformed docstrings no longer break a strict build;
  the documentation builds without the compiled extensions.
- Tests that were not portable: the QmeQ 1.1 electron-phonon kernel check
  compares under one consistent eigenvector sign gauge, the batched
  Hilbert-transform check allows FFT reassociation on aarch64, and the
  Cython build-directory probe skips when Cython is intentionally absent.

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
