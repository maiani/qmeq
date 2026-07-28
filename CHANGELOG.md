# QmeQ Changelog

## [Unreleased]

### Added

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
- Add a six-notebook tutorial path in `examples/tutorials/`, each notebook
  stating a prediction, building the smallest useful model, and asserting
  physical and numerical checks: a first sequential-transport calculation,
  Coulomb blockade in the Anderson model, bias-gate stability diagrams,
  coherence in a double dot with the Pauli/Lindblad/Redfield/1vN comparison,
  energy and heat transport with thermovoltage and unit conversions, and
  cotunnelling with the RTD and 2vN approaches including a quantum-dot heat
  engine and many-body input. The path covers the material of the legacy
  tutorials.
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
  discovery, declare a supported-Python range of `>=3.10`, and reduce `setup.py`
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

### Fixed

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

### Removed

- Remove the outdated `README.rst`; `README.md` is now the canonical README and
  is used as the package long description.

### Fixed

- Use `expm1` in the pure-Python and Cython Bose functions for accuracy near
  zero, and guard the electron-phonon forms against large positive arguments.
- Define all compiled special-function names through pure-Python fallbacks when
  the Cython extensions are unavailable.
- Fix string comparisons used when expanding spin-symmetric input data.
- Fix Sphinx configuration, duplicate API metadata, malformed docstring
  references, and wrapper-page headings so documentation builds cleanly.

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
