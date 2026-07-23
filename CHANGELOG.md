# QmeQ Changelog

## [Unreleased]

### Added

- Add a prioritized maintenance roadmap in `TODO.md`.
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
