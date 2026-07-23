# AGENTS.md

Guidance for AI agents working in the QmeQ repository.

## What this is

QmeQ is a Python package for calculating electron transport (particle and
energy currents) through quantum-dot devices described by Anderson-type models,
using approximate density-matrix / quantum master equation methods. It is a
scientific library — correctness of the physics matters more than convenience,
and results from different approximations are expected to disagree (see the
"Physics disclaimer" in [qmeq/__init__.py](qmeq/__init__.py)).

Implemented approaches: Pauli, Lindblad, Redfield, first-order von Neumann
(1vN), second-order von Neumann (2vN), and Real Time Diagrammatics (RTD).
First-order methods also have electron-phonon variants.

## Current focus & direction

The active effort is **modernization and maintenance**, not new physics. The
prioritized roadmap lives in [TODO.md](TODO.md) and user-facing changes go in
[CHANGELOG.md](CHANGELOG.md) under `[Unreleased]`; read both before starting —
they are the source of truth for what is in flight.

Done recently (see the changelog): packaging moved to `pyproject.toml` with a
`>=3.10` Python floor and dependency extras (`test`, `docs`, `dev`); the
tutorials and example scripts were vendored into [examples/](examples/), are
rendered in the Sphinx docs, and now run as tests; and the docs build cleanly
with warnings-as-errors.

Where it is going, roughly in priority order:

- **CI modernization** — run tests (including the example `--runslow` suite and
  the `-W` docs build) across supported Python versions and OSes, and build the
  Cython extensions and wheels in CI rather than only on release.
- **Exercise both code paths** — make sure tests cover the compiled Cython
  modules *and* the pure-Python fallbacks, not silently just one.
- **Runtime maintenance** — numerical stability (e.g. the el-ph Bose function),
  replacing mutable default arguments, converting stray `print`s to warnings,
  and clearing deprecated NumPy/SciPy/Cython APIs.

Guiding principle: this is a scientific library — **physics correctness comes
before convenience**, and different approximations are expected to disagree.
When in doubt, prefer a change that is easy to verify against the existing
tests and reference data over a cleverer one that is not.

## Layout

- [qmeq/builder/](qmeq/builder/) — user-facing entry points. `Builder` is the
  main class; `BuilderManyBody` / `BuilderElPh` handle many-body input and
  electron-phonon coupling. Start here to understand the public API.
- [qmeq/approach/](qmeq/approach/) — the master-equation solvers.
  - [qmeq/approach/base/](qmeq/approach/base/) — the six core approaches.
  - [qmeq/approach/elph/](qmeq/approach/elph/) — electron-phonon variants.
  - `aprclass.py` (`Approach`, `ApproachElPh`, `ApproachBase2vN`) and
    `kernel_handler.py` are the shared machinery.
- [qmeq/specfunc/](qmeq/specfunc/) — special functions used by the kernels.
- [qmeq/wrappers/](qmeq/wrappers/) — LAPACK wrappers and shared numeric types
  (`mytypes.py`: `doublenp`, `complexnp`).
- `qdot.py`, `indexing.py`, `leadstun.py`, `baths.py` — quantum-dot
  Hamiltonian, Fock-state indexing, lead tunneling, phonon baths.
- [qmeq/tests/](qmeq/tests/) — pytest suite (`test_*.py`); `data_*.py` hold
  reference values. `test_examples.py` runs the vendored examples (see below).
- [examples/](examples/) — vendored learning material: `scripts/` (runnable
  `.py`), `tutorial/` and `appendix/` (notebooks). Rendered into the docs via
  nbsphinx-link (`docs/source/examples/*.nblink`); not shipped in the package.

## Cython / pure-Python duality (important)

Many hot-path modules exist in two forms: a pure-Python `.py` and a Cython
`.pyx` (with a `.pxd` header), the latter prefixed `c_` — e.g.
`base/pauli.py` and `base/c_pauli.pyx`. The compiled versions are optional:
the package **imports and runs without the extensions built**, falling back to
pure Python. When editing a `.py` that has a `c_` twin, keep the two
implementations behaviorally consistent, and remember tests may be exercising
only the fallback if extensions aren't compiled.

## Build / test / docs

```bash
# Editable install with compiled extensions (needs Cython + a C compiler)
pip install -e .
# With tooling extras: test deps, doc deps, or everything (test+docs+cython+build)
pip install -e '.[test]'   # or .[docs], or .[dev]
# Force regeneration of .c files from .pyx (otherwise checked-in .c is reused)
python setup.py build_ext --inplace --cython

# Run the test suite (fast examples run; long 2vN/RTD ones are skipped)
pytest qmeq/tests
# Include the long-running example scripts and notebooks
pytest qmeq/tests --runslow

# Build HTML docs (Sphinx); needs the `docs` extra + the pandoc binary
cd docs && make html                              # output in docs/build/html/index.html
sphinx-build -b html -W --keep-going source build/html   # warnings-as-errors, as CI should

# Remove all build artifacts + generated .c/.so files
python clean.py
```

The example scripts write figures (`*.png`) and data (`*.dat`) into the working
directory; these are gitignored. `test_examples.py` runs them in a temp dir so
they never touch the tree — do the same if running examples manually, or expect
untracked output files.

Build details live in [setup.py](setup.py) (extension list, OpenMP flags) and
[pyproject.toml](pyproject.toml). By default the build reuses checked-in
Cython-generated `.c` files; pass `--cython` to regenerate from `.pyx`.

## Conventions & gotchas

- Python 3 only (>=3.6); depends on NumPy and SciPy.
- Use `doublenp` / `complexnp` from [qmeq/wrappers/mytypes.py](qmeq/wrappers/mytypes.py)
  for array dtypes rather than hard-coding, to stay consistent with the Cython side.
- Legacy class aliases (`Builder_many_body`, `Builder_elph`) are kept for
  backwards compatibility — don't remove them.
- If you change a `.pyx`/`.pxd`, the checked-in `.c` becomes stale; rebuild with
  `--cython` and verify behavior against the pure-Python twin.
- Known maintenance work and priorities are tracked in [TODO.md](TODO.md);
  user-facing changes go in [CHANGELOG.md](CHANGELOG.md) under `[Unreleased]`.

## Before finishing a change

- Run `pytest qmeq/tests` and make sure it passes.
- If you touched a `c_*` extension, rebuild it and confirm both the compiled and
  pure-Python paths still work.
- Update [CHANGELOG.md](CHANGELOG.md) for user-visible changes.
