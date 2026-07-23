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

Done recently (see the changelog): the newer `cvsvensson/qmeq` maintenance
history was merged; packaging moved to `pyproject.toml` with a `>=3.10` Python
floor and dependency extras (`test`, `docs`, `dev`); tutorials and example
scripts were vendored into [examples/](examples/), rendered in the Sphinx docs,
and run as tests; the docs build cleanly with warnings-as-errors; and backend
selection is now explicit through `QMEQ_BACKEND`.

Where it is going, roughly in priority order:

- **CI modernization** — run tests (including the example `--runslow` suite and
  the `-W` docs build) across supported Python versions and OSes, and build the
  Cython extensions and wheels in CI rather than only on release.
- **Portable extension builds** — exercise pure-Python and compiled jobs
  separately, make OpenMP optional and platform-aware, and validate installed
  wheels and source distributions.
- **Cython 3 standardization** — declare and test the supported Cython range,
  make compiler directives explicit, and audit exception/`nogil` semantics.
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
- [qmeq/_backend.py](qmeq/_backend.py) — centralized runtime backend selection,
  extension-group loading, and diagnostic status.
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

Backend selection is process-wide and is read before the first `import qmeq`:

- `QMEQ_BACKEND=auto` (default) uses Cython when the complete extension set is
  available and falls back quietly only when extensions are cleanly absent;
  broken or partial installations still fail.
- `QMEQ_BACKEND=python` forces the pure-Python runtime and skips extension
  builds.
- `QMEQ_BACKEND=cython` requires the compiled runtime and fails clearly if an
  extension is absent or broken.

Do not restore broad `ImportError` fallbacks around individual extensions.
They hide ABI and dependency failures and can produce a mixed installation.
Use the centralized loader in `qmeq._backend` for any new optional compiled
component. `qmeq.get_backend_status()` reports the selected implementation and
must agree with the classes exercised by backend tests.

## Build / test / docs

```bash
# Editable compiled install (needs Cython, a C compiler, and currently OpenMP)
QMEQ_BACKEND=cython pip install -e '.[dev]'
# Editable pure-Python install
QMEQ_BACKEND=python pip install -e '.[test]'

# Force regeneration of generated C from .pyx/.pxd
QMEQ_BACKEND=cython python setup.py build_ext --inplace --cython

# Run the fast suite against each backend in separate processes
QMEQ_BACKEND=python pytest qmeq/tests
QMEQ_BACKEND=cython pytest qmeq/tests
# Include the long-running example scripts and notebooks
QMEQ_BACKEND=python pytest qmeq/tests --runslow
QMEQ_BACKEND=cython pytest qmeq/tests --runslow

# Build HTML docs (Sphinx); needs the `docs` extra + the pandoc binary
cd docs
QMEQ_BACKEND=python sphinx-build -b html -W --keep-going source build/html

# Remove all build artifacts + generated .c/.so files
python clean.py
```

The example scripts write figures (`*.png`) and data (`*.dat`) into the working
directory; these are gitignored. `test_examples.py` runs them in a temp dir so
they never touch the tree — do the same if running examples manually, or expect
untracked output files.

Build details live in [setup.py](setup.py) (extension list, OpenMP flags) and
[pyproject.toml](pyproject.toml). The `.pyx`/`.pxd` files are the canonical
extension sources. Generated `.c` files are ignored and untracked; a clean
build regenerates them, while the legacy `--cython` option forces regeneration
when local generated files already exist. The roadmap tracks removal or
formalization of that legacy path.

The code currently builds and passes with Cython 3, but the build requirement
does not yet declare a minimum or tested range. Treat Cython 3 warning cleanup
as semantic work: do not add `noexcept` merely to silence a diagnostic when an
error needs to propagate.

## Conventions & gotchas

- Python 3.10 or newer; runtime dependencies are NumPy and SciPy.
- Use `doublenp` / `complexnp` from [qmeq/wrappers/mytypes.py](qmeq/wrappers/mytypes.py)
  for array dtypes rather than hard-coding, to stay consistent with the Cython side.
- Legacy class aliases (`Builder_many_body`, `Builder_elph`) are kept for
  backwards compatibility — don't remove them.
- If you change a `.pyx`/`.pxd`, force regeneration and verify behavior against
  the pure-Python twin. Generated `.c` and platform binaries are build
  artifacts, not reviewable source changes.
- `QMEQ_BACKEND` is captured on first import. Use separate processes when
  testing more than one backend; changing the environment afterward does not
  switch an imported package.
- A successful native build and a green compiled suite are separate gates.
  Confirm the active implementation with `qmeq.get_backend_status()`.
- Example scripts can produce ignored figures and data in their working
  directory; run them through `test_examples.py` or from a temporary directory.
- Known maintenance work and priorities are tracked in [TODO.md](TODO.md);
  user-facing changes go in [CHANGELOG.md](CHANGELOG.md) under `[Unreleased]`.

## Before finishing a change

- Run the fast suite with the backend relevant to the change.
- If you touched backend routing, a `.py`/`.pyx` pair, `.pyx`/`.pxd`, numeric
  types, build configuration, or package contents, run both forced backends in
  separate processes and confirm the reported backend.
- For compiled changes, regenerate the extensions from source before testing;
  do not rely on stale `.c` or `.so` files.
- Build docs with `-W --keep-going` after documentation or import-surface
  changes.
- For packaging changes, inspect a clean wheel and source distribution and test
  the installed artifacts, not only the working tree.
- Update [CHANGELOG.md](CHANGELOG.md) for user-visible changes.
