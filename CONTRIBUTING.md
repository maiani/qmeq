# Working on QmeQ

This guide is for whoever maintains or changes QmeQ. It covers the
development setup, the code layout, the rules that keep the physics and the two
backends honest, the test gates, and the release procedure. Installing QmeQ to
use it is described in [INSTALL.md](INSTALL.md).

QmeQ is in maintenance. A change fixes a shipped feature, keeps the package
working with new Python, NumPy, SciPy or Cython releases, or keeps CI and
packaging working. New approaches, observables and public API are out of
scope.

## Setting up

```bash
git clone https://github.com/qmeq/qmeq.git
cd qmeq
pip install -e ".[dev]"                # tests, docs, Ruff, Cython, build, twine
python setup.py build_ext --inplace    # compile the extensions in place
python -c "import qmeq; print(qmeq.get_backend_status())"
```

Python 3.12 or newer is required. Building the extensions needs a C compiler
(see [INSTALL.md](INSTALL.md#c-compiler)); the generated C goes to
`build/cython/` and the shared libraries next to their `.pyx` sources. Rebuild
after every change to a `.pyx` or `.pxd` file: an editable install does not
recompile on import.

Two environment variables control the build and the run:

- `QMEQ_BACKEND=auto|python|cython` selects the implementation. It is read once,
  at the first `import qmeq`, and holds for the whole process. `auto` uses the
  compiled backend only when the complete extension set imports; `cython`
  fails if it does not; `python` forces the pure-Python twins.
  `qmeq.get_backend_status()` reports what is active.
- `QMEQ_OPENMP=auto|on|off` controls OpenMP at build time. `auto` probes the
  compiler and builds serially if it cannot. A serial build gives the same
  results up to the last bits of reduced quantities.

`python clean.py` removes caches and the in-place extensions;
`--caches-only` keeps the extensions and `--dry-run` lists what would go.

## Layout

| Path | Contents |
|---|---|
| `qmeq/builder/` | `Builder` and its variants: the public entry point, input validation, transport options |
| `qmeq/qdot.py`, `qmeq/leadstun.py`, `qmeq/baths.py` | dot Hamiltonian, lead tunnelling and phonon baths |
| `qmeq/indexing.py` | many-body state indexing (`Lin`, `charge`, `sz`, `ssq`) |
| `qmeq/approach/base/` | the approaches, each `name.py` with a compiled `c_name.pyx` twin except `RTDnoise.py` |
| `qmeq/approach/elph/` | electron-phonon variants of the first-order approaches |
| `qmeq/approach/` | shared machinery: kernel handlers, `dm_layout` (the packed density-matrix layout), `counting` (counting statistics), `diagnostics` (warnings and stationary-state checks), `rtd_diagrams` and `rtd_blocks` (RTD diagram enumeration and coherence blocks) |
| `qmeq/specfunc/` | special functions and lead integrals, with compiled twins |
| `qmeq/_backend.py` | backend selection |
| `qmeq/tests/` | the test suite and its reference bundles under `data/` |
| `docs/` | the MkDocs documentation |
| `examples/` | tutorials, example scripts and appendix notebooks |
| `scripts/reference_data/` | maintainer-only generators of the reference bundles |
| `recipe/` | the Conda recipe |
| `.github/` | CI and release workflows, and the artifact-inventory check |

## Rules for changing the code

### Preserve the physics

QmeQ's approaches are approximations and are expected to disagree outside
their validity regimes. Do not make results agree by weakening tolerances,
renormalising outputs, hiding warnings, or changing a convention without a
derivation. Diagnose a difference at the smallest observable layer available:
kernel blocks, then the stationary state, then currents, then derived
quantities.

For a numerical change:

- write down units, signs, index orientation, conjugation and normalisation;
- keep structural identities, historical regression values and independent
  analytic or exact checks apart;
- keep order-resolved quantities when agreement of a sum could hide a
  cancellation;
- exercise nontrivial controls, such as unequal couplings, physical complex
  phases, degeneracies and limiting scalings; and
- document the validity domain and the known failure mode, not only the
  working case.

A non-obvious formula, diagram rule or numerical construction taken from the
literature gets a nearby comment or docstring with a stable key from
[REFERENCES.md](REFERENCES.md) and the equation or section, for example
`[Emary2009, Eqs. (40)-(41)]`. Explain any local sign, normalisation or index
mapping rather than implying the code transcribes the paper. Verify a new
bibliographic entry before using it. Routine code needs no citation.

### Keep the two backends together

Most hot paths exist twice, as a pure-Python `name.py` and a compiled
`c_name.pyx`. Both must behave identically and cite the same sources, and a
change to one is a change to both. The pairs are the approaches in
`qmeq/approach/base/` and `qmeq/approach/elph/`, `aprclass` and
`kernel_handler` in `qmeq/approach/`, and `specfunc` and `specfunc_elph` in
`qmeq/specfunc/`.

The RTD population diagrams have one enumeration in Python,
`qmeq/approach/rtd_diagrams.py`, which `pyRTD` and RTDnoise evaluate. The
compiled `c_RTD.pyx` enumerates the same diagrams with its own loops, so a
change to the enumeration must be made there too;
`qmeq/tests/test_rtd_diagrams.py` holds the two together.

Backend routing goes through `qmeq._backend` only. Do not add per-module
`ImportError` fallbacks. `.pyx` and `.pxd` files are the source; generated
`.c` files, shared libraries and build directories are not. OpenMP is reached
only through the guarded `QMEQ_OMP_THREAD_NUM` shim in `c_RTD.pyx`, never by
calling `omp_*` directly, so that a serial build still compiles and links.

### Treat reference data as immutable

Reference bundles live under `qmeq/tests/data/<bundle>/` as validated JSON/NPZ
pairs and are loaded through `qmeq/tests/reference_data.py`.

- Tests never regenerate or overwrite expected values.
- Generators live in `scripts/reference_data/` and require an explicit output
  location.
- A historical bundle names its exact source revision, generator, model
  inputs, array schema and trust classification, and is generated only from
  that pinned source checkout.
- A snapshot of the current tree characterises it; it does not prove it
  correct. Correctness claims rest on analytic limits, exact solvers,
  conservation laws or independently derived results.
- When pinned arrays must change, the commit explains the physical or
  convention change that requires it. A fixture is never refreshed to make a
  failing test pass.

### What a test asserts

A test asserts an invariant or an independently known value, never only the
code's present output. Tolerances are justified by the numerics, not widened
until a test passes. A test that would also pass on the broken code needs a
control that shows it can fail.

## Test gates

The fast gates run on every push and pull request:

```bash
ruff check .
QMEQ_BACKEND=python pytest qmeq/tests     # about 10 minutes
QMEQ_BACKEND=cython pytest qmeq/tests     # about 3 minutes
```

Run both backends after changing a Python/Cython pair, backend routing,
numerical types, the build, packaged contents or the reference
infrastructure. Run them one after the other in a given checkout:
`test_backend.py` renames a compiled module for a moment, which breaks a run of
the other suite at the same time. A test that needs a different backend
starts a fresh process; it never changes `QMEQ_BACKEND` after the import.

The example scripts and notebooks are slow, compiled-backend tests, and CI
runs them weekly:

```bash
QMEQ_BACKEND=cython pytest qmeq/tests/test_examples.py --runslow -m "example and not notebook"
QMEQ_BACKEND=cython pytest qmeq/tests/test_examples.py --runslow -m notebook
```

They are second-order parameter sweeps, so running them uncompiled measures the
uncompiled kernels rather than the examples. Running them through pytest keeps
their figures and data in a temporary directory.

After a docstring or documentation change, build the documentation as CI does
(see [docs/README.md](docs/README.md)):

```bash
QMEQ_BACKEND=python mkdocs build --strict -f docs/mkdocs.yml
```

For packaging changes, a working-tree pass does not qualify the artifacts.
Build the wheel and sdist, check their contents, install each outside the
source tree, confirm the backend and run the installed tests:

```bash
python -m build
python .github/scripts/check_artifact_inventory.py dist/
python -m twine check dist/*
pytest --pyargs qmeq.tests        # from an environment with the installed artifact
```

Validate workflow changes with `actionlint` when it is available.

## Documentation

Documentation is part of a behaviour change:

- public API and parameter semantics: NumPy-style docstrings in the source;
- user workflows and the validity of each approach: `docs/docs/guide/`;
- derivations: `docs/docs/theory/`;
- internal index, sign, layout and sentinel contracts:
  `docs/docs/conventions/`; and
- user-visible changes: `CHANGELOG.md`, under `[Unreleased]`.

Comments, docstrings and documentation state what the code does, in the
present tense. The history of a change belongs in its commit message.

## Before a change is merged

- `git diff --check`, the focused tests, and every gate the changed files
  trigger;
- no test regenerated its own expected data; and
- the documentation and `[Unreleased]` are updated when behaviour visible to
  users changed.

## Releasing

The version is `__version__` in `qmeq/__init__.py`. A release is a tag
`v<version>` on the tested commit; both release workflows refuse a tag that
does not match the version.

Pushing the tag starts two workflows:

- `build_wheels.yml` builds wheels for CPython 3.12-3.14 on Linux, Windows and
  Intel and Apple Silicon macOS, and the sdist. It tests each after
  installation and attaches them to the GitHub release. macOS wheels are built
  without OpenMP so that they install on older macOS releases.
- `release.yml` builds the Conda packages from `recipe/`, tests the installed
  packages, and publishes them to the prefix.dev channel `andmai/science`
  through trusted publishing. Pre-release tags build `linux-64` only; final
  tags also build `linux-aarch64` and `osx-arm64`. A manual run with
  `full-matrix: true` builds every platform for a pre-release.

A release is ready when:

- the fast suites pass on both backends across the CI matrix;
- the slow example and notebook suites pass (`slow.yml`);
- the documentation builds strictly from a clean checkout;
- the wheel and sdist contents have been checked, and both pass their tests
  after installation outside the source tree on both backends;
- `[Unreleased]` in `CHANGELOG.md` has become one coherent section for the
  release;
- the package, documentation and tag versions agree; and
- the artifacts come from the tested commit and are published only after
  these checks pass.
