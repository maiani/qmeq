# AGENTS.md

Operational guidance for AI agents working in QmeQ. For the project overview,
features, installation, examples, authorship, and citation information, use
[README.md](README.md), [INSTALL.md](INSTALL.md), and
[AUTHORS.md](AUTHORS.md); do not duplicate them here.

## Start with the sources of truth

- [TODO.md](TODO.md) owns priorities and open implementation work.
- [CHANGELOG.md](CHANGELOG.md) owns user-visible changes under `[Unreleased]`.
- [REFERENCES.md](REFERENCES.md) owns stable keys for external literature used
  by the implementation. The README remains the source for citing QmeQ itself.
- Root `*_devplan.md` files are temporary coordination documents. They may
  guide implementation, but production code, tests, fixtures, and permanent
  documentation must state durable conventions and provenance directly rather
  than link to a development plan or copy its phase labels.
- [docs/README.md](docs/README.md) describes documentation ownership and build
  status. The successor tree is [docs/](docs/); [legacy_docs/](legacy_docs/) is
  retained temporarily and must not receive new material.

Read the relevant source and tests before editing. Treat line numbers and status
claims in plans as snapshots: verify them against the working tree.

## Preserve scientific meaning

QmeQ's approaches are approximations and are expected to disagree outside
their validity regimes. Do not make results agree by weakening tolerances,
renormalizing outputs, hiding warnings, or changing a convention without a
derivation. Diagnose differences at the smallest observable layer available:
kernel blocks, stationary state, current, then derived quantities.

For numerical changes:

- write down units, signs, index orientation, conjugation, and normalization;
- separate structural identities from historical regression values and from
  independent analytic or exact checks;
- preserve order-resolved quantities when agreement of a sum could hide a
  cancellation;
- exercise nontrivial controls such as unequal couplings, physical complex
  phases, degeneracies, and limiting scalings where relevant; and
- document the validity domain and known failure mode, not only the happy path.

For non-obvious formulas, diagram rules, or numerical constructions derived
from the literature, add a nearby source comment or docstring using a stable
key from `REFERENCES.md` and the relevant equation or section, for example
`[Emary2009, Eqs. (40)-(41)]`. Explain any local sign, normalization, or index
mapping instead of implying that adapted code is a verbatim transcription.
Verify new bibliographic entries before using them; do not guess equation
numbers or add vague author-year comments. Paired Python/Cython implementations
must cite the same source. Avoid citations on routine code where they add no
scientific provenance.

## Reference data are immutable test inputs

Reference bundles live under `qmeq/tests/data/<bundle>/` as validated JSON/NPZ
pairs and are loaded through `qmeq/tests/reference_data.py`.

- Routine tests must never regenerate or overwrite expected values.
- Generators belong in `scripts/reference_data/` and must require an explicit
  output location.
- Historical bundles must name an exact source revision, generator, model
  inputs, array schema, and trust classification. Generate them only from the
  pinned pristine source checkout.
- A current-tree snapshot is a characterization fixture, not independent proof
  of correctness. Use analytic limits, exact solvers, conservation laws, or
  independently derived results for correctness claims.
- When arrays change, explain which physical or convention change requires it;
  never refresh a fixture merely to make a failing test green.

## Python and Cython are separate gates

Hot paths often have a pure-Python `.py` implementation and a compiled `c_*.pyx`
twin. Keep them behaviorally consistent. `.pyx` and `.pxd` files are canonical;
generated `.c`, shared libraries, and build directories are not reviewable
source changes.

`QMEQ_BACKEND` is process-wide and read before the first `import qmeq`:

- `python` forces the pure-Python implementation;
- `cython` requires the complete compiled implementation; and
- `auto` selects Cython only when the extension set is complete.

Never restore per-module broad `ImportError` fallbacks; use `qmeq._backend` for
backend routing. Test forced backends in separate processes and confirm the
active implementation with `qmeq.get_backend_status()`. A successful build does
not prove that tests imported the compiled classes.

When changing `.pyx` or `.pxd`, regenerate and rebuild before testing. Keep
OpenMP optional through the existing `QMEQ_OPENMP=auto|on|off` machinery; do
not call `omp_*` directly from Cython outside the guarded shim.

## Testing workflow

Start focused, then expand in proportion to risk. The standard fast gates are:

```bash
ruff check .
QMEQ_BACKEND=python pytest qmeq/tests
QMEQ_BACKEND=cython pytest qmeq/tests
```

Run both forced backends after changing backend routing, paired Python/Cython
logic, numerical types, build configuration, packaged contents, or shared
reference infrastructure. Tests that switch backends must spawn fresh
processes rather than mutate the environment after import.

Examples and notebooks are explicit slow tests:

```bash
QMEQ_BACKEND=python pytest qmeq/tests/test_examples.py --runslow -m "example and not notebook"
QMEQ_BACKEND=python pytest qmeq/tests/test_examples.py --runslow -m notebook
```

Run examples through their tests so generated figures and data stay in a
temporary directory. For packaging work, build wheel and sdist, inspect their
contents, install each outside the source tree, confirm backend status, and run
the installed-copy tests. A working-tree pass is not an artifact qualification.

## Documentation routing

Documentation is part of a behavior change:

- public API and parameter semantics: NumPy-style source docstrings;
- user workflows and approach validity: `docs/docs/guide/`;
- derivations: `docs/docs/theory/`;
- internal index, sign, layout, and sentinel contracts:
  `docs/docs/conventions/`;
- findings, open questions, and planned work: `TODO.md` or the relevant
  root development plan; and
- user-visible release notes: `CHANGELOG.md`.

Do not leave project-management notes in production source or user pages. A
reader should learn what the code does, not which temporary pass discovered it.

Do not add new content to `legacy_docs/`.

After a docstring or documentation edit, run the builds used by CI:

```bash
cd legacy_docs
QMEQ_BACKEND=python sphinx-build -b html -W --keep-going source build/html
cd ..
mkdocs build -f docs/mkdocs.yml
```

## Worktree and finish discipline

Assume uncommitted changes belong to the user. Before editing or staging:

- confirm the repository top level;
- inspect staged and unstaged changes separately;
- preserve unrelated edits, generated artifacts, and intentional deletions;
- stage explicit relevant paths only; and
- do not commit, tag, push, or publish without the corresponding user approval.

Before handing off a change:

- run `git diff --check` and the focused tests;
- run the backend, documentation, packaging, or slow-test gates triggered by
  the files changed;
- validate workflow changes with `actionlint` when available;
- confirm that no test regenerated its own expected data;
- inspect the final staged diff and list anything intentionally left unstaged;
  and
- update durable documentation and `[Unreleased]` when behavior visible to
  users changed.
