# QmeQ documentation

This is QmeQ's documentation tree, built with MkDocs + Material. It replaces
the Sphinx documentation that used to live at this path; that tree now lives
at [`legacy_docs/`](../legacy_docs/) and still builds warning-clean in CI.

**This tree is incomplete.** It grew out of an internal-conventions notebook
and has a full [Conventions](docs/conventions/index.md) section, plus a
started [Guide](docs/guide/index.md) section for user-and-API-facing material.
The user manual, complete API reference, theory pages, and worked examples
that `legacy_docs/` carries have **not** all been migrated here yet. If you want
those today, go to `legacy_docs/` (build it with Sphinx, see below) or read the
docstrings directly in `qmeq/`.

## Status

Incomplete by design. Each page states what has been verified and what has
not, using the same Verified / Stated / Open discipline described in
[docs/index.md](docs/index.md#how-this-is-written). Absence of a page means
nobody has written that material here yet, not that it doesn't exist — check
`legacy_docs/` and the docstrings in the meantime.

## Building

```bash
pip install -e ".[docs]"          # mkdocs + mkdocs-material, from pyproject.toml
mkdocs serve -f docs/mkdocs.yml   # live preview on localhost:8000
mkdocs build --strict -f docs/mkdocs.yml   # static site into docs/site/
```

`--strict` turns a broken internal link or a nav entry with no matching page
into a build failure; CI runs the build this way.

## Building the legacy Sphinx tree

```bash
pip install -e ".[docs-sphinx]"
cd legacy_docs
QMEQ_BACKEND=python sphinx-build -b html -W --keep-going source build/html
```
