# QmeQ documentation

This is QmeQ's documentation tree, built with MkDocs + Material. It contains
the user guide, tutorials, theory notes, generated API reference, and internal
implementation conventions.

## Status

Each page states what has been verified and what remains open. API details are
generated from the source docstrings, which remain the source of truth.

## Building

```bash
pip install -e ".[docs]"          # mkdocs + mkdocs-material, from pyproject.toml
mkdocs serve -f docs/mkdocs.yml   # live preview on localhost:8000
mkdocs build --strict -f docs/mkdocs.yml   # static site into docs/site/
```

`--strict` fails the build on any warning: a broken internal link, a nav entry
with no matching page, an unresolved mkdocstrings reference, or a docstring
Griffe cannot parse. CI runs the same command.
