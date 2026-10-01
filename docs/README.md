# QmeQ documentation

This is QmeQ's documentation tree, built with MkDocs + Material. It contains
the user guide, tutorials, theory notes, generated API reference, and internal
implementation conventions.

## Where things go

| content | location |
|---|---|
| workflows and the validity of each approach | `docs/guide/` |
| derivations | `docs/theory/` |
| internal index, sign, layout and sentinel contracts | `docs/conventions/` |
| public API and parameter semantics | NumPy-style docstrings in the source, rendered under `docs/api/` |
| tutorials and examples | `examples/`, rendered through `docs/notebooks` |
| user-visible changes | `CHANGELOG.md` |

The docstrings are the source of truth for the API.

## Building

```bash
pip install -e ".[docs]"          # mkdocs + mkdocs-material, from pyproject.toml
mkdocs serve -f docs/mkdocs.yml   # live preview on localhost:8000
QMEQ_BACKEND=python mkdocs build --strict -f docs/mkdocs.yml   # into docs/site/, as CI does
```

`--strict` fails the build on any warning: a broken internal link, a nav entry
with no matching page, an unresolved mkdocstrings reference, or a docstring
Griffe cannot parse. CI runs the same command.
