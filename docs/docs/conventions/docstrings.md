# Docstrings and generated API documentation

QmeQ uses NumPy-style source docstrings. Mkdocstrings reads those docstrings
to build the [API reference](../api/index.md), so public parameter semantics
belong beside the implementation instead of being copied into a second manual.

## Portable conventions

- Use NumPy `Parameters`, `Returns`, `Raises`, and `Attributes` sections.
- Prefer plain, fully qualified names when an automatic cross-reference would
  make the source harder to read.
- Put large tables, derivations, and diagrams in the appropriate guide,
  theory, or conventions page and link to them from the docstring.
- Wrap index and call expressions in backticks: `` `szlst[charge][sz]` ``,
  `` `Phi[1](k)` ``. Bare, a bracketed name is read as a Mkdocstrings
  shorthand cross-reference, and `name[1](k)` is read as a Markdown link with
  `k` as its target; both then fail the strict build.
- Indent a parameter's continuation lines four spaces past the parameter name.
  Griffe warns about three, which is the most common accidental width.
- For a documented module variable, place a string literal immediately after
  the assignment; Griffe and editor tooling can read that form.

```python
NO_INDEX = -1
"""Sentinel returned when a density-matrix element has no packed index."""
```

The source still contains some reStructuredText-style inline math and roles.
Mkdocstrings renders their surrounding NumPy sections, but new or edited
docstrings should use Markdown-compatible inline notation where practical.

## Verification

After changing public docstrings, build the site from the repository root:

```bash
QMEQ_BACKEND=python mkdocs build --strict -f docs/mkdocs.yml
```

`--strict` turns every warning into a failure, so an unresolved reference or an
unparsable docstring fails the build rather than degrading a page quietly. The
documentation CI job runs the same command.
