# Docstrings and the Sphinx exit

Sphinx itself is planned for removal, but has not happened yet: the directory
move landed first (the Sphinx tree is now [`legacy_docs/`](../../../legacy_docs/)
and this MkDocs tree is `docs/`), and `legacy_docs/` is still the CI gate that
builds warning-clean. Removing Sphinx as a dependency requires migrating the
docstring markup described below, plus the API reference and rendered
examples in [Beyond docstrings](#beyond-docstrings). This page records what
that will and will not touch, so work done in the meantime does not deepen
the hole.

## Current state

- `legacy_docs/source/conf.py` enables `autodoc`, `napoleon`, `mathjax`,
  `nbsphinx` and `nbsphinx_link`.
- Autodoc covers exactly **four** modules: `qmeq.indexing`, `qmeq.qdot`,
  `qmeq.baths`, `qmeq.leadstun`. Nothing under `qmeq/approach/` is in the API
  docs, which is why the layout specification had to be written by hand.
- The build runs with `-W --keep-going`, so a malformed docstring in one of
  those four modules is a build failure, not a warning.
- `default_role` is unset, so a single backtick falls back to
  `title-reference` and renders as italics rather than code.

## What survives the switch, and what does not

`napoleon` and `griffe` (the parser behind `mkdocstrings`) both read
**NumPy-style sections**, so the structural part of every docstring is already
portable. What breaks is inline reST.

| markup | portable? | count in `qmeq/` |
|---|---|---|
| NumPy `Parameters` / `Returns` sections | **yes** | 17 files |
| `` :math:`...` `` | no — Markdown wants `$...$` | 175 hits, 9 files |
| `:mod:` `:meth:` `:class:` `:data:` `:func:` | no — become cross-reference syntax or plain links | 34 hits |
| reST simple tables (`==== ====`) | no — Markdown wants pipe tables | 9 lines, 1 file |
| reST directives (`.. note::`) | no — Material wants `!!! note` | 5 hits |

So the migration is roughly **220 inline markers across ~10 files**, almost all
of it `:math:`. That is a mechanical pass, not a rewrite — and it is small
because the sections, which are the hard part, already transfer.

## Policy while Sphinx is still the builder

**Keep one convention.** Write reST-flavoured docstrings everywhere, including
in modules that autodoc does not currently reach. The temptation is to write
Markdown-first in `qmeq/approach/`, since nothing parses it today and it would
be future-proof — but mixed conventions inside one package cost more
readability now than the pass they would save later, and the whole point of this
work is legibility.

**Prefer the portable subset where it costs nothing.** NumPy sections always.
Plain prose over a cross-reference role when the bare name is just as clear:
write `see StateIndexingDM.get_ind_dm0` rather than
`` :meth:`StateIndexingDM.get_ind_dm0` `` unless the link genuinely earns its
keep.

**Put tables and diagrams here, not in docstrings.** A reST table in a docstring
is markup that will need converting; the same table on one of these pages is
already Markdown. Docstrings can point at a page.

!!! warning "Build the docs after touching one of the four autodoc modules"
    `qmeq/indexing.py` is in autodoc and builds under `-W`. A malformed table or
    role there fails the build. Verify with:

    ```bash
    cd legacy_docs
    QMEQ_BACKEND=python sphinx-build -b html -W --keep-going source build/html
    ```

## Beyond docstrings

Dropping Sphinx also drops two things that are not docstring markup:

- **The API reference itself.** `mkdocstrings[python]` is the replacement and
  reads the NumPy sections directly. It would also, for the first time, be able
  to cover `qmeq/approach/`.
- **The rendered examples.** 11 `.nblink` files feed notebooks into the docs via
  `nbsphinx` with `nbsphinx_execute = 'never'`. MkDocs equivalents exist
  (`mkdocs-jupyter`), but the notebooks are also run as tests, so the two uses
  need to stay in step.

Neither is a docstring problem, and neither should block the exit; they just
need to be in the plan.

## Module-variable documentation

Use a docstring **after** the assignment, not a `#:` comment:

```python
NO_INDEX = -1
"""Returned by get_ind_dm0 in place of an index when ..."""
```

`#:` is Sphinx-autodoc-only syntax, appears nowhere else in this repository, and
is not read by `griffe`. The string form is read by Sphinx, by `griffe`, and by
editors on hover — so it survives the switch and is more useful before it.
