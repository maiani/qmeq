# API reference

Generated from QmeQ's docstrings with
[mkdocstrings](https://mkdocstrings.github.io/), carrying over the Sphinx
`autodoc` tree that used to live at `legacy_docs/source/`.

- [Builder](builder.md) — `qmeq.builder`: the user-facing entry point.
- [Approaches](approach.md) — `qmeq.approach`: the master-equation solvers.
- [Model construction](model.md) — `qmeq.indexing`, `qmeq.qdot`,
  `qmeq.leadstun`, `qmeq.baths`.
- [Special functions](specfunc.md) — `qmeq.specfunc`.
- [Wrappers](wrappers.md) — `qmeq.wrappers`.

Most modules ship in two forms: a pure-Python module and a Cython extension
with a `c_` prefix. `QMEQ_BACKEND` selects between them at import time. The
compiled ones are listed on each page but carry no generated block; see the
note at the foot of any page for why.
