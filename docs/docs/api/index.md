# API reference

Generated from QmeQ's docstrings with
[mkdocstrings](https://mkdocstrings.github.io/).

- [Builder](builder.md) — `qmeq.builder`: the user-facing entry point.
- [Approaches](approach.md) — `qmeq.approach`: the master-equation solvers.
- [Model construction](model.md) — `qmeq.indexing`, `qmeq.qdot`,
  `qmeq.leadstun`, `qmeq.baths`, and the dtypes of `qmeq.wrappers`.
- [Special functions](specfunc.md) — `qmeq.specfunc`.

Start from the Builder page: it is the public interface. The Approaches and
Special functions pages document internals, for work on QmeQ itself.

Most approach, special-function and wrapper modules ship in two forms: a
pure-Python module and a Cython extension with a `c_` prefix. `QMEQ_BACKEND`
selects between them at import time, and `qmeq.get_backend_status()` reports
the choice. The compiled ones are listed at the foot of the Approaches,
Special functions and Model construction pages, without generated entries of
their own.
