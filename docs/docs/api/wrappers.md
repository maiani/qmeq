# Wrappers

Shared dtypes and the Cython LAPACK bindings. Ported from
`legacy_docs/source/wrappers/`.

## `qmeq.wrappers`

::: qmeq.wrappers

## `qmeq.wrappers.mytypes`

::: qmeq.wrappers.mytypes


## Compiled extensions

The `c_*` modules carry no documentation of their own, by design: each opens
with *"For docstrings see documentation of module `<name>`"*, pointing at the
pure-Python module it mirrors. Read that module's entry above. The canonical
source is the `.pyx`/`.pxd` file, not the generated C.

`qmeq.wrappers.c_mytypes` mirrors `qmeq.wrappers.mytypes`.
`qmeq.wrappers.c_lapack` has no pure-Python twin: it wraps
`scipy.linalg.cython_lapack` for the compiled solver, where the pure-Python
path calls `numpy.linalg` directly.
