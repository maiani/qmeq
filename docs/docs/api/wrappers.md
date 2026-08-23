# Wrappers

Shared dtypes and the Cython LAPACK bindings. Ported from
`legacy_docs/source/wrappers/`.

## `qmeq.wrappers`

::: qmeq.wrappers

## `qmeq.wrappers.mytypes`

::: qmeq.wrappers.mytypes


## Compiled extensions

Sphinx's `autodoc` documented the `c_*` extension modules by importing them.
`mkdocstrings` reads source statically and there is no `.py` to read, so these
modules get no generated block here. Their docstrings live in the pure-Python
module each one mirrors — which is what the `.pyx` sources themselves say:
*"For docstrings see documentation of module `<name>`"*. The canonical source is
the `.pyx`/`.pxd` file, not the generated C.

`qmeq.wrappers.c_mytypes` mirrors `qmeq.wrappers.mytypes`.
`qmeq.wrappers.c_lapack` has no pure-Python twin: it wraps
`scipy.linalg.cython_lapack` for the compiled solver, where the pure-Python
path calls `numpy.linalg` directly.
