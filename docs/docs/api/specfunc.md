# Special functions

Ported from `legacy_docs/source/specfunc/`.

## `qmeq.specfunc`

::: qmeq.specfunc

## `qmeq.specfunc.specfunc`

::: qmeq.specfunc.specfunc

## `qmeq.specfunc.specfunc_elph`

::: qmeq.specfunc.specfunc_elph


## Compiled extensions

Sphinx's `autodoc` documented the `c_*` extension modules by importing them.
`mkdocstrings` reads source statically and there is no `.py` to read, so these
modules get no generated block here. Their docstrings live in the pure-Python
module each one mirrors — which is what the `.pyx` sources themselves say:
*"For docstrings see documentation of module `<name>`"*. The canonical source is
the `.pyx`/`.pxd` file, not the generated C.

`qmeq.specfunc.c_specfunc`, `qmeq.specfunc.c_specfunc_elph`.
