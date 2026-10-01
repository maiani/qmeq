# Approaches

The master-equation solvers and their shared machinery. Most of what this page
documents is internal: a calculation is set up and solved through the
[Builder](builder.md), and the approaches are chosen by `kerntype`. The page is
the reference for working on the approaches themselves.

## `qmeq.approach`

::: qmeq.approach

## `qmeq.approach.aprclass`

::: qmeq.approach.aprclass

## `qmeq.approach.kernel_handler`

::: qmeq.approach.kernel_handler

## `qmeq.approach.counting`

::: qmeq.approach.counting

## `qmeq.approach.dm_layout`

States the packed-real density-matrix layout as nine numbered rules; see also
[Density-matrix layout](../conventions/density-matrix-layout.md).

::: qmeq.approach.dm_layout

## `qmeq.approach.diagnostics`

Stationary-solution diagnostics shared by the approaches.

::: qmeq.approach.diagnostics


## `qmeq.approach.rtd_diagrams`

::: qmeq.approach.rtd_diagrams

## `qmeq.approach.rtd_blocks`

::: qmeq.approach.rtd_blocks

## Tunnelling approaches

### `qmeq.approach.base`

::: qmeq.approach.base

### `qmeq.approach.base.pauli`

::: qmeq.approach.base.pauli

### `qmeq.approach.base.lindblad`

::: qmeq.approach.base.lindblad

### `qmeq.approach.base.redfield`

::: qmeq.approach.base.redfield

### `qmeq.approach.base.neumann1`

::: qmeq.approach.base.neumann1

### `qmeq.approach.base.neumann2`

::: qmeq.approach.base.neumann2

### `qmeq.approach.base.RTD`

::: qmeq.approach.base.RTD

### `qmeq.approach.base.RTDnoise`

::: qmeq.approach.base.RTDnoise

## Electron-phonon approaches

### `qmeq.approach.elph`

::: qmeq.approach.elph

### `qmeq.approach.elph.pauli`

::: qmeq.approach.elph.pauli

### `qmeq.approach.elph.lindblad`

::: qmeq.approach.elph.lindblad

### `qmeq.approach.elph.redfield`

::: qmeq.approach.elph.redfield

### `qmeq.approach.elph.neumann1`

::: qmeq.approach.elph.neumann1


## Compiled extensions

The compiled `c_*` modules have no generated entries: each mirrors the
pure-Python module of the same name above, whose docstrings document both. The
canonical source is the `.pyx`/`.pxd` file, not the generated C.

`qmeq.approach.c_aprclass`, `qmeq.approach.c_kernel_handler`,
`qmeq.approach.base.c_pauli`, `qmeq.approach.base.c_lindblad`,
`qmeq.approach.base.c_redfield`, `qmeq.approach.base.c_neumann1`,
`qmeq.approach.base.c_neumann2`, `qmeq.approach.base.c_RTD`,
`qmeq.approach.elph.c_pauli`, `qmeq.approach.elph.c_lindblad`,
`qmeq.approach.elph.c_redfield`, `qmeq.approach.elph.c_neumann1`.

RTDnoise has no compiled module. `ApproachRTDnoise` evaluates the shared
diagram records of `qmeq.approach.rtd_diagrams` in Python and takes its direct
and exchange scalar integrals from `qmeq.specfunc.c_specfunc` when the Cython
backend is active; `ApproachPyRTDnoise` is all Python. The compiled `c_RTD`
enumerates the RTD diagrams with its own loops, while `pyRTD` evaluates the
shared records.
