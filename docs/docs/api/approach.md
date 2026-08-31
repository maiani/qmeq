# Approaches

The master-equation solvers.

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

# Tunnelling approaches

## `qmeq.approach.base`

::: qmeq.approach.base

## `qmeq.approach.base.pauli`

::: qmeq.approach.base.pauli

## `qmeq.approach.base.lindblad`

::: qmeq.approach.base.lindblad

## `qmeq.approach.base.redfield`

::: qmeq.approach.base.redfield

## `qmeq.approach.base.neumann1`

::: qmeq.approach.base.neumann1

## `qmeq.approach.base.neumann2`

::: qmeq.approach.base.neumann2

## `qmeq.approach.base.RTD`

::: qmeq.approach.base.RTD

## `qmeq.approach.base.RTDnoise`

::: qmeq.approach.base.RTDnoise

# Electron-phonon approaches

## `qmeq.approach.elph`

::: qmeq.approach.elph

## `qmeq.approach.elph.pauli`

::: qmeq.approach.elph.pauli

## `qmeq.approach.elph.lindblad`

::: qmeq.approach.elph.lindblad

## `qmeq.approach.elph.redfield`

::: qmeq.approach.elph.redfield

## `qmeq.approach.elph.neumann1`

::: qmeq.approach.elph.neumann1


## Compiled extensions

The `c_*` modules carry no documentation of their own, by design: each opens
with *"For docstrings see documentation of module `<name>`"*, pointing at the
pure-Python module it mirrors. Read that module's entry above. The canonical
source is the `.pyx`/`.pxd` file, not the generated C.

`qmeq.approach.c_aprclass`, `qmeq.approach.c_kernel_handler`,
`qmeq.approach.base.c_pauli`, `qmeq.approach.base.c_lindblad`,
`qmeq.approach.base.c_redfield`, `qmeq.approach.base.c_neumann1`,
`qmeq.approach.base.c_neumann2`, `qmeq.approach.base.c_RTD`,
`qmeq.approach.elph.c_pauli`, `qmeq.approach.elph.c_lindblad`,
`qmeq.approach.elph.c_redfield`, `qmeq.approach.elph.c_neumann1`.
