# Getting started: the `Builder` API

## The four builder classes

`qmeq.Builder` is the entry point for constructing and solving a system. It
is a thin wrapper: which underlying class it becomes is decided by which
constructor you call.

| class | input | use when |
|---|---|---|
| `Builder` / `BuilderBase` | single-particle `hsingle`/`coulomb`/`tleads` dictionaries | the common case: you specify the dot in single-particle terms |
| `BuilderManyBody` | many-body `Ea`/`Na`/`Tba` arrays directly | you already have the many-body eigenspectrum and tunneling matrix (e.g. from an external diagonalization) |
| `BuilderElPh` | adds a phonon bath (`velph`, `tlst_ph`, `dband_ph`, `bath_func`) | electron-phonon coupling matters |
| `BuilderManyBodyElPh` | many-body input **and** a phonon bath | both of the above |

`Builder.base(...)`, `Builder.many_body(...)`, `Builder.elph(...)`, and
`Builder.many_body_elph(...)` are factory methods that construct these four
directly (`qmeq/builder/builder.py`, `qmeq/builder/builder_base.py`,
`qmeq/builder/builder_elph.py`).

!!! warning "`BuilderManyBodyElPh` may not solve"
    `BuilderManyBodyElPh` constructs successfully, but calling
    `solve(qdq=False, rotateq=False)` raises `IndexError` in `get_ind_dm0`,
    because `si_elph` is never set up for many-body input, on both backends.

## A minimal example

```python
import qmeq

system = qmeq.Builder(
    nsingle=1,
    hsingle={(0, 0): 0.0},
    coulomb={},
    nleads=2,
    tleads={(0, 0): 0.1, (1, 0): 0.1},
    mulst={0: 1.0, 1: -1.0},
    tlst={0: 0.2, 1: 0.2},
    dband={0: 100.0, 1: 100.0},
    kerntype="Pauli",
)
system.solve()

system.current         # array of length nleads
system.energy_current  # array of length nleads
system.heat_current    # array of length nleads
system.phi0            # stationary reduced density matrix, packed
```

Under the compiled `cython` backend, this example produces
`current = [0.0309954, -0.0309954]`, `energy_current = [0.0, 0.0]`,
`heat_current = [-0.0309954, -0.0309954]`, `phi0 = [0.5, 0.5]`, and a truthy
`system.success`. (`energy_current` is exactly zero here only because a
single spinless level carries one transition energy per lead direction; that
is a property of this toy model, not of the Pauli approach in general.)

## Choosing `kerntype`

`kerntype` selects the [approach](approaches.md): `'Pauli'` (default),
`'Lindblad'`, `'Redfield'`, `'1vN'`, `'2vN'`, `'RTD'` (alias `'pyRTD'`), or
`'RTDnoise'` (Python traversal with compiled scalar integrals when the Cython
backend is active), or `'pyRTDnoise'` (all Python). A `'py'`-prefixed name
forces the pure-Python implementation for that approach regardless of
`QMEQ_BACKEND`; the bare name uses whatever the backend loader selected. An
unrecognized string warns (`QmeqWarning`) and falls back to `'Pauli'` rather
than raising. `kerntype` can also be a custom `Approach` subclass. (Validated
by `validate_kerntype`, `qmeq/builder/validation.py`.)

`system.kerntype` can be reassigned on an already-built system — it rebuilds
the solver in place but keeps the model (tutorial 4 uses exactly this to sweep
over approaches on the same system), via `BuilderBase.set_kerntype`
(`qmeq/builder/builder_base.py`).

## `countingleads`: opt-in counting statistics

`countingleads` (default `None`) takes a nonempty iterable of distinct,
in-range integer lead indices and enables the first two zero-frequency
particle-current cumulants for Pauli, Lindblad, Redfield, and 1vN (RTD's
counting statistics go through the separate `RTDnoise` approach instead —
see [The approaches](approaches.md#rtdnoise)). After `solve()`,
`system.current_noise` is `[I, S]` for the aggregate counted leads and
`system.current_noise_matrix` is their lead-resolved noise covariance matrix,
ordered as `countingleads`. Passing a string, a duplicate index, an
out-of-range index, or an empty iterable raises (`TypeError`/`ValueError`)
rather than silently doing something else (`validate_countingleads`,
`qmeq/builder/validation.py`). The counting-statistics formulas themselves
are covered in `legacy_docs/source/theory/counting_statistics.rst`.

## `off_diag_corrections`: RTD-specific

`off_diag_corrections` (default `True`) includes RTD's off-diagonal
corrections in the population kernel. It has no effect for approaches other
than RTD/RTDnoise. RTDnoise resolves the same correction by lead and transferred
charge so that it contributes consistently to current and noise. Set it to
`False` only to reproduce the historical population-only RTDnoise kernel; see
[The approaches](approaches.md#rtdnoise) for the remaining validity limits.

## The main result attributes

After `system.solve()`:

| attribute | meaning |
|---|---|
| `phi0` | stationary reduced density matrix, packed per approach (see [Density-matrix layout](../conventions/density-matrix-layout.md)) |
| `phi1` | first-order density-matrix elements, shape `(nleads, ndm1)` |
| `current`, `energy_current`, `heat_current` | per-lead arrays, length `nleads` |
| `current_noise`, `current_noise_matrix` | present only with `countingleads` set (see above) |
| `kern` | the assembled kernel (Liouvillian) matrix |
| `success` | whether the solve succeeded. Under the pure-Python backend this is a genuine `bool`; under the compiled backend it comes back as Python `int` `0`/`1` (a Cython `bint` crossing into Python) — truthy either way, but not `is True` |
| `niter`, `iters` | 2vN only: iteration count and per-iteration `Iterations2vN` records |

Each attribute name is routed to the object that actually owns it (`qd`,
`leads`, `appr`, or `funcp`) via `attribute_map` in `BuilderBase`
(`qmeq/builder/builder_base.py`). The pure-Python backend sets `success` in
`Approach.solve_kern` as a plain `self.success = True`
(`qmeq/approach/aprclass.py`); the compiled backend stores it as a
Cython-typed `_success` property in `qmeq/approach/c_aprclass.pyx`, which is
what produces the bool-vs-int split noted above.

## Solving

`solve()` on a first-order or RTD system takes `qdq`/`rotateq`/`masterq`/
`currentq` booleans controlling which stages run (diagonalize the dot
Hamiltonian, rotate the tunneling matrix into that eigenbasis, solve the
master equation, compute the current). 2vN's `solve()` instead iterates to a
self-consistent energy-resolved solution and additionally takes
`niter`/`func_iter`; see [The approaches](approaches.md#2vn). (Signatures:
`Approach.solve` and `ApproachBase2vN.solve` in `qmeq/approach/aprclass.py`.)

## What this page does not cover

`BuilderElPh`'s full parameter set (`velph`, `tlst_ph`, `dband_ph`,
`bath_func`, `eps_elph`), the `indexing` options beyond the default
(`'Lin'`, `'sz'`, `'ssq'`) and the `symmetry='spin'` shortcut, and the
`add`/`change`/`remove_states`/`sort_eigenstates` model-editing methods are
all real parts of the API that are not yet written up here — see
`legacy_docs/source/builder/` and the docstrings in `qmeq/builder/` in the
meantime.
