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
are derived on the [counting-statistics theory page](../theory/counting-statistics.md).

## `off_diag_corrections`: RTD-specific

`off_diag_corrections` (default `True`) includes RTD's off-diagonal
corrections in the population kernel. It has no effect for approaches other
than RTD/RTDnoise. RTDnoise resolves the same correction by lead and transferred
charge so that it contributes consistently to current and noise. Set it to
`False` only to reproduce the historical population-only RTDnoise kernel; see
[The approaches](approaches.md#rtdnoise) for the remaining validity limits.

## Electron-phonon systems

Use `BuilderElPh` (or `Builder.elph`) when bosonic baths drive transitions
inside the dot. In addition to the electronic model, provide:

| argument | meaning |
|---|---|
| `nbaths` | number of phonon baths (stored on `system.si.nbaths`) |
| `velph[(bath, i, j)]` | single-particle coupling from orbital `j` to `i`; an array may instead have shape `(nbaths, nsingle, nsingle)` |
| `tlst_ph` | bath temperatures |
| `dband_ph` | bath bandwidths |
| `bath_func` | optional density-of-states function for each bath |
| `itype_ph` | phonon integral selector: `0` includes principal parts, `2` omits them |
| `eps_elph` | small integration stabilizer near the Bose-function singularity |

```python
system = qmeq.Builder.elph(
    nsingle=2,
    hsingle={(0, 0): -0.5, (1, 1): 0.5},
    coulomb={},
    nleads=2,
    tleads={(0, 0): 0.1, (1, 1): 0.1},
    mulst={0: 1.0, 1: -1.0},
    tlst={0: 0.2, 1: 0.2},
    dband={0: 100.0, 1: 100.0},
    nbaths=1,
    velph={(0, 0, 1): 0.05, (0, 1, 0): 0.05},
    tlst_ph={0: 0.2},
    dband_ph={0: 100.0},
    kerntype="Pauli",
)
```

The builder constructs `Vbbp`, the many-body electron-phonon coupling, from
`velph` and rotates it with the dot eigenstates during `solve()`. The
electron-phonon variants support Pauli, Lindblad, Redfield, and 1vN. They do
not implement particle counting. If the many-body energies and couplings are
already known, use `BuilderManyBodyElPh`/`Builder.many_body_elph` and supply
`Ea`, `Na`, `Tba`, and `Vbbp` directly. Complete signatures are in the
[Builder API](../api/builder.md).

## Indexing and spin symmetry

`indexing` controls how Fock and many-body states are grouped; it is fixed
when the builder is constructed and cannot be changed in place.

| value | grouping and intended use |
|---|---|
| `'Lin'` | binary (linear) Fock-state order |
| `'charge'` | charge sectors; the default without spin symmetry |
| `'sz'` | charge and spin projection $S_z$; requires even `nsingle` |
| `'ssq'` | charge, $S_z$, and total-spin information; requires even `nsingle` |

`symmetry='spin'` duplicates the supplied spin-up orbitals and lead channels
for spin down. When `indexing` is omitted it selects `'ssq'` for first-order
approaches. RTD/RTDnoise require `indexing='charge'` and do not support the
spin shortcut; 2vN supports only `'Lin'` and `'charge'`. Unsupported
combinations emit `QmeqWarning` and select a supported indexing, so inspect
`system.indexing` when adapting an existing model. The internal packed-layout
contract is documented separately in [State indexing](../conventions/state-indexing.md).

## Editing an existing model

The builder keeps the single-particle input and derived many-body objects in
sync when its editing methods are used:

- `add(...)` increments Hamiltonian, interaction, tunnelling, or reservoir
  entries. `BuilderElPh.add(...)` also accepts `velph`, `tlst_ph`, and
  `dlst_ph`.
- `change(...)` assigns new values to named entries and can also replace
  `countingleads`. Use `dlst`/`dlst_ph` for bandwidth changes in these methods.
- `remove_states(dE)` excludes many-body eigenstates more than `dE` above the
  ground state; `use_all_states()` restores them.
- `remove_coherences(dE)` excludes coherences whose energy splitting exceeds
  `dE`.
- `sort_eigenstates(srt)` changes the ordering used for inspection and output.
- `remove_fock_states(indices)` removes selected linear-index Fock states and
  rebuilds the dependent couplings.

Call `solve()` after changing model inputs. Prefer these methods over mutating
`qd`, `leads`, or `baths` arrays directly, because the methods rebuild the
dependent many-body matrices. For a different `indexing`, construct a new
builder.

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
