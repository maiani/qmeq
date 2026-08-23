# State indexing

`qmeq.indexing` provides three indexing classes and one badly overloaded
lookup method. This page documents the overload, because it is the single
biggest obstacle to reading the kernel-assembly code.

## `maptype`: one method, four unrelated questions

`get_ind_dm0(b, bp, charge, maptype=1)` does not return "the index". It returns
one of four different properties depending on `maptype`, and **only two of them
are indices**:

| `maptype` | returns | meaning |
|---|---|---|
| `0` | `int` | Flat, **unreduced** index: `lenlst[c]*dictdm[b] + dictdm[bp] + shiftlst0[c]`. Always valid. This is what addresses `mapdm0`, `booldm0` and `conjdm0` directly. |
| `1` | `int` | **Reduced** index in `[0, ndm0)`, or `-1` if the element is not carried. The default, and what "the index" normally means. |
| `2` | `bool` | Is this the **representative** orientation, used to enumerate the element exactly once? |
| `3` | `bool` | Is this the **stored** orientation, which fixes the sign of the imaginary part? |

So `get_ind_dm0(b, bp, c, maptype=3)` returns a boolean from a method whose
docstring long promised `int: Index of the zeroth order density matrix
element`. That overload accounts for much of why insertion code reads as noise.

!!! tip "Prefer the named accessors"
    `StateIndexingDM` now provides `get_ind_dm0_bool(b, bp, charge)` and
    `get_ind_dm0_conj(b, bp, charge)` for `maptype` 2 and 3, matching the names
    the Cython handler already used. The integer interface still works and is
    unchanged.

### 2 and 3 are different predicates

They are easy to conflate because they coincide under `indexing='charge'`.
They do not coincide under `'ssq'`, where the representative marking and the
orientation marking come apart. Code that uses one where it means the other
will pass every `charge` test and fail only on symmetry-reduced runs.

## Class hierarchy

The three indexing classes are **siblings**, not a chain:

```
StateIndexing                     # no dm0 lookups at all
├── StateIndexingPauli            # populations only
├── StateIndexingDM               # packed real, Hermiticity-reduced
└── StateIndexingDMc              # complex, both orientations kept
```

`issubclass(StateIndexingDMc, StateIndexingDM)` is `False`. This is worth
stating because the names suggest otherwise — "DMc" reads like a specialisation
of "DM", and it is not one.

!!! warning "There is no shared base for the `dm0` contract"
    `get_ind_dm0` is **not** defined on `StateIndexing`. Each of the three
    subclasses defines its own, with different selectors and different
    semantics. So there is no single type that means "provides the `dm0`
    lookups" — which is why signatures that accept more than one of them must
    spell out a union rather than name a common ancestor.

`StateIndexingDMc` does carry `ndm0`, `ndm0r` and `npauli`, so it satisfies
`KernelHandler.__init__` by duck typing even though it cannot support the
conjugation-dependent methods (`conjdm0` is `None`).

!!! danger "An approach can hold two indexing objects of different classes"
    The electron-phonon approaches carry **both**:

    | attribute | class | built by |
    |---|---|---|
    | `si` | `StateIndexingDM` | the approach's `indexing_class_name` |
    | `si_elph` | `StateIndexingDMc` | `BuilderElPh.create_si_elph` |

    So inside `qmeq/approach/elph/`, whether `get_ind_dm0` is being called on a
    `StateIndexingDM` or a `StateIndexingDMc` depends on which attribute the
    local name `si` was bound to — and in `elph/pauli.py` and `elph/neumann1.py`
    the local `si` is `self.si_elph`, not `self.si`.

    This was found the hard way: adding `get_ind_dm0_bool` to `StateIndexingDM`
    alone and converting those call sites broke all six electron-phonon
    reference tests with an `AttributeError`. Any helper reached from elph code
    must exist on `StateIndexingDMc` too.

## Which selectors exist on which class

Not every class supports every `maptype`, and the differences are structural
rather than accidental:

| class | `0` | `1` | `2` | `3` | used by |
|---|---|---|---|---|---|
| `StateIndexingPauli` | yes | yes | yes | — | populations only, so there is no orientation to store |
| `StateIndexingDM` | yes | yes | yes | yes | Pauli, Lindblad, Redfield, 1vN, RTD |
| `StateIndexingDMc` | yes | yes | yes | — | 2vN; both orientations stored independently, so `conjdm0 is None` |

## Unsupported values used to fail silently

!!! danger "Historical failure mode — fixed"
    Every unsupported `maptype` fell off the `if`/`elif` chain and returned
    `None`. Confirmed for `maptype=3` on both `StateIndexingDMc` and
    `StateIndexingPauli`, and for `maptype=4` on `StateIndexingDM`.

    `None` is not an error in NumPy — it is `np.newaxis`. So a wrong `maptype`
    reshaped an array rather than raising, and the failure surfaced far from
    its cause.

All three classes now raise `ValueError` naming the selectors they support and
why the missing ones do not exist.

!!! example "This was not a purely theoretical hazard"
    Turning the silent `None` into a `ValueError` immediately failed
    `test_various.py::test_get_phi0_and_get_phi1`. `Builder.get_phi0` was
    calling `get_ind_dm0(b, bp, bcharge, maptype=3)` **unconditionally**, then
    branching on `type(self.si).__name__ == 'StateIndexingDMc'` and discarding
    the result on that branch.

    So for every 2vN system the code asked `StateIndexingDMc` for a conjugation
    map it does not have, got `None` back, and silently threw it away. Harmless
    in outcome, but it means the silent return was load-bearing for a shipped
    code path — not merely a trap waiting for a future caller.

    Fixed by moving the lookup into the branch that uses it. The lesson
    generalises: a sentinel that never raises hides not just future mistakes but
    existing ones, and you only find out how many when you make it loud.

## What `si` actually is

The `si` parameter threaded through the approaches and kernel handlers is
**not** a single type:

- `StateIndexingDM` for Pauli, Lindblad, Redfield, 1vN and RTD.
- `StateIndexingDMc` for 2vN — which is why `c_kernel_handler.pyx` branches on
  `isinstance(si, StateIndexingDMc)` to set `no_conjugates`.

A handler receiving `StateIndexingDMc` uses only the sizes; the insertion
methods that need a conjugation map are not exercised on that path.

!!! note "Why there are no type annotations here"
    The obvious annotation `si: StateIndexingDM` would be a false statement for
    the 2vN path. The repository also carries no type hints anywhere else, so
    the parameter is documented in the NumPy-style `Parameters` block instead,
    naming both classes and the attributes actually used.

## Sentinel and offset conventions

Two small conventions that used to be open-coded at 19 sites across the
pure-Python and Cython kernel handlers, and are now named:

- **`EXCLUDED = -1`** (`qmeq.approach.dm_layout`, mirrored as a `cdef enum` in
  `c_kernel_handler.pxd`) — the "element is not carried" sentinel returned by
  `maptype=1`.
- **`imag_offset = ndm0 - npauli`** — the distance from a reduced index to its
  imaginary partner. The existence test is now `i >= npauli`, which is
  equivalent to the older `i + ndm0 - npauli >= ndm0` but says what it means.

!!! note "The reconstruction is implemented twice"
    `KernelHandler.get_phi0_element` and `Builder.get_phi0` both expand a packed
    entry back to a complex number, independently. They agree, but the second
    copy in `qmeq/builder/various.py` open-codes rule L5 with its own offset
    arithmetic. It now cites the rule; it has not been merged, because the two
    have different exclusion behaviour (`get_phi0` returns `0.0` for an element
    outside the carried set *and* for a mismatched charge).

Inserting at an excluded endpoint is now a no-op rather than a write through
the `-1` sentinel into the last row or column. Every shipped caller already
guards with `is_included`; a probe run of the full suite with a hard assertion
never fired, so the guard changes no tested behaviour and only closes a
silent-corruption path for future callers.

## Open questions

- Whether the `maptype` overload should be split into four separate methods
  outright, leaving `get_ind_dm0` as the reduced-index lookup only. The named
  accessors cover selectors 2 and 3; selector 0 has no named form yet.
