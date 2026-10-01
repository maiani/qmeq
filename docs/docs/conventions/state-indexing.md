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

So `get_ind_dm0(b, bp, c, maptype=3)` returns a boolean, and the return type
is `int | bool`.

!!! tip "Prefer the named accessors"
    `get_ind_dm0_bool(b, bp, charge)` and `get_ind_dm0_conj(b, bp, charge)` are
    the named forms of `maptype` 2 and 3, matching the names the Cython handler
    uses. `get_ind_dm0_bool` exists on all three classes and
    `get_ind_dm0_conj` on `StateIndexingDM`; selector 0 has no named form.

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

    Any indexing helper reached from electron-phonon code must therefore exist
    on `StateIndexingDMc` too.

## Which selectors exist on which class

Not every class supports every `maptype`, and the differences are structural
rather than accidental:

| class | `0` | `1` | `2` | `3` | used by |
|---|---|---|---|---|---|
| `StateIndexingPauli` | yes | yes | yes | — | populations only, so there is no orientation to store |
| `StateIndexingDM` | yes | yes | yes | yes | Pauli, Lindblad, Redfield, 1vN, RTD |
| `StateIndexingDMc` | yes | yes | yes | — | 2vN; both orientations stored independently, so `conjdm0 is None` |

## Unsupported selectors raise

An unsupported `maptype` raises `ValueError`, naming the selectors the class
supports and why the missing ones do not exist. A silent `None` would be read
by NumPy as `np.newaxis` and reshape an array far from the cause.

## What `si` actually is

The `si` parameter threaded through the approaches and kernel handlers is
**not** a single type:

- `StateIndexingDM` for Pauli, Lindblad, Redfield, 1vN and RTD.
- `StateIndexingDMc` for 2vN — which is why `c_kernel_handler.pyx` sets
  `no_conjugates = not isinstance(si, StateIndexingDMc)`.

A handler receiving `StateIndexingDMc` uses only the sizes; the insertion
methods that need a conjugation map are not exercised on that path.

!!! note "Why `si` is not annotated"
    The obvious annotation `si: StateIndexingDM` would be a false statement for
    the 2vN path, so the parameter is documented in the NumPy-style
    `Parameters` block instead, naming both classes and the attributes actually
    used. See [Type hints](typing.md).

## Sentinel and offset conventions

Two small conventions, named rather than open-coded, shared by the pure-Python
and Cython kernel handlers:

- **`NO_INDEX = -1`** (`qmeq.approach.dm_layout`, mirrored as a `cdef enum` in
  `c_kernel_handler.pxd`) — the "element is not carried" sentinel returned by
  `maptype=1`.
- **`imag_offset = ndm0 - npauli`** — the distance from a reduced index to its
  imaginary partner. The existence test is `i >= npauli`.

!!! note "The reconstruction is implemented twice"
    `KernelHandler.get_phi0_element` and `Builder.get_phi0` both expand a packed
    entry back to a complex number, independently. They agree, but the second
    copy in `qmeq/builder/various.py` open-codes rule L5 with its own offset
    arithmetic and cites the rule. They differ in exclusion behaviour:
    `get_phi0` returns `0.0` for an element outside the carried set *and* for
    a mismatched charge.

Inserting at an excluded endpoint is a no-op, rather than a write through the
`-1` sentinel into the last row or column.
