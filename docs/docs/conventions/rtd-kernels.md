# RTD kernel matrices

The RTD approach accumulates into eight separate arrays through two handler
methods that select their destination with an integer. This page records what
the integers mean, what the array names mean, and where the two backends differ.

## The `mi` selector

`KernelHandlerRTD.add_matrix_element` and `set_matrix_element_dd` both end in an
argument named `mi` that picks the destination array:

| `mi` | array | meaning |
|---|---|---|
| 0 | `Wdd` | population kernel |
| 1 | `WE1` | first energy-current kernel |
| 2 | `WE2` | second energy-current kernel |
| 3 | `ReWdn` | $\Re\,W_{dn}^{(1)}$ |
| 4 | `ImWdn` | $\Im\,W_{dn}^{(1)}$ |
| 5 | `ReWnd` | $\Re\,W_{nd}^{(1)}$ |
| 6 | `ImWnd` | $\Im\,W_{nd}^{(1)}$ |
| 7 | `Lnn_inv` | inverse coherence propagator — see below |

Call sites used to look like this:

```python
kh.add_matrix_element(temp1, l, a2, b2, charge, a1, a1, charge, 5)
```

Nine positional arguments ending in a bare `5`, knowable only from the method's
docstring. They now name the destination:

```python
kh.add_matrix_element(temp1, l, a2, b2, charge, a1, a1, charge, RtdMatrix.ReWnd)
```

`RtdMatrix` is an `IntEnum` in `kernel_handler.py` whose member names match the
arrays they select. The compiled path cannot use a Python enum inside `nogil`
code, so `c_kernel_handler.pxd` mirrors it as `RtdMatrixC` with `MAT_`-prefixed
members — a `cdef enum` has no namespace.

!!! tip "The mirror is tested, not trusted"
    `RtdMatrixC` is declared `cpdef` rather than `cdef` specifically so that it
    is visible from Python and the two copies can be compared member-for-member
    in the test suite. A plain `cdef` enum would let them drift silently, which
    is the failure mode this whole exercise keeps running into.

61 call sites were converted: 30 in `RTD.py` and 31 in `c_RTD.pyx`.

## `Lnn` does not hold a Liouvillian

The array historically called `Lnn` holds the **inverse** of the bare coherence
energy splitting, not the coherence-sector Liouvillian. It is populated as

```python
add_element_Lnn(a1, b1, charge, 1.0 / E1)
```

and consumed as the middle factor of the elimination

$$W_{\text{corr}} \mathrel{+}= W_{dn}^{(1)} \, L_{nn}^{-1} \, W_{nd}^{(1)}$$

so it is the propagator used to eliminate coherences, with a clamp on small
splittings. The name is actively misleading and the coherent-RTD plan calls out
that it must not be reused ambiguously in a new engine. The Phase 0 reference
fixtures already record it under the honest key `inverse_Lnn`.

## The two backends route it differently

!!! warning "Same name, different shape and different path"
    | | pure Python | Cython |
    |---|---|---|
    | shape | 2-D, `(kern_size2, kern_size2)` | 1-D, `(kern_size2,)` |
    | written by | `add_element_Lnn`, a dedicated method | `add_matrix_element(..., 7)` |
    | element written | `Lnn[indx, indx]` | `Lnn[indx2]` |

    So the pure-Python side stores a full, almost entirely zero matrix whose
    diagonal is the propagator, while the compiled side stores just that
    diagonal — and `mi = 7` is a valid selector only in the compiled backend.
    The pure-Python `add_matrix_element` would fail on it, because it indexes
    `mats[mi]` with three subscripts.

This is why `set_matrix_list` builds its list with `getattr(self, name, None)`:
the arrays it refers to do not all exist in both backends.

## Coherence axis

The coherence axis of `Wdn`, `Wnd` and `Lnn_inv` is **not** the packed `ndm0r`
layout used everywhere else. See rule L9 in
[Density-matrix layout](density-matrix-layout.md#rtd-uses-a-different-coherence-packing-rule-l9).

## Three coupled asymmetries, not one

The off-diagonal-correction block differs between the backends in three ways at
once, so none of them is separately decidable:

| | pure Python | Cython |
|---|---|---|
| `ReWnd`, `ImWnd` | `(nleads, n_nn, n_dd)` — lead-resolved | `(n_nn, n_dd)` — lead-summed at insertion |
| `Lnn_inv` | `(n_nn, n_nn)` dense diagonal | `(n_nn,)` bare diagonal |
| mutation | none; `np.sum(ImWnd, 0)` at use | `diag_matrix_multiply` scales `ReWnd`/`ImWnd` **in place** |

The last one has the widest reach: after `add_off_diag_corrections`, the
compiled `ReWnd` no longer holds the raw kernel — it holds
$L_{nn}^{-1} W_{nd}$.

### Measured facts

- `Lnn_inv` is **exactly diagonal** on every Phase 0 RTD scenario: nothing
  writes off the diagonal, so the dense form carries no information the vector
  does not.
- `ReWdn @ diag(v) @ Wnd` and `(ReWdn * v) @ Wnd` are **bitwise equal**, checked
  on the fixture scenarios and on random matrices up to $812\times812$. Summing
  exact zeros is exact in IEEE arithmetic regardless of BLAS ordering.
- There is **no** robustness difference for non-finite values: with an infinity
  in `ReWdn` both forms produce `nan`, because `inf * 0.0` is `nan` either way.
- The memory at stake is small. $n_{nn} = 2(\mathrm{ndm0} - \mathrm{npauli})$,
  which is 54 for a four-orbital dot, so the dense array is about 23 KB. The
  case for changing it is legibility and backend symmetry, **not** performance.

### How to decide it

Not on tidiness — on what the coherent-RTD plan's Phase 2 needs. That phase
compares generated first-order blocks against these arrays *element by element
and lead by lead*. Only the pure-Python layout has a lead axis to compare
against, and the compiled arrays are destroyed in place. So the answer is mixed:

1. **`Lnn_inv` → adopt the Cython convention** (1-D) in pure Python. Bitwise
   safe, per above.
2. **`ReWnd`/`ImWnd` → keep the pure-Python convention** (lead-resolved).
   Cython's lead-summing trades away the axis Phase 2 needs.
3. **Remove the in-place mutation**, so the raw kernel survives a solve.

### The acceptance criterion writes itself

`test_qmeq_11_references.py` already carries one compensating branch per
asymmetry — `np.diag(inverse_lnn)` for the compiled layout, and
`reference = inverse_lnn @ np.sum(reference, axis=0)` for the compiled
`ReWnd`/`ImWnd`. Those branches exist *because* of the asymmetries, so each
change is done when its branch can be deleted and the fixtures still pass
untouched. A test that gets shorter is the proof.

Sequence them separately: (1) is low-risk and provable now; (3) changes what
`ReWnd` means after a solve and needs its own verification; (2) is a
$\times n_{\text{leads}}$ memory increase in the compiled path and should be
measured before it is chosen, since Phase 2 runs in shadow mode against the
pure-Python oracle anyway and may not need it.
