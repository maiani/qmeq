# RTD kernel matrices

The RTD approach historically accumulated into eight separate arrays through
two handler methods whose integer argument selected the destination. The
`RtdMatrix` enum now names that compatibility interface. New block families use
explicit arrays instead of extending the historical selector. This page records
both conventions and where the two backends differ.

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
splittings. The name is actively misleading and must not be reused ambiguously
in a coherence-retaining engine. Reference fixtures therefore record it under
the honest key `inverse_Lnn`.

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

## Counting-resolved coherence elimination

`qmeq.approach.rtd_blocks` owns the shared traversal and composition boundary
for the pure-Python population RTD and RTDnoise implementations. The diagram
formulas still live on `ApproachPyRTD`, but their `nd`/`dn` coordinates are
generated once and consumed either as the ordinary zero-field correction or as
a counting-resolved block. This is deliberately separate from `RtdMatrix`:
Laplace-derivative blocks are inserted into explicit arrays through the same
canonical packed-coordinate mapping.

For any first-order block element, its signed transfer is

$$q=N_{\mathrm{final}}-N_{\mathrm{initial}},$$

so $q=+1$ denotes an electron entering the dot. Lead and transfer resolved
blocks are written $W_{dn}^{\alpha,q}$ and $W_{nd}^{\beta,q'}$. RTDnoise stores
$q=-1,0,+1$ on its historical three-entry axes; code indexes those axes with
the signed integers themselves, so Python index `-1` addresses the final slot.

Combining each stored real/imaginary channel as `Re + 1j*Im`, the effective
population block is projected separately in every transfer sector:

$$
W_{\mathrm{corr}}^{\alpha\beta;qq'}
=\operatorname{Im}\!\left[
W_{dn}^{\alpha,q}G_{nn}^{(0)}W_{nd}^{\beta,q'}
\right].
$$

Summing over $\beta,q,q'$ gives the correction attributed to lead $\alpha$ in
`Wdd`; summing also over $\alpha$ gives the ordinary population kernel. This
sum is tested directly against `ApproachPyRTD.add_off_diag_corrections`'s
historical real-channel expression.

### The Laplace derivative of the correction

Write the zero-field Schur product of one sector as

$$P^{\alpha\beta;qq'}=W_{dn}^{\alpha,q}\,G_{nn}^{(0)}\,W_{nd}^{\beta,q'}.$$

Two structural facts, both *measured* rather than assumed (and asserted by
`test_correction_projection_keeps_the_only_nonzero_channel`), fix how it is
stored. $P$ is purely imaginary, and $\partial_z P$ is purely real. So the
correction and its derivative are one analytic object,

$$W_{\mathrm{corr}}(z)=-i\,W_{dn}(z)\,G_{nn}(z)\,W_{nd}(z),$$

whose value at $z=0$ is real and whose derivative there is purely imaginary.
That matches the convention every other array in the counting path obeys --
kernels real, `_dz` arrays purely imaginary -- which is not cosmetic:
`nonmarkovian_current_noise_matrix` forms $I_i R_j$ from them and a real noise
depends on it ([Emary2009, Eqs. (40)-(41)]).

Taking $\operatorname{Im}$ of the *derivative*, by apparent symmetry with the
value, keeps the identically zero channel and silently discards the whole term.
That was the state of this code before the derivation above; the resulting
`coherence_correction_dz` was zero to machine precision.

The composition itself is the product rule,

$$
\partial_z W_{\mathrm{corr}}^{\alpha\beta;qq'}
=-i\left[
\left(\partial_z W_{dn}\right)G_{nn}^{(0)}W_{nd}
+W_{dn}\left(\partial_z G_{nn}^{(0)}\right)W_{nd}
+W_{dn}G_{nn}^{(0)}\left(\partial_z W_{nd}\right)
\right]^{\alpha\beta;qq'},
$$

gated directly against an independent finite-$z$, transfer-resolved reference by
`test_finite_laplace_correction_derivative_is_directly_gated`. That reference
rebuilds every sector from $W_{dn}(z)$, $G_{nn}(z)$ and $W_{nd}(z)$ and
central-differences it, using no part of the product rule above. Nothing else
constrains this quantity: it enters the noise only at $O(\Gamma^3)$, so neither
the zero-field reduction to ordinary RTD, nor the diagonal limit, nor the
non-interacting residual order can see it.

**One input to that gate is a convention and not yet a derivation.** The
reference must say how the bare coherence resolvent depends on $z$, and it uses

$$G_{nn}(z)=\frac{1}{\Delta E+z},\qquad
\partial_z G_{nn}^{(0)}=-\left(G_{nn}^{(0)}\right)^2,$$

which is what the implementation assumes. The test's negative control shows the
opposite orientation differs by more than a factor of two, so the choice cannot
drift unnoticed -- but a gate that pins a convention is not a proof of it.
`Lnn_inv` is a *real* diagonal stand-in for a complex resolvent in a packed
real/imaginary coherence layout, and in that layout the orientation does not
follow from inspection. Deriving it from the Leijnse-Wegewijs branch and
orientation rules belongs with the conjugate-partner derivation, and until then
this orientation should be read as recorded, not established.

Runtime arrays name this derivative explicitly with the suffix `_dz`, for
example `Lpm_second_dz`; `dot` is reserved for quantum-dot terminology. The
first-order block and bare-propagator derivatives are analytic.

Two things about the first-order analytic derivative are easy to get wrong and
are both pinned by tests. First, `phi` and the Fermi function take the *scaled*
argument $u=(E-\mu)/T$, so $\partial_z$ carries $1/T_\alpha$ for the lead of
that vertex; without it the derivative has the dimension of an energy and the
noise stops being covariant under an overall rescaling of $E$, $\mu$, $T$ and
$\Gamma$ -- invisible at $T=1$, an $O(\Gamma^2)$ error everywhere else.
Second, each contribution is $a\,(\pi f(u)+\eta\,i\phi(u))$ where the bracket
is a *single* analytic function,

$$\pi f(u) + i\phi(u)
= \frac{\pi}{2} - i\psi\!\left(\tfrac12-\tfrac{iu}{2\pi}\right)
+ i\log\frac{D}{2\pi},$$

so $\partial_z\left[a(\pi f+\eta i\phi)\right]
=\frac{a}{T}\left(\eta\,\pi f'(u)+i\phi'(u)\right)$. The real $\pi f'$
channel cannot be dropped: it cancels only when the two Keldysh partners share
$u$, which is exactly the diagonal limit. Note the $\phi'$ term is governed by
$a$, the coefficient of $\pi f$ -- not by the coefficient of $i\phi$, which
differs from it in a third of the branches.

The acceptance test is the diagonal limit. Setting the two coherence indices
equal turns the block, diagram by diagram, into the diagonal first-order
kernel, so both the value and the Laplace derivative must reproduce
`Lpm_first` and `Lpm_first_dz` -- and the derivative must come out purely
imaginary, because that is where $\pi f'$ cancels. Run at unequal lead
temperatures this also pins the per-lead $1/T_\alpha$. The explicit
second-order direct/exchange integrals use one scale-aware centered derivative
with

$$h=\sqrt[3]{\epsilon_{\rm mach}}\,
\max(1,|E_1|,|E_2|,|E_3|,|T_1|,|T_2|),$$

which balances centered-truncation and floating-point roundoff error. A
full-kernel test compares it with an independently stepped five-point stencil.
The counting-field structure follows
[Emary's non-Markovian formulation](https://arxiv.org/abs/0902.3544), while the
population/coherence block elimination follows the real-time diagrammatic
construction of
[Leijnse and Wegewijs](https://arxiv.org/abs/0807.4027).

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

- `Lnn_inv` is **exactly diagonal** on every pinned RTD scenario: nothing
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

The reusable block boundary requires comparisons *element by element and lead
by lead*. Only the pure-Python layout has a lead axis to compare against, and
the compiled arrays are destroyed in place. So the answer is mixed:

1. **`Lnn_inv` → adopt the Cython convention** (1-D) in pure Python. Bitwise
   safe, per above.
2. **`ReWnd`/`ImWnd` → keep the pure-Python convention** (lead-resolved).
   Cython's lead-summing trades away the axis a counting or coherent consumer
   needs.
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
measured before it is chosen. The current counted correction is pure Python and
does not require that memory increase in the compiled population solver.
