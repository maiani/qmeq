# RTD kernel matrices

The RTD approach accumulates into eight arrays through two handler methods
whose last argument selects the destination, named by the `RtdMatrix` enum.
Block families outside those eight, such as the Laplace-derivative blocks, use
explicit arrays instead. This page records both conventions, the shared diagram
enumeration, and where the two backends differ.

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

Call sites name the destination:

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
    in the test suite. A plain `cdef` enum would let them drift silently.

## `Lnn_inv` does not hold a Liouvillian

`Lnn_inv` holds the **inverse** of the bare coherence energy splitting, not
the coherence-sector Liouvillian. It is populated as

```python
kh.add_element_Lnn_inv(a1, b1, charge, 1.0/E1)
```

and consumed as the middle factor of the elimination

$$W_{\text{corr}} \mathrel{+}= W_{dn}^{(1)} \, L_{nn}^{-1} \, W_{nd}^{(1)}$$

so it is the propagator used to eliminate coherences, with a clamp on small
splittings. Reference fixtures record it under the key `inverse_Lnn`.

## The two backends route it differently

!!! warning "Same name, different shape and different path"
    | | pure Python | Cython |
    |---|---|---|
    | shape | 2-D, `(kern_size2, kern_size2)` | 1-D, `(kern_size2,)` |
    | written by | `add_element_Lnn_inv`, a dedicated method | `add_matrix_element(..., MAT_LNN_INV)` |
    | element written | `Lnn_inv[indx, indx]` | `Lnn_inv[indx2]` |

    So the pure-Python side stores a full, almost entirely zero matrix whose
    diagonal is the propagator, while the compiled side stores just that
    diagonal — and `mi = 7` is a valid selector only in the compiled backend.
    The pure-Python `add_matrix_element` would fail on it, because it indexes
    `mats[mi]` with three subscripts.

This is why `set_matrix_list` builds its list with `getattr(self, name, None)`:
the arrays it refers to do not all exist in both backends.

## `WE1` and `WE2` are contractions, not orders

`generate_row_1st_energy_kernel` and `generate_row_2nd_energy_kernel` name two
contractions of one `O(Gamma^2)` energy-current correction. Neither is a
first-order block, and the leading energy current is not in either: it is the
`LE` contraction of `Wdd`, added separately in `generate_current`.

That is why `rtd_order = 1` leaves both arrays zero and still reproduces the
sequential energy and heat currents exactly. Reading `WE1` as "the first-order
energy kernel" would predict the opposite, and would make the order-1 gate look
like a missing term.

## The second-order `.real` is a partner sum, not a truncation

`generate_col_diag_kern_2nd_order` evaluates each `SecondOrderDiagram` record
and passes the real part of its value to `add_element_2nd_order`, and
`generate_fct` takes the real part `xcb` of a two-amplitude product. Neither
discards flux-dependent physics from the charge current.

The enumeration fixes `eta1 = 1`, `p1 = 1` and `p4 = 1` (the paper's
numbering) and recovers the omitted half from a symmetry. `add_element_2nd_order` opens with `fct = 2*fct`, so a
stored `Re(t)` reaches the kernel as `t + conj(t)`. The omitted `eta0 = -1`
partner is the complex conjugate of the *complete* four-vertex contribution,
integral included -- not the vertex-by-vertex conjugate, which is a different
quantity when the integral is complex. Taking the real part is how that pair
is summed.

Two gates hold this down, both at a generic plaquette flux where the
four-amplitude product carries a physical phase:

- `test_complex_flux_second_order_kernel_matches_stationary_rtd` builds
  RTDnoise's explicitly resolved `eta0 = +-1` partners and reproduces the
  ordinary-RTD kernel. It also requires each *transfer-resolved* block to be
  real to `1e-14` before the sectors are summed, so the cancellation is
  pairwise rather than an accident of the total.
- `test_complex_flux_rtdnoise_observables_have_cubic_residuals` compares
  ordinary RTD's current against the exact non-interacting solver in
  `qmeq.tests.noninteracting_negf_solver`, over four coupling scales, and
  requires a cubic error slope. A dropped `O(Gamma^2)` contribution would make
  that slope quadratic.

`xcb` needs no diagram argument at all: `Tba[l]` is Hermitian by construction,
so `Tba[l, b, c]*Tba[l, c, b]` is `|Tba[l, b, c]|^2` and the projection removes
roundoff. Measured on a flux-carrying double dot whose amplitudes have
`max|Im Tba| = 2e-2`, the product's imaginary part is `4e-21`.

This covers the charge current only. QmeQ does not compute the RTD energy and
heat currents for complex amplitudes; both are filled with `nan`.

## One diagram enumeration for RTD and RTDnoise

`qmeq.approach.rtd_diagrams` enumerates the population diagrams, and the two
approaches only evaluate them. `first_order_diagrams` yields one
`FirstOrderDiagram` per two-vertex diagram of a kernel row, as a gain or a
loss through a state one charge away. `second_order_diagrams` yields one
`SecondOrderDiagram` per independent four-vertex diagram of a column. Each
record carries its topology (direct or exchange), the two leads, `eta1`,
`p1`, `p2`, the visited states, the four-amplitude product, the propagator
energies, the lead parameters and its population endpoints. `pyRTD` evaluates
a record with `integralD`/`integralX` and adds twice its real part.
`RTDnoise` evaluates it with the counting integrals and their Laplace
derivatives, keeping its transfer labels.

A record stands for more diagrams than the one it enumerates:

- **outer Keldysh flips** `p0, p3 = +-1` give the four insertions
  `(final, initial)`, `(flipped, initial)`, `(final, initial_flipped)` and
  `(flipped, initial_flipped)`, with signs `+, -, +, -`;
- **the inverted partner** `eta0 = -1` has the complex-conjugate value, the
  negative complex-conjugate Laplace derivative and the same transfer labels
  [LeijnseWegewijs2008, Eqs. (B1)-(B3), (D1)-(D3)], [Emary2009, Eq. (31)].
  `pyRTD` adds it through the real part, and RTDnoise in
  `_complete_second_order_conjugate_partners`.

`counting_labels` gives the transferred charges `(q0, q1)` of the four
insertions at the `r0` and `r1` contractions, positive into the dot. Their
sum equals the charge of the row population minus that of the column, at
every insertion of every diagram. `test_counting_labels_conserve_charge`
checks this identity, which does not follow from the label formula alone.

Records are yielded in a fixed order, and both approaches accumulate in that
order, so the assembled arrays do not depend on which approach consumed the
stream. The compiled `c_RTD.pyx` enumerates the same diagrams with
hand-written loops. A change to the enumeration must be applied there too.
`test_compiled_rtd_matches_the_record_based_python_rtd` and the RTD reference
bundle in `qmeq/tests/data/qmeq_11/` hold the two together.

## Coherence axis

The coherence axis of `Wdn`, `Wnd` and `Lnn_inv` is **not** the packed `ndm0r`
layout used everywhere else. See rule L9 in
[Density-matrix layout](density-matrix-layout.md#rtd-uses-a-different-coherence-packing-rule-l9).

## Counting-resolved coherence elimination

`qmeq.approach.rtd_blocks` owns the traversal and composition of the
first-order population-coherence blocks for the pure-Python RTD and RTDnoise.
The block formulas are `ApproachPyRTD.generate_col_nondiag_kern_1st_order_dn`
and `generate_col_nondiag_kern_1st_order_nd`; their coordinates are generated
once and consumed either as the ordinary zero-field correction or as a
counting-resolved block. This is deliberately separate from `RtdMatrix`:
Laplace-derivative blocks are inserted into explicit arrays through the same
canonical packed-coordinate mapping.

For any first-order block element, its signed transfer is

$$q=N_{\mathrm{final}}-N_{\mathrm{initial}},$$

so $q=+1$ denotes an electron entering the dot. Lead and transfer resolved
blocks are written $W_{dn}^{\alpha,q}$ and $W_{nd}^{\beta,q'}$. RTDnoise stores
$q=-1,0,+1$ on three-entry transfer axes; code indexes those axes with
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
`Wdd`; summing also over $\alpha$ gives the ordinary population kernel, which
`test_counting_resolved_coherence_correction_reduces_to_standard_rtd` checks
against ordinary RTD.

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
value, keeps the identically zero channel and silently discards the whole term:
`coherence_correction_dz` would then be zero to machine precision.

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

The bare-resolvent orientation follows from the free molecular line, not from
the numerical gate. Leijnse and Wegewijs define

$$\Pi^0(z_{\rm LW})=i\left(z_{\rm LW}-L\right)^{-1}$$

in Eq. (49), while their stationary kinetic equation is
$0=(-iL+W)P$ [LeijnseWegewijs2008, Eqs. (19), (49)]. Acting on the ordered
coherence $|a\rangle\langle b|$, the molecular Liouvillian has eigenvalue
$\Delta E_{ab}=E_a-E_b$. Write $s$ for the physical Laplace variable and $x$ for the energy-like
variable used by QmeQ's stored `_dz` derivatives. The retarded vertex blocks
have digamma argument $1/2+s/(2\pi T)-i\eta u/(2\pi)$.
Their coded continuation $u\mapsto u+\eta x/T$ therefore requires $x=is$,
so $\partial_x=-i\partial_s$. The free coherence line must use the same
continuation:

$$\frac{1}{s+i\Delta E}=\frac{-i}{\Delta E-x}.$$

Extracting the factor $-i$ into $W_{\rm corr}=-iW_{dn}G_{nn}W_{nd}$ leaves

$$G_{nn}(x)=\frac{1}{\Delta E-x},\qquad
\partial_xG_{nn}^{(0)}=+\left(G_{nn}^{(0)}\right)^2.$$

The two ordered coherence partners have splittings $\Delta E$ and $-\Delta E$,
but the same $-x$ shift. Using a $+x$ shift for this line while keeping the
vertex continuation above reverses only the resolvent contribution to the
noise. The finite physical-Laplace test evaluates the retarded digamma blocks
and $1/(s+i\Delta E)$ directly to check this relative orientation. A separate
energy-variable test checks the stored product rule and its opposite-sign
negative control.

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
The counting-field structure follows Emary's non-Markovian formulation
[Emary2009], while the population/coherence block elimination follows the
real-time diagrammatic construction of [LeijnseWegewijs2008].

## The backends store the correction blocks differently

The off-diagonal-correction blocks differ between the backends in three ways:

| | pure Python | Cython |
|---|---|---|
| `ReWnd`, `ImWnd` | `(nleads, n_nn, n_dd)`, lead-resolved | `(n_nn, n_dd)`, lead-summed at insertion |
| `Lnn_inv` | `(n_nn, n_nn)`, nonzero only on the diagonal | `(n_nn,)`, the diagonal |
| after `add_off_diag_corrections` | unchanged; leads are summed at use | `diag_matrix_multiply` scales `ReWnd` and `ImWnd` **in place** |

After a compiled solve `ReWnd` and `ImWnd` therefore hold
$L_{nn}^{-1} W_{nd}$, not the raw kernel. Code comparing the two backends must
compensate for all three: `test_qmeq_11_references.py` does so with one branch
per difference, `np.diag(inverse_lnn)` for the compiled `Lnn_inv` and
`inverse_lnn @ np.sum(reference, axis=0)` for the compiled `ReWnd`/`ImWnd`.
Only the pure-Python layout keeps the lead axis that the counting-resolved
correction needs.
