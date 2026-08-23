# Density-matrix layout

How a Hermitian density matrix becomes the real vector QmeQ solves for.

The authoritative statement is the module docstring of
`qmeq.approach.dm_layout`, which numbers the rules L1–L9. This page explains
the two things the rules do not: *why* the layout looks like this, and which
conventions were recovered by measurement rather than read off the source.

## Why real and not complex

The natural first question is why the code splits real and imaginary parts by
hand instead of using complex arrays. Three reasons, in decreasing order of
force.

### It is forced by Hermiticity

Hermitian matrices form a **real** vector space, not a complex one:
$i\\cdot H$ is anti-Hermitian, so multiplying by $i$ leaves the space. QmeQ uses
$\\rho^\\dagger = \\rho$ to store each pair $|b\\rangle\\langle b'|$ and
$|b'\\rangle\\langle b|$ once (rule L3). Once you do that, the kernel is a
real-linear map that is **not complex-linear**, and a map that is not
complex-linear has no complex matrix to be.

You can see exactly where linearity breaks. In `KernelHandler.set_matrix_element`:

```python
aap_sgn = +1 if self.si.get_ind_dm0_conj(a, ap, acharge) else -1
```

When the source element is stored in the conjugate orientation, the
contribution involves $\\rho^*$ — and conjugation is not complex-linear. That
`±1` *is* the conjugation.

The escape hatch is to stop imposing Hermiticity: keep both orientations as
independent complex unknowns and complex linearity survives. That is exactly
what the 2vN approaches do, with `StateIndexingDMc` and `dtype = complexnp`.
The choice is per-approach and deliberate.

### It is smaller

Degrees of freedom for a system of `nsingle` orbitals, measured:

| `nsingle` | flat complex | unique complex | packed real (`ndm0r`) |
|---|---|---|---|
| 2 | 12 real dof | 10 | **6** |
| 3 | 40 | 28 | **20** |
| 4 | 140 | 86 | **70** |
| 5 | 504 | 284 | **252** |

A complex vector over unique elements still wastes `npauli` imaginary slots
that are structurally zero — populations are real diagonal entries of a
Hermitian matrix — and unconstrained they would drift numerically.

### Flops are roughly a wash

Real LU at $n=70$ is about $\\tfrac{2}{3}\\cdot 70^3 \\approx 2.3\\times10^5$
flops; complex LU at $n=43$ is about $4\\cdot\\tfrac{2}{3}\\cdot 43^3 \\approx
2.1\\times10^5$. Efficiency is not the deciding argument. Structure is.

!!! info "What it costs"
    Readability. The built-in $-i$, the `aap_sgn`, and the four-way write
    pattern in `set_matrix_element` are all consequences of packing a
    real-linear map on Hermitian matrices into real coordinates.

## The insertion convention (rule L6)

**The packed kernel implements $\\rho \\mapsto -i\\,(W\\rho)$.** A value `fct`
inserted at `(b, bp) <- (a, ap)` contributes `-1j * fct * rho[a, ap]` to
element `(b, bp)` of the result.

This explains a difference that otherwise looks like a physics disagreement:

- 1vN and Redfield pass their kernel entries directly — `kh.set_matrix_element(fct_aap, ...)`
- Lindblad passes `kh.set_matrix_element(1j*fct_aap, ...)`

The Lindblad dissipator carries no $i$ of its own, so it must cancel the one
built into the insertion path. **This is a convention artifact, not a
difference of physics.** (Assembling a random complex superoperator through
`set_matrix_element` and comparing against dense complex algebra confirms
`kern @ pack(rho)` matches `pack(-1j * W @ rho)` to `3.9e-16`; pinned by
`test_L6_packed_kernel_implements_minus_i_times_the_complex_action`.)

### Why the $-i$ must not be "cleaned up"

It is load-bearing. Because of it,

```
set_energy(E, b, bp, c) == set_matrix_element(complex(E, 0), b, bp, c, b, bp, c)
```

holds **exactly** — the two arrays are identical, with a difference of
`0.0`. The bare Liouvillian $-i[H,\\rho]$ and the tunnelling kernel share one
insertion path;
`set_energy` is a fast path, not a separate convention. Removing the $-i$ would
force an explicit `-1j` at every 1vN, Redfield and RTD call site and break that
unification.

This matters for anyone adding a new approach: **`L0` is not something you
build separately.** It already exists, assembled by the shared
`Approach.generate_kern` loop.

## The matrix-free path negates its imaginary rows

`KernelHandlerMatrixFree` does **not** produce the same vector as the assembled
kernel. Its output differs by a per-row sign on the imaginary block:

$$
\\texttt{dphi0\\_dt} = D \\cdot (\\texttt{kern} \\, @ \\, \\texttt{phi0}),
\\qquad
D = \\mathrm{diag}(+1 \\text{ on } [0, \\texttt{ndm0}),\; -1 \\text{ on } [\\texttt{ndm0}, \\texttt{ndm0r}))
$$

This holds exact to `2e-14` across `nsingle` 2–4 and `charge`/`sz` indexing.
End to end, `mfreeq=True` and `mfreeq=False` agree to about `1e-4` — the
iterative solver's tolerance — and Pauli, which has no imaginary rows, agrees
to `1e-11`, consistent with a discrepancy that is invisible where there are
no imaginary rows.

**Why this is not a bug for stationary solving:** scaling a row of a
homogeneous linear system does not change its null space, and the flip is
uniform across every contribution. So `kern @ phi0 = 0` and `dphi0_dt = 0` have
the same solutions.

!!! warning "Where it would bite"
    `dphi0_dt` is *not* the packed time derivative. Any future time
    propagation, residual diagnostic, or convergence measure built on the
    matrix-free path must account for the sign, or use the assembled kernel.

Pinned by `test_L6_matrix_free_negates_the_imaginary_rows`, which also asserts
the two are *not* equal — so correcting one path without the other cannot pass
silently.

## Trace is multiplicity-weighted (rule L8)

The obvious rule is wrong. The trace is **not** `sum(phi0[0:npauli])` in
general:

$$
\\mathrm{tr}(\\rho) = \\sum_{i<\\texttt{npauli}} m_i \\, \\texttt{phi0}[i]
$$

where $m_i$ counts the physical diagonal elements sharing stored index $i$.

- Under `indexing='charge'` and `'sz'`, every $m_i = 1$ and the plain sum is correct.
- Under `'ssq'` it is not. A stored index stands for a whole symmetry
  multiplet. Measured for `nsingle=4`, `ssq`: multiplicities
  `{0:1, 1:2, 2:2, 3:3, 4:1, 5:1, 6:1, 7:2, 8:2, 9:1}`, summing to `nmany = 16`
  across `npauli = 10` stored indices.

`Approach.generate_norm_vec` already builds this correctly — its
`norm_vec[bb] += 1` per many-body state accumulates exactly those
multiplicities. The rule was simply never written down, and a reimplementation
that "simplified" it to a plain sum would break `ssq` silently.

!!! danger "Consequence for new solvers"
    Any independent trace constraint, normalization check, or stationary-state
    diagnostic written against this layout must use the weighted form.

## The symmetry map is a surjection, not a bijection

Under `'ssq'`, several physical elements map to one stored index. Measured for
`nsingle=4`:

| indexing | carried elements | distinct stored indices |
|---|---|---|
| `charge` | 70 | 43 (each off-diagonal pair contributes 2 orientations) |
| `ssq` | 30 | 15 (plus genuine many-to-one identification) |

So `unpack(pack(x)) == x` holds only for states already in the symmetry-invariant
subspace. An adapter API must not promise a round trip in general.

## RTD uses a different coherence packing (rule L9)

`KernelHandlerRTD` inserts into `Wdn`, `Wnd` and `Lnn`, whose coherence axis is
**not** the `ndm0r` layout. There a coherence with reduced index `i` sits at
`i - npauli`, with its imaginary part at `i - npauli + imag_offset`, giving an
axis of length `2*(ndm0 - npauli)` and no population entries at all.

That layout also picks orientation by comparing `b > bp` directly rather than
consulting `conjdm0`. Under `indexing='charge'` — the only mode RTD supports —
the two agree, because `conjdm0` is true exactly when `b < bp` in `statesdm`
order.

Kept as-is and documented: legacy RTD depends on it.

## Open questions

- Whether the matrix-free row sign should be normalized to match the assembled
  kernel. It is currently harmless and pinned by a test; changing it touches
  both backends for no functional gain.
- Whether `booldm0` and `conjdm0` should be renamed. They are public,
  documented attributes, so named accessors were added instead of a rename.
