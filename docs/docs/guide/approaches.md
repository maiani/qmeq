# The approaches

QmeQ implements seven master-equation approaches, selected through the
`kerntype` argument to `Builder` (see [Getting started](getting-started.md)).
This page consolidates what each one approximates, what it solves for, its
validity domain, and its known failure modes — material that was previously
scattered across tutorial 6's validity table, the `qmeq/__init__.py`
disclaimer, and the RTD warning implementations.

## Overview table

Adapted from tutorial 4
(`examples/tutorials/04_coherence_and_approximations.ipynb`, "Choosing an
approximation"), with 2vN and RTDnoise added.

| approach | `kerntype` | keeps coherences | order in $\Gamma$ | solves for | use it when |
|---|---|---|---|---|---|
| Pauli | `'Pauli'` | no | first | populations only | eigenstates well separated ($\Delta E\gg\Gamma$) or protected by a selection rule |
| Lindblad | `'Lindblad'` | yes | first | populations + coherences | coherences matter and a positive density matrix is required |
| Redfield | `'Redfield'` | yes | first | populations + coherences | coherences matter; more complete energy dependence than Lindblad, at the cost of positivity guarantees |
| 1vN | `'1vN'` | yes | first | populations + coherences | coherences matter; retains more of the reservoir energy dependence than Redfield |
| 2vN | `'2vN'` | yes | second | energy-resolved `phi1(k)`, iterated to self-consistency | first-order transport is blocked (cotunnelling), or broadening/level-width effects matter |
| RTD | `'RTD'` / `'pyRTD'` | eliminated, not propagated | second | populations only | cotunnelling without the cost of 2vN's energy grid; needs `Γ ≪ T` and no near-degenerate same-charge states |
| RTDnoise | `'RTDnoise'` / `'pyRTDnoise'` | eliminated, not propagated | second | populations + first two current cumulants | the above, plus the zero-frequency current noise |

Valid `kerntype` strings are validated by `validate_kerntype`
(`qmeq/builder/validation.py`); the pure-Python/compiled name pairing (e.g.
`'RTD'` vs `'pyRTD'`) is set up in `qmeq/builder/builder_base.py`.

## Shared limitation: first-order methods need $\Gamma\ll T$

Pauli, Lindblad, Redfield, and 1vN are all first order in the tunnel coupling
$\Gamma$. None of them describes cotunnelling, level broadening, or Kondo
correlations — that requires a second-order approach (2vN or RTD). A
perturbative result is only trustworthy where the next order is negligible: a
genuine second-order feature scales as $\Gamma^2$, and one that scales faster
is dominated by physics the approach has dropped (demonstrated numerically in
tutorial 4's "Choosing an approximation" summary and tutorial 6's
$\Gamma$-scaling discussion).

## Per-approach notes

### Pauli

Classical rate equation over the populations of the many-body eigenstates
only (`get_kern_size` returns `si.npauli`). No coherences, so no interference
between transport paths. `principal_part` is always `'omit'` — there is no
principal-value contribution to add.

**Known failure mode:** when two eigenstates that both couple to the same
lead are not split by much more than $\Gamma$ (the failure criterion is
$2\Omega\lesssim\Gamma$), dropping the coherence between them is
uncontrolled — tutorial 4 measures the Pauli current at up to **40 times**
the coherent (Redfield/1vN/Lindblad) result at $2\Omega=0.08\Gamma$ in a
coherently-coupled double dot, which is a real qualitative failure, not a
small correction.

### Lindblad

Keeps coherences in Gorini-Kossakowski-Sudarshan-Lindblad form, which
guarantees the propagated density matrix stays positive. The principal-value
contribution is the **Lamb shift** — the lead-induced renormalization of the
dot's many-body energies — available via `principal_part='digamma'` (wide-band
digamma approximation only; `'quad'` raises `ValueError`, enforced by
`resolve_transport_options` in `qmeq/builder/validation.py` — there is no
numerical-quadrature Lamb shift).

**Known failure mode / limitation:** the guaranteed positivity is bought by
evaluating rates in a form that differs from Redfield/1vN — tutorial 4
measures Lindblad running **6-13% below** Redfield/1vN in a regime where all
three are valid. That gap is the price of the approximation, not a bug in
either. The Lamb shift itself is wide-band-digamma only (no `'quad'` option),
does not include a phonon-induced shift for the electron-phonon variant, and
stiffens the kernel — `legacy_docs/source/theory/lambshift.rst` recommends
checking the solution (e.g. `symq=False`, or comparing lead currents) for
weakly coupled models before trusting the last digits.

### Redfield and 1vN

Both keep coherences and go beyond Lindblad's secular treatment of the
reservoir correlation functions, at the cost of Redfield's and 1vN's
positivity guarantee — per the package disclaimer (see
[Overview](overview.md)), both "can violate positivity of the reduced density
matrix and lead to currents flowing against the bias." 1vN retains more of
the reservoir energy dependence than Redfield. In the regime tutorial 4 tests,
the two agree with each other to about 1%.

**Known failure mode:** positivity violation and against-bias currents are
possible outside their validity domain — the disclaimer names this
explicitly, and no runtime check in QmeQ currently flags it.

### 2vN

Second order. Solves an integral equation for the energy-resolved first-order
density matrix `phi1(k)` on a grid `Ek_grid` of `kpnt` points, iterated to
self-consistency (`niter` iterations; `system.iters` records each
`Iterations2vN` step). These attributes live on `ApproachBase2vN`
(`qmeq/approach/aprclass.py`). Unlike RTD, 2vN keeps *both* orientations of
every density-matrix element as independent complex unknowns
(`StateIndexingDMc`, `dtype = complexnp`) rather than reducing by Hermiticity,
so it does not need RTD's diagonal-density-matrix approximation.

**Validity / convergence controls:** two numerical controls must be checked
independently of the physics — convergence in `niter` and in `kpnt` (grid
density) — plus a physical requirement that `dband` be wide enough to resolve
the temperature scale across the band (tutorial 6). None of these convergence
checks certifies the *physical* accuracy of the second-order expansion
itself: tutorial 6 notes that "a converged 2vN result at $\Gamma\sim T$ is a
precisely computed approximation, not a precise answer."

**Known failure mode:** neither `bandwidth` nor `principal_part` is used by
2vN (`resolve_transport_options` raises `ValueError` if either is supplied
explicitly for `kerntype='2vN'`); indexing is restricted to `'Lin'` or
`'charge'` (`validate_indexing`, `qmeq/builder/validation.py`).

### RTD

Second-order Real Time Diagrammatics. Unlike 2vN, RTD **eliminates** rather
than propagates same-charge coherences — it solves only for populations
(`get_kern_size` returns `si.npauli`, same as Pauli), using an inverse
same-charge energy splitting (misleadingly named `Lnn`/`Lnn_inv` — see
[RTD kernel matrices](../conventions/rtd-kernels.md#lnn-does-not-hold-a-liouvillian))
to integrate the coherences out. It always uses `bandwidth='infinite'`,
`principal_part='digamma'` (`itype=1`) — enforced by
`resolve_transport_options`, which raises if a caller asks for anything
else — and only `indexing='charge'`.

**Validity domain**, per tutorial 6's "Validity, in one place" table:

| requirement | why |
|---|---|
| $\Gamma\ll T$ | perturbative in $\Gamma$; not Kondo physics |
| features scale as $\Gamma^2$ | otherwise dominated by neglected orders |
| `itype=1` / `dband` $\gg$ all energies | the kernel is derived in the wide-band limit |
| no near-degenerate states on one lead | RTD propagates a diagonal density matrix, i.e. it needs the eliminated-coherence approximation to hold |
| `indexing='charge'`, no `mfreeq`/`symmetry` | unsupported combinations |
| agreement with 2vN | different expansions agreeing is real evidence; either alone is not |

**Known failure modes:**

- **Unequal-temperature bandwidth cutoff.** RTD's published second-order
  integrals use `dband` as a finite wide-band *regulator* even though
  `bandwidth='infinite'` is selected. With unequal lead temperatures, QmeQ
  warns (`RTDBandwidthWarning`) when the smallest cutoff is below 1000x the
  largest transport scale, but the fix is "rerun with larger `dband` and check
  convergence" — there is no automated convergence answer yet.
- **Near-degenerate same-charge states.** RTD warns (`RTDCoherenceWarning`)
  when the closest same-charge splitting is within a factor of 5 of the
  Fermi-weighted sequential escape broadening. The diagnostic also reports
  `gamma_upper_bound`, the older occupation-independent spectral-width scale,
  but does not warn from that deliberately conservative bound. Separately,
  `RTDNoBroadeningWarning` reports when no sequential escape broadening exists
  for the active states — in that case the stationary kernel may be singular.
- **Complex tunnel amplitudes.** The energy and heat currents are filled with
  `nan` (with a warning) for models with flux or interference, because the
  derivation is unfinished (commented-out terms exist in `RTD.py`). The
  particle current is unaffected.
- **Discarded imaginary parts.** The population kernel assembly discards the
  imaginary part of a four-amplitude product at several sites. For the
  *particle* current this is benign: against an exact non-interacting result at
  generic plaquette flux, RTD converges at the expected third order in the
  coupling, matching the zero-flux case. It has not been checked away from the
  non-interacting limit, or for the energy current.

The validity table above is from tutorial 6.

### RTDnoise

The zero-frequency counting-statistics companion to RTD (`kerntype='RTDnoise'`
/ `'pyRTDnoise'`, pure Python only — there is no compiled `RTDnoise`). After
`solve()`, `system.current_noise` is `[I, S]` from the full fourth-order (in
$H_T$, i.e. second order in $\Gamma$) kernel; `current_noise_first` is the
sequential (lowest-order) result; `current_noise_o4trunc` gives both current
and noise at both orders for comparison; `current_noise_matrix` /
`current_noise_matrix_first` are the lead-resolved covariance matrices.

**Known limitations:**

- **Complex amplitudes are broken here, and silently.** With real tunnel
  amplitudes, `off_diag_corrections=True` includes the same eliminated-coherence
  block as ordinary RTD, resolved by lead and transferred charge for noise; the
  legacy `False` mode remains available. But RTDnoise's *own* second-order
  traversal conjugates one vertex factor instead of the whole amplitude
  product, so at a generic flux its second-order kernel is wrong — and the
  damage is not confined to the noise. Measured against the exact
  non-interacting reference at flux `pi/2`, the **particle current** loses the
  `O(Gamma**2)` accuracy it claims (residual exponent 2.0 instead of 3.0), with
  no warning raised. Ordinary RTD is unaffected on that path: its exponent
  stays 3.0 at the same operating point, so its second-order `.real`
  projections are an identity there rather than a truncation. Treat any
  RTDnoise number at complex amplitudes as unvalidated, including the current.
- **Requires a nonempty `countingleads`** and raises `ValueError` without one;
  matrix-free solving (`mfreeq=True`) raises `NotImplementedError`.
- **Laplace derivatives use two controlled paths.** The first-order blocks and
  bare coherence propagator are differentiated analytically, per-lead in
  `1/T`, with the reduction to the diagonal first-order kernel as the
  acceptance test. The explicit second-order direct/exchange integrals use a
  scale-relative centered derivative, with a step proportional to the largest
  energy or temperature scale; the assembled derivative is tested against an
  independent five-point stencil. The step is a pure fraction of the model's
  own energy scale with no absolute floor, so it is unit covariant: results do
  not change if the whole model is expressed in different energy units.
- **The noise inherits RTD's wide-band error as an `O(Gamma**2)` term.**
  Against the exact non-interacting reference, the corrected-noise residual is
  cubic in the coupling over a normal operating window but settles onto
  `Gamma**2` deep in the weak-coupling tail. That term is the finite-`dband`
  truncation, not a missing counting contribution: its coefficient falls
  roughly as `1/dband` (measured `4.7e-4`, `6.0e-5`, `7.9e-6` at `dband` `1e4`,
  `1e5`, `1e6`) and it is unaffected by converging the Matsubara/Ozaki pole
  count. Converging `dband` restores the cubic residual over the same window
  (exponent 2.62, 3.26, 3.28 at `dband` `1e5`, `1e7`, `1e8`). Treat `dband`
  convergence as a requirement for noise, not only for unequal temperatures.
- Inherits every RTD limitation above -- including RTD's own complex-amplitude
  limitation on the *energy* and *heat* currents, which is separate from the
  above: `WE1`/`WE2` keep only `gamma.real`, so with any significant
  `gamma.imag` both are filled with `nan` and a warning is raised rather than a
  wrong number returned -- and including the unequal-temperature
  `dband` requirement (RTDnoise counting calculations must also be repeated at
  increasing `dband` until convergence).
- Counting is not implemented for 2vN, electron-phonon approaches, or
  matrix-free solvers (any approach, not just RTDnoise).

The output-array semantics above (`current_noise_o4trunc` etc.) are set out
in `legacy_docs/source/theory/counting_statistics.rst`.

## Transport integration options, by approach

The `bandwidth` (`'finite'`/`'infinite'`) and `principal_part`
(`'quad'`/`'digamma'`/`'omit'`) options (or the legacy `itype` shorthand) are
part of what each approach's validity domain means in practice, per the
branch logic in `resolve_transport_options` (`qmeq/builder/validation.py`):

| approach | supported `(bandwidth, principal_part)` |
|---|---|
| Pauli | `(finite, omit)`, `(infinite, omit)` — no principal-value term exists |
| 1vN, Redfield | `(finite, quad)`, `(infinite, digamma)`, `(finite, omit)`, `(infinite, omit)` |
| Lindblad | `(finite, digamma)`, `(infinite, digamma)`, `(finite, omit)`, `(infinite, omit)` — `quad` is not implemented |
| RTD, RTDnoise | `(infinite, digamma)` only |
| 2vN | neither option is used |

Further reading on these options, including the legacy `itype` mapping and
worked examples: `legacy_docs/source/theory/transport_options.rst`.
