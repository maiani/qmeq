# The approaches

QmeQ implements seven master-equation approaches, selected through the
`kerntype` argument to `Builder` (see [Getting started](getting-started.md)).
This page collects what each one approximates, what it solves for, its
validity domain, how it is validated, and its known failure modes.

## Overview table

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
(`qmeq/builder/validation.py`). Every approach has a pure-Python form, named
with a `py` prefix (`'pyRTD'`, `'pyLindblad'`, ...), and the unprefixed name
selects the compiled form when the Cython backend is active; see
[INSTALL.md](https://github.com/qmeq/qmeq/blob/master/INSTALL.md#backend-selection).

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
small correction. At an exact degeneracy the Pauli result depends on which
basis of the degenerate subspace the diagonaliser returns;
`test_the_basis_of_a_degenerate_subspace_does_not_change_the_current` uses
this as its control, while every coherent approach is basis independent there.

### Lindblad

Keeps coherences in Gorini-Kossakowski-Sudarshan-Lindblad form, which
guarantees the propagated density matrix stays positive. The principal-value
contribution is the **Lamb shift** — the lead-induced renormalization of the
dot's many-body energies — selected by `principal_part`: `'digamma'` (wide-band
digamma form), `'quad'` (principal values integrated over the band), or
`'omit'` (no shift). `principal_part` has **no default** for Lindblad: a
Lindblad system without it raises `ValueError`. `principal_part='omit'` with
`itype=0` gives the kernel without a Lamb shift, which is QmeQ 1.1's Lindblad
kernel.

**Known failure mode / limitation:** the guaranteed positivity is bought by
evaluating rates in a form that differs from Redfield/1vN — tutorial 4
measures Lindblad running **6-13% below** Redfield/1vN in a regime where all
three are valid. That gap is the price of the approximation, not a bug in
either. The Lamb shift does not include a phonon-induced shift for the
electron-phonon variant, and
stiffens the kernel — the [Lamb-shift theory page](../theory/lambshift.md) recommends
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
possible outside their validity domain. QmeQ warns when the stationary state
has a negative population or a trace away from one (see
[Runtime diagnostics](#runtime-diagnostics)). A violation inside those
tolerances, or a current against the bias, is not flagged.

### 2vN

Second order. Solves an integral equation for the energy-resolved first-order
density matrix `phi1(k)` on a grid `Ek_grid` of `kpnt` points, iterated to
self-consistency (`niter` iterations; `system.iters` records each
`Iterations2vN` step). These attributes live on `ApproachBase2vN`
(`qmeq/approach/aprclass.py`). Unlike RTD, 2vN keeps *both* orientations of
every density-matrix element as independent complex unknowns
(`StateIndexingDMc`, `dtype = complexnp`) rather than reducing by Hermiticity,
so it does not need RTD's diagonal-density-matrix approximation.

**Convergence controls.** `niter` and `kpnt` are numerical controls, checked
independently of the physics. The grid spans the band, so at fixed `kpnt` its
spacing grows with `dband`: the band must be wide against the temperature and
the transition energies, and the grid fine against the temperature and the
level widths. Converge `kpnt` first, then `niter`. On the spinless double dot
of the 2vN equilibrium test (`hsingle={(0, 0): -0.3, (1, 1): 0.4, (0, 1):
0.2}`, $U=2$, $\Gamma=2\pi\cdot0.3^2\approx0.57$, $T=1$, `dband=20`, bias
$\pm0.5$), the left current measured against its converged value is:

| `kpnt` (`niter=8`) | 32 | 64 | 128 | 256 | 512 |
|---|---|---|---|---|---|
| relative error | 4.7% | 0.75% | 0.34% | 0.07% | 0.013% |

| `niter` (`kpnt=512`) | 1 | 2 | 3 |
|---|---|---|---|
| relative error | 2% | 0.08% | below $10^{-6}$ |

None of these convergence checks certifies the *physical* accuracy of the
second-order expansion itself: tutorial 6 notes that "a converged 2vN result
at $\Gamma\sim T$ is a precisely computed approximation, not a precise answer."

**Known failure mode: an equilibrium current at finite interaction.** At
equal chemical potentials and temperatures, 2vN carries a small current when
$U\neq0$. It is not a numerical error. On the model above at zero bias, the
measured current:

- is converged in `niter` (by 8 iterations), in `kpnt` (it changes by 0.2%
  when the grid doubles) and in `dband` (it approaches its limit as
  $1/$`dband`);
- grows from zero with $U$; and
- without interaction falls to zero as the grid is refined, which
  `test_2vN_equilibrium_current_vanishes_with_the_grid_without_interaction`
  checks.

It scales as $\Gamma^3$, against $\Gamma$ for the biased current. In that
model it is $4.3\times10^{-3}$ of the current at a bias of $T$, and the ratio
falls by 12.8, 15.6 and 15.9, approaching 16, for successive factors of 4 in
$\Gamma$. It is therefore attributed to the 2vN truncation at finite
interaction. The attribution rests on these measurements, of which the test
suite pins only the non-interacting half, not on a derivation. 2vN does not
resolve a current that is not large compared with the equilibrium current of
the same model.

**Supported options:** neither `bandwidth` nor `principal_part` is used by
2vN (`resolve_transport_options` raises `ValueError` if either is supplied
explicitly for `kerntype='2vN'`); indexing is restricted to `'Lin'` or
`'charge'` (`validate_indexing`, `qmeq/builder/validation.py`).

### RTD

Second-order Real Time Diagrammatics. Unlike 2vN, RTD **eliminates** rather
than propagates same-charge coherences — it solves only for populations
(`get_kern_size` returns `si.npauli`, same as Pauli), using an inverse
same-charge energy splitting (`Lnn_inv`; see
[RTD kernel matrices](../conventions/rtd-kernels.md#lnn_inv-does-not-hold-a-liouvillian))
to integrate the coherences out. It uses `bandwidth='infinite'` with
`principal_part='digamma'` (the legacy `itype=1`) and `indexing='charge'`.
Other explicit `bandwidth` or `principal_part` values raise `ValueError`; an
explicit `itype` other than 1, another indexing or `symmetry='spin'` is
replaced with a `QmeqWarning`; `mfreeq=True` raises `ValueError`.

`'pyRTD'` evaluates the shared enumeration of the population diagrams in
`qmeq.approach.rtd_diagrams`, which RTDnoise also uses. The compiled `'RTD'`
enumerates the same diagrams with its own loops in `c_RTD.pyx`, held to the
records by `test_compiled_rtd_matches_the_record_based_python_rtd`; the two
differ only in speed. See
[RTD kernel matrices](../conventions/rtd-kernels.md#one-diagram-enumeration-for-rtd-and-rtdnoise).

**Validity domain:**

| requirement | why |
|---|---|
| $\Gamma\ll T$ | perturbative in $\Gamma$; not Kondo physics |
| features scale as $\Gamma^2$ | otherwise dominated by neglected orders |
| `dband` $\gg$ all energies | the kernel is derived in the wide-band limit |
| no same-charge pair split by less than about five times its sequential escape broadening | RTD propagates a diagonal density matrix, i.e. it needs the eliminated-coherence approximation to hold |
| `indexing='charge'`, no `mfreeq` or `symmetry='spin'` | unsupported combinations |
| agreement with 2vN | different expansions agreeing is real evidence; either alone is not |

**Known failure modes:**

- **Unequal-temperature bandwidth cutoff.** RTD's published second-order
  integrals use `dband` as a finite wide-band *regulator* even though
  `bandwidth='infinite'` is selected. With unequal lead temperatures, QmeQ
  warns (`RTDBandwidthWarning`) when the smallest cutoff is below 1000x the
  largest transport scale. The warning does not supply the converged answer;
  see [Checking `dband` convergence](#checking-dband-convergence-at-a-thermal-bias).
- **Near-degenerate same-charge states.** RTD warns (`RTDCoherenceWarning`)
  when the closest same-charge splitting is within a factor of 5 of the
  Fermi-weighted sequential escape broadening, and records the case in
  `approach.rtd_coherence_diagnostics`, which also reports
  `gamma_upper_bound`, an occupation-independent spectral-width scale that
  does not trigger the warning, and `clamped_coherences`, the number of
  splittings small enough for the inverse to be clamped. Separately,
  `RTDNoBroadeningWarning` reports when no sequential escape broadening exists
  for the active states — in that case the stationary kernel may be singular.
- **Complex tunnel amplitudes.** RTD does not compute the energy and heat
  currents for models with complex tunnel amplitudes, such as models with a
  flux or with interference. Both are filled with `nan`, and a
  `QmeqRuntimeWarning` is raised. The particle current is unaffected.
- **Many-body input.** Two of the three energy-current kernels use the
  single-particle amplitudes. With `BuilderManyBody` input they are dropped,
  with a `QmeqRuntimeWarning`, and the energy and heat currents are wrong
  whenever a single-particle state couples to more than one lead. Assign
  `nsingle` and `tleads_array` on the system to restore them (tutorial 6).
- **Real parts of the four-amplitude contributions.** The population kernel
  adds twice the real part of each enumerated four-vertex contribution. This
  is the sum with its inverted partner, not a truncation; see
  [RTD kernel matrices](../conventions/rtd-kernels.md#the-second-order-real-is-a-partner-sum-not-a-truncation).

### RTDnoise

The zero-frequency counting-statistics companion to RTD (`kerntype='RTDnoise'`
/ `'pyRTDnoise'`). Both evaluate the shared diagram enumeration in Python;
`'RTDnoise'` selects compiled direct/exchange scalar functions when the Cython
backend is active, while `'pyRTDnoise'` stays all-Python. After `solve()`,
`system.current_noise` is `[I, S]` from the full fourth-order (in $H_T$, i.e.
second order in $\Gamma$) kernel; `current_noise_first` is the sequential
(lowest-order) result; `current_noise_o4trunc` gives both current and noise at
both orders for comparison; `current_noise_matrix` /
`current_noise_matrix_first` are the lead-resolved covariance matrices. Their
formulas and conventions are on the
[counting-statistics theory page](../theory/counting-statistics.md).

**Known limitations:**

- **Requires a nonempty `countingleads`**; `solve()` raises `ValueError`
  without one. `mfreeq=True` raises `ValueError` on construction.
- **Laplace derivatives use two controlled paths.** The first-order blocks and
  bare coherence propagator are differentiated analytically, per-lead in
  `1/T`, with the reduction to the diagonal first-order kernel as the
  acceptance test. The explicit second-order direct/exchange integrals use a
  scale-relative centered derivative, with a step proportional to the largest
  energy or temperature scale; the assembled derivative is tested against an
  independent five-point stencil. The step is a pure fraction of the model's
  own energy scale with no absolute floor, so it is unit covariant: results do
  not change if the whole model is expressed in different energy units.
- **Equal- and unequal-temperature bandwidth roles differ.** At equal lead
  temperatures the counted direct/exchange integrals use the same analytic
  Appendix-D wide-band real component as stationary RTD. Their individual
  `ln(dband)` terms cancel in the assembled zero-field kernel, so its
  stationary state, current, and noise are invariant under an auxiliary
  bandwidth sweep up to numerical roundoff. Unequal-temperature integrals use
  the Ozaki representation and must be checked for cutoff convergence; the
  noise converges more slowly than the current.
- Inherits every RTD limitation above, including the `nan` energy and heat
  currents for complex amplitudes.
- Counting is not implemented for 2vN, electron-phonon approaches, or
  matrix-free solvers (any approach, not just RTDnoise).

### What validates RTD and RTDnoise

Tests in `qmeq/tests/test_rtdnoise_physics_validation.py` grade the
second-order kernel against an exact non-interacting (NEGF) solver,
`qmeq/tests/noninteracting_negf_solver.py`, by the order in $\Gamma$ of the
residual rather than by a tolerance:

- with `off_diag_corrections=True`, the current residual is cubic in the
  coupling for real amplitudes; without the correction the current and noise
  residuals are quadratic
  (`test_noninteracting_residuals_have_the_expected_coupling_orders`);
- the corrected noise residual is cubic at a practical `dband` of 50
  (`test_corrected_noise_is_cubic_at_practical_bandwidth`), and at half the
  temperature
  (`test_temperature_half_noise_has_cubic_residual_in_calibrated_window`);
- at a generic plaquette flux, the RTD and RTDnoise currents and the noise
  have cubic residuals
  (`test_complex_flux_rtdnoise_observables_have_cubic_residuals`), and an
  orbital rephasing or a full $2\pi$ flux period changes no observable
  (`test_complex_flux_observables_are_invariant_under_orbital_rephasing`).

With interaction, a particle-hole-symmetric, spin-degenerate Anderson dot deep
in Coulomb blockade reproduces the elastic-cotunnelling current within
$2\times10^{-3}$
(`test_interacting_deep_blockade_matches_elastic_cotunnelling_current`). Its
noise is only checked to be finite: without an intrinsic spin-relaxation
bath the dot is in the strong-cotunnelling regime, where no Poisson identity
applies. Structural identities (column sums, conservation, equilibrium,
symmetric covariances, charge conservation of the counting labels) hold on
interacting and multi-lead systems (`test_rtdnoise_structural_invariants.py`,
`test_rtd_diagrams.py`).

Not graded: interacting systems outside deep blockade, splittings of order
$\Gamma$ or below (where the elimination is invalid by construction), noise
values at finite interaction, and the energy current for complex amplitudes.

### Checking `dband` convergence at a thermal bias

At unequal lead temperatures repeat the calculation at increasing `dband`
until every reported quantity stops changing. For a spin-degenerate level
between two pairs of leads at temperatures 0.1 and 0.3:

```python
import qmeq

t = 0.05
system = qmeq.Builder(
    nsingle=2, hsingle={(0, 0): 0.2, (1, 1): 0.2}, coulomb={(0, 1, 1, 0): 1.0},
    nleads=4, tleads={(0, 0): t, (1, 0): 0.8*t, (2, 1): t, (3, 1): 0.8*t},
    mulst={0: 0.1, 1: -0.1, 2: 0.1, 3: -0.1},
    tlst={0: 0.1, 1: 0.3, 2: 0.1, 3: 0.3},
    dband=1e2, kerntype="RTDnoise", countingleads=[0],
)
for dband in (1e2, 1e3, 1e4, 1e5):
    system.change(dlst=dband)
    system.solve()
    current, noise = system.current_noise
    print(f"dband={dband:.0e}  I={current.real:.7e}  S={noise.real:.7e}")
```

```text
dband=1e+02  I=-1.6892651e-05  S=1.8363638e-03
dband=1e+03  I=-1.6896548e-05  S=1.8344732e-03
dband=1e+04  I=-1.6896598e-05  S=1.8341920e-03
dband=1e+05  I=-1.6896599e-05  S=1.8341544e-03
```

Each decade of `dband` moves the current by $2\times10^{-4}$, then
$3\times10^{-6}$; the noise moves by $10^{-3}$, then $1.5\times10^{-4}$ and
$2\times10^{-5}$, so it sets the bandwidth needed. At equal temperatures the
result does not depend on `dband`, and no sweep is needed.

## Runtime diagnostics

QmeQ reports, rather than hides, results it cannot vouch for. Every
diagnostic is a `QmeqWarning` or `QmeqRuntimeWarning`, so filtering
`qmeq.QmeqWarning` silences them as a group.

- **Unphysical stationary states**, every approach: a negative population, a
  trace away from one, or a non-finite entry warns once per approach and is
  recorded in `approach.stationary_diagnostics`, with a `physical` flag.
- **RTD regime warnings**: `RTDBandwidthWarning`, `RTDCoherenceWarning` and
  `RTDNoBroadeningWarning`, described under [RTD](#rtd).
- **A finite band that silences a lead**: with `bandwidth='finite'`, a lead
  whose band excludes every transition it couples to carries no current.
- **A band the wide-band options ignore**: with `bandwidth='infinite'`, a band
  that excludes a transition its lead couples to is not applied. RTD and
  RTDnoise are exempt, because `dband` is their regulator.
- **Indexing that drops couplings**: under `'sz'` or `'ssq'` indexing,
  couplings that break the spin symmetry the indexing assumes.
- **RTDnoise Laplace projection**: `RTDNoiseLaplaceProjectionWarning` when a
  discarded real part of a Laplace derivative is not roundoff.

## Transport integration options

The `bandwidth` and `principal_part` options, which combinations each approach
supports, and the legacy `itype` mapping are described in
[Transport integration options](../theory/transport-options.md).
