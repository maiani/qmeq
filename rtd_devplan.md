# Development plan: RTD validation, counting, general systems, and coherence

Status: P0 and P1 are complete. P1's exit gate was reopened after the second
derivative audit and is now met: the finite-\(z\) projection, the
differentiation step, and the superseded historical values are all resolved,
and the residual \(O(\Gamma^2)\) noise term is identified as RTD's finite-`dband`
wide-band truncation rather than a defect. One convention is carried into P2 as its
own derivation item: the bare-resolvent orientation \(G_{nn}(z)=1/(\Delta E+z)\)
is gated by a negative control, not derived. P2 is next.

This is the single authoritative RTD development plan. It covers the validation
foundation, counting-resolved population kernel, arbitrary complex systems,
and the eventual full-coherence solver whose stationary unknown is
\(\rho=\{\rho_{ab}:N_a=N_b\}\), followed by a profile-driven compiled
evaluator for the shared implementation.

## 1. Objective

Proceed in five explicit priorities:

1. **P0: validate what exists** — characterize the current RTDnoise path,
   establish independent analytic and historical oracles, and turn every known
   invariant and defect reproducer into a test before changing production
   numerics;
2. **P1: add the off-diagonal correction with the smallest defensible change**
   — make zero-frequency noise available with `off_diag_corrections=True`, from
   the same effective population kernel that default `RTD` uses;
3. **P2: support complex amplitudes and arbitrary population-space systems** —
   repair and derive the conjugate/counting construction, remove topology- and
   real-amplitude assumptions, and only then unify duplicated traversals; and
4. **P3: implement the full-coherence approach** once the
   population/counting machinery is tested and general; and
5. **P4: add a profile-driven Cython evaluator** after RTD, RTDnoise, and the
   coherent approach consume one validated topology/record stream. The
   compiled path lowers and evaluates that shared representation; it must not
   introduce a separate handwritten RTDnoise topology.

Explicit non-goals across all priorities are cumulants beyond the second,
energy/heat-current *noise*, and finite-frequency noise. Through P2,
same-charge coherences remain **eliminated**, not solved for; P3 then promotes
them to the stationary unknown. P1 must carry the elimination term consistently
under the counting field without prematurely becoming a coherent solver.

### 1.1 Why this ordering

P0 is deliberately larger than a regression snapshot. It must distinguish
three things: historical compatibility, exact physics in the non-interacting
limit, and structural identities that remain valid at \(U\neq0\). A test that
only replays today's answer cannot certify today's answer.

P1 is a minimal *scope*, not a one-line change. At zero counting field QmeQ
already builds \(W_{dn}^{(1)}\), \(W_{nd}^{(1)}\), and their Schur-complement
correction. Noise additionally needs those blocks resolved by transferred
charge, plus their counting and Laplace derivatives. The minimal route is to
extract and label the existing first-order coherence construction while leaving
the second-order population traversal and real-amplitude behavior untouched.
It does **not** require the general diagram-record engine first.

P2 owns the broader refactor because complex amplitudes expose exactly where
implicit conjugate partners and duplicated traversal are unsafe. The current
`eta0=-1` defect must be fixed only after its transfer labels are derived from
the diagram rules, not merely from agreement of the summed kernel.

P3 follows the population work because the population-only elimination is invalid near
degeneracy; a full-coherence stationary unknown is the principled solution, not
another regularization of the Schur complement.

P4 comes last because compiling the current RTDnoise implementation would
preserve its duplicated second-order traversal as a third implementation beside
Python RTD and compiled RTD. Profiling and compilation become worthwhile only
after P2 and P3 establish the shared record format, scalar evaluators, and
assembly contracts that the compiled backend can reuse.

### 1.2 Reused infrastructure and full-coherence boundary

Reused as-is, already closed:

- the QmeQ 1.1 provenance corpus and its JSON/NPZ schema under
  `qmeq/tests/data/qmeq_11/`, regenerated only from pristine commit
  `96cc51076458b11f7db81a5d7d8df04c30bf8384`;
- the packed-real specification in `qmeq/approach/dm_layout.py` with its nine
  numbered rules, `LiouvilleState`, and `test_dm_layout.py`;
- `NO_INDEX`, `QMEQ_STRICT_INDEX=1`, and the `RtdMatrix` enum; and
- the standing hazard recorded there: inside `qmeq/approach/elph/`, a local
  `si` may be a `StateIndexingDMc`.

Deferred to P3, and explicitly *not* prerequisites for P0-P2:

- the full-coherence solvers `L0+W^{(1)}` and `L0+W^{(1)}+W^{(2)}` in `ndm0r`
  (P3.2 and P3.4 below);
- generic `dn`, `nd`, `nn` second-order blocks (P3.3); and
- the marked-vertex current kernel in the full Liouville space (P3.1).

Brought forward at different priorities:

- the convention sheet (section 3), restricted first to what the
  population projection and its first-order coherence elimination require;
- the immutable diagram-record architecture (P2 and P3.1), in P2
  rather than as a prerequisite for P1;
- analytic counting and Laplace derivatives (sections 3.1-3.2),
  restricted to the effective population kernel.

The diagram records introduced here must carry the fields P3.1 lists,
including arbitrary same-charge Liouville endpoints in their *type*,
even while this plan only ever asks for population endpoints and first-order
coherence endpoints. A record format that can only express `dd` would have to
be replaced rather than extended.

## 2. Current state

Line references are to the working tree at the time of writing.

### 2.1 G1 — counting omitted the default coherence correction (resolved in P1)

Before P1, `RTDnoise` raised `NotImplementedError` when
`off_diag_corrections=True`, although default `RTD` has that option **on**.
P1 now resolves the first-order population/coherence blocks by lead and signed
charge transfer, composes the correction in each transfer sector, and adds both
the correction and its Laplace derivative to the counting kernel. The
historical `False` mode remains supported and pinned by its original fixtures.

### 2.2 G2 — real projections in the diagram assembly

The second-order population assembly discards the imaginary part of the
four-amplitude product at every insertion site: `tempD.real` and `tempX.real`
at `qmeq/approach/base/RTD.py:716,719,729,732,749,752,762,765,782,790,799,807,821,829,838,846`
and the mirrored sites in `qmeq/approach/base/c_RTD.pyx`. The first-order
second-order-adjacent quantity `xcb` at `RTD.py:458` is likewise projected with
`.real`.

Two readings are possible and they are not both right:

- **benign** — each discarded imaginary part is cancelled by a conjugate
  partner diagram carrying the complex-conjugate amplitude product against the
  *same real* integral, so `Re` applied per contribution equals `Re` applied to
  the sum; or
- **lossy** — the conjugate partner is not generated, or is generated against a
  different integral value, and a genuine flux-dependent contribution is being
  dropped.

The `add_element_2nd_order` shortcut, which fills four population entries from
one evaluated contribution, is exactly where a partner can be implicitly
assumed. P2 therefore expands and validates those partners before P3 reuses
them in the full Liouville space.

Independent circumstantial evidence that this is not merely cosmetic: the
downstream benchmark that consumes this kernel restricts its operating points
to plaquette fluxes that are integer multiples of \(\pi\) with \(B_{1y}=B_{2y}=0\),
"so the tunnel-amplitude products multiplying \(\mathcal D\) and \(\mathcal X\)
are real in exact arithmetic". That restriction exists because the general case
is not trusted. Removing it is a goal of this plan.

Separately and definitely lossy: the energy-current blocks `WE1`/`WE2` keep only
`gamma.real` (`RTD.py:539,540,559,560,609,610,628,629`), set the `ImGamma` flag,
and then fill `energy_current` and `heat_current` with `nan`
(`RTD.py:409-420`). The commented-out `gamma.imag` terms at those sites are an
unfinished derivation, not a disabled feature.

Finally, `test_RTD_ignores_roundoff_scale_tunnel_phase` pins only that a
\(2\times10^{-18}\) phase does not perturb the current. It says nothing about a
physical phase.

### 2.2b Measured status of G2 (2026-08-23)

The two readings above were tested against the P0 reference solver as soon as it
existed, and the **benign reading wins for the particle current**. Recorded here
because it narrows the P2 audit.

Setup: spinless two-orbital dot, \(U=0\) so the reference solver is exact,
\(\varepsilon=(-1.0,0.7)\), \(\tau=0.4\), \(\mu=\pm1.5\), \(T=1\),
wide band, one lead phase set to \(\Phi_L=0.45\) — a generic flux, neither
\(0\) nor \(\pi\). Coupling scale halved five times. Residual against the
exact result, fitted exponent per halving:

| kernel | fitted exponent | reading |
| --- | --- | --- |
| Pauli, first order only | 2.00 | \(\Gamma^2\) physics absent, as expected |
| RTD, `off_diag_corrections=False` | 2.04 | \(\Gamma^2\) physics **incomplete** |
| RTD, `off_diag_corrections=True` | 3.00 | correct at the retained order |
| RTD, `off_diag_corrections=True`, \(\Phi_L=0\) | 3.00 | control |

Three conclusions.

**The real projections are not dropping flux-dependent \(\Gamma^2\) physics from
the particle current.** At generic flux the default kernel converges at exactly
the derived exponent, indistinguishable from the zero-flux control. The
enumerated `.real` sites are therefore an identity there, not a truncation —
each discarded imaginary part is cancelled by a conjugate partner, as the benign
reading of section 2.2 supposed. The audit in section 3.3 still has to *show*
why, but it is now documenting a correct implementation rather than repairing a
broken one.

**The off-diagonal correction is load-bearing, quantitatively.** Turning it off
degrades the exponent from 3 to 2 — the same exponent as dropping the entire
second-order block. So a kernel without it is not "slightly different"; it is
wrong at the order it claims to retain. This is the strongest available argument
that G1 is a defect and not a stylistic gap: `RTDnoise` computes noise from a
kernel that is demonstrably incomplete at \(O(\Gamma^2)\).

**RTDnoise's second-order kernel is wrong at complex amplitudes**, and the
imaginary part first noticed in `current_noise[0]` is a symptom rather than the
defect. Its second-order population kernel disagrees with RTD's by **63% at
\(\Phi=\pi/2\)** and 21% at \(\Phi=0.45\), agreeing to \(2.8\times10^{-6}\)
at \(\Phi=0\); the discrepancy scales as exactly \(\Gamma^2\) (constant to four
digits over a 16-fold range in coupling). Since RTD's kernel is the one verified
above, RTDnoise's is wrong — so its **current** is wrong too, not only its noise.

The mechanism is localized. `generate_col_diag_kern_2nd_order_lpm` enumerates an
`eta0 = +1` block and an `eta0 = -1` block, the second intended as the conjugate
partner of the first. Bucketing the handler's contributions by `eta0` shows the
intended relation \(S(-1)=S(+1)^\ast\) holds to \(2.9\times10^{-16}\) for real
amplitudes and is **96% violated** at \(\Phi=\pi/2\). The cause is in the
amplitude products: against `eta0 = +1`'s
`t1 * Tba[r1,a2p,a3p].conj() * Tba[r0,a3p,a0].conj()`, the `eta0 = -1` block
writes `... * Tba[r0,a0,a3p].conj()`, which — `Tba` being Hermitian in its state
indices — conjugates **one vertex factor** instead of the whole product. The two
coincide only when the product is real. A comment on that block already flagged
the area: *"other cases for p0,p3 still added through symmetry (eta screws it)"*.

The fix is defined and verified in principle: generate only `eta0 = +1` and write
its complex conjugate into the `eta0 = -1` slots, **keeping those slots' own
counting labels**, since the partner transfers the opposite charge and folding it
into the `+1` block would corrupt the transfer resolution. `2 Re[S(+1)]`
reproduces RTD's second-order kernel to \(2.6\times10^{-6}\) at every flux
tested — the same residual as at \(\Phi=0\), i.e. pure numerics. Two invariants
become the acceptance test, and both fail today: the `eta0`-summed `Lpm_second`
must be real (its imaginary part is presently 40% of scale), and its real part
must equal RTD's second-order kernel. `Lpm_first` is unaffected (`|t|^2`, real by
construction, imaginary part exactly zero); `Lpm_second_dz` inherits the same
complex-partner defect and is fixed by the same P2 change.

Scope of the claim, stated so it is not over-read. Tested: the *particle*
current, at \(U=0\), two leads, spinless, splitting \(\gg\Gamma\), one generic
flux, default RTD. Not tested, and therefore still open: the noise path itself;
\(U\neq0\), which the reference solver cannot reach; near-degenerate splitting; and the
energy current, whose `gamma.real` sites are a *separate* and definitely lossy
truncation (section 2.2, last paragraph) that this probe does not touch. G2 is
substantially narrowed, not closed.

### 2.3 G3 — fixed-step Laplace derivatives (resolved in P1)

The historical implementation shifted every propagator energy by a fixed
`lpm_h = 1e-8`, giving an observed relative roundoff floor of about
\(2.6\times10^{-8}\) in the calibrated double-dot kernel. P1 removes that
fixed step completely. First-order population/coherence blocks and the bare
coherence propagator use analytic derivatives. The complex Matsubara/Ozaki
direct and exchange integrals use a single scale-aware centered rule,

\[
h=\sqrt[3]{\epsilon_{\rm mach}}\max(1,|E_1|,|E_2|,|E_3|,|T_1|,|T_2|),
\]

because manually differentiating both branch-stabilized special-function sums
would duplicate their fragile analytic structure. The production full-kernel
derivative is checked against an independent five-point stencil using a
different step. Runtime derivative arrays use `_dz` to name the Laplace
variable explicitly.

### 2.4 G4 — duplicated traversal

`ApproachPyRTDnoise` subclasses `ApproachPyRTD` and re-implements
`generate_row_1st_order_kernel` and `generate_col_diag_kern_2nd_order` as
`..._lpm` variants. The two copies must agree on branch signs, orientation,
integral arguments, and fermionic signs, and nothing enforces that they do. Any
fix to one — including G2 — silently does not apply to the other.

### 2.5 G5 — no compiled counting path

There is no `c_RTDnoise`. `pyRTDnoise` is Python-only. This is acceptable for
now. P4 addresses the missing compiled counting path after the numerics and the
shared RTD/coherent architecture are settled, per the plan's "profile before
compiling" rule.

### 2.6 What is already good

`qmeq/approach/counting.py` provides `stationary_projected_pseudoinverse` and
`nonmarkovian_current_noise_matrix`, and the latter already implements the
projected non-Markovian covariance with the mixed energy/counting derivative
terms. The cumulant machinery is not the problem; its inputs are.

`qmeq/tests/test_counting.py:575` already compares a **spinless single resonant
level** against the exact scattering current and noise. That test is one P0
special case of the general reference solver.

## 3. Theory to derive before writing code

Each item below produces a written, reviewed section of the convention sheet.
Every generator or kernel test must cite one by name.

Primary-source roles are fixed up front:

- Leijnse and Wegewijs, [arXiv:0807.4027](https://arxiv.org/abs/0807.4027),
  defines the fourth-order real-time diagrammatics and the need for the complete
  density matrix; it is the source for vertex, branch, orientation, and
  conjugation rules.
- Emary, [arXiv:0902.3544](https://arxiv.org/abs/0902.3544), supplies the
  counting-field and non-Markovian cumulant construction used by RTDnoise.
- Klich, [arXiv:cond-mat/0209642](https://arxiv.org/abs/cond-mat/0209642),
  supplies the single-particle determinant identity underlying the independent
  non-interacting oracle.
- Thielmann *et al.*,
  [arXiv:cond-mat/0501534](https://arxiv.org/abs/cond-mat/0501534), and
  Kaasbjerg and Belzig,
  [arXiv:1504.01155](https://arxiv.org/abs/1504.01155), supply complementary
  interacting cotunnelling current/noise checks. They do not validate arbitrary
  interacting systems, but they prevent the \(U=0\) oracle from becoming the
  only arbiter.

### 3.1 Counting-resolved effective kernel

Fix the counting convention on the vertex records themselves, then derive the
effective population kernel with both the counting field and the Laplace
variable retained:

\[
W_{\rm eff}(\boldsymbol\chi,z)=
W_{dd}(\boldsymbol\chi,z)
+W_{dn}^{(1)}(\boldsymbol\chi,z)\,
G_{nn}(\boldsymbol\chi,z)\,
W_{nd}^{(1)}(\boldsymbol\chi,z).
\]

Establish, do not assume, the following order counting at the retained
truncation \(O(\Gamma^2)\):

- \(W_{dn}^{(1)}\) and \(W_{nd}^{(1)}\) are each \(O(\Gamma)\), so the product is
  \(O(\Gamma^2)\) already with the **bare** coherence propagator
  \(G_{nn}^{(0)}(z)=[z-L_{nn}]^{-1}\), \(L_{nn,(ab)}=i(E_a-E_b)\);
- any dressing of \(G_{nn}\) by \(W_{nn}^{(1)}\) enters the product at
  \(O(\Gamma^3)\) and is therefore **excluded** at this truncation;
- \(G_{nn}^{(0)}\) carries no counting field, so \(\partial_{\chi}\) acts only on
  the two vertex blocks; and
- \(\partial_z G_{nn}^{(0)}=-\,[G_{nn}^{(0)}]^2\), which is where the analytic
  Laplace derivative of the correction comes from.

If any of these fails on inspection, the failure is the finding and the plan
stops here.

Derive and record the four objects the second cumulant needs —
\(\partial_{\chi_r}W_{\rm eff}\), \(\partial_{\chi_r}\partial_{\chi_s}W_{\rm eff}\),
\(\partial_z W_{\rm eff}\), \(\partial_z\partial_{\chi_r}W_{\rm eff}\) — as
explicit expressions in the block derivatives, using the product rule and
\(\partial_z G^{(0)}=-[G^{(0)}]^2\). Do not obtain them by differentiating an
assembled zero-field matrix.

### 3.2 Where the real projection belongs under counting

This is the subtlest point in the plan and it must be settled analytically.

The present code applies `Re` to the assembled correction
(`RTD.add_off_diag_corrections`, which combines `ReWdn·Lnn_inv·ImWnd` and
`ImWdn·Lnn_inv·ReWnd`). Under a counting field that prescription is ambiguous,
because \(W_{\rm eff}(\boldsymbol\chi)\) is **not real** for \(\chi\neq0\): it is

\[
W_{\rm eff}(\boldsymbol\chi)=\sum_{\mathbf k}e^{i\mathbf k\cdot\boldsymbol\chi}
W_{{\rm eff},\mathbf k},
\]

and it is each **transfer-resolved component** \(W_{{\rm eff},\mathbf k}\) that
must be real, being the rate of a classical transfer process. Therefore:

- the real projection is applied per transfer sector \(\mathbf k\), **before**
  the \(\chi\) sum, and
- \(\mathrm{Re}\) and \(\partial_\chi\) do not commute in the naive sense;
  \(\partial_\chi\) brings down factors of \(ik\), so
  \(\partial_\chi\,\mathrm{Re}_{\mathbf k}[\cdot]\neq\mathrm{Re}[\partial_\chi\,\cdot]\).

Derive the correct statement, verify that it reduces to the existing zero-field
`add_off_diag_corrections` expression at \(\boldsymbol\chi=0\) exactly, and pin
that reduction as a test. A plan that gets this backwards produces noise that
looks plausible, conserves charge, and is wrong — the same failure mode the
downstream benchmark's frontier-model trials exhibited.

### 3.3 Complex amplitudes

For each `.real` site enumerated in section 2.2, determine which of the two
readings in that section holds, by hand-enumerating the conjugate partner set on
a minimal sector. Record per site: the partner diagram, whether the integral
factor is common, and therefore whether the projection is an identity or a
truncation.

Then state the physical invariants the repaired assembly must satisfy, which are
falsifiable independently of the enumeration:

- **gauge covariance** — rephasing dot many-body states, \(|a\rangle\to
  e^{i\theta_a}|a\rangle\), leaves every observable invariant;
- **flux dependence** — only gauge-invariant plaquette fluxes may enter
  observables, and the dependence must be \(2\pi\)-periodic in each flux;
- **reality of the assembled kernel** — the assembled population kernel is real
  and column-sum zero, \(\mathbf 1^TW^c=0\), per channel, for arbitrary complex
  amplitudes and not only for real ones; and
- **continuity** — observables are continuous in the flux through \(0\) and
  \(\pi\), so the existing real-amplitude fixtures are recovered as limits
  rather than as a separate branch.

The last one matters operationally: if the repair is correct, no existing
fixture at flux \(\in\{0,\pi\}\) moves. If a real-amplitude fixture *does* move,
that is a bug in the repair, not an expected consequence of it. Fixtures at
generic flux do not exist yet and are created by this plan.

For the energy current, derive the missing `gamma.imag` terms currently
commented out at `RTD.py:543,544,563,564,613,614,632,633`, or establish that
they vanish. This lifts the `nan` fill and closes the corresponding correctness item in
`TODO.md`. It is gated separately from the particle current and may ship later;
until it does, the `nan` and its warning stay.

### 3.4 Near-degeneracy policy

The elimination uses a clamped inverse splitting. The clamp is an ad hoc
regularization that P3.0 forbids in the full-coherence kernel;
here it stays, because the population-only projection genuinely breaks down at
\(|E_a-E_b|\lesssim\Gamma\) and clamping is not a repair for that.

Required instead: the clamp must be **observable**. When it engages, record it
in the stationary diagnostics and warn once, so that a user cannot silently
receive a number produced by the clamp, and so that the validation suite can
assert the clamp never fires in the regimes it grades. `qmeq/approach/diagnostics.py`
is the place for it.

## 4. The non-interacting reference solver

This is the new verification instrument and the reason the rest of the plan is
falsifiable. It must be written **before** the kernel changes it will judge, and
it must share no code with the diagrammatics — not the integrals, not the
indexing, not the many-body basis construction.

### 4.1 What it computes

At \(U_1=U_2=U_{12}=0\) the dot is a non-interacting single-particle problem and
the transport is exactly solvable at all orders in \(\Gamma\), for arbitrary
complex hopping, arbitrary non-collinear Zeeman fields, arbitrary bias, and
unequal lead temperatures. Build the single-particle Hamiltonian \(h\) on the
spin-orbital space, the lead coupling matrices \(\Gamma_\alpha\), and

\[
G^r(E)=\Big[E-h+\tfrac i2\sum_\alpha\Gamma_\alpha\Big]^{-1},
\qquad
\mathbf t(E)=\Gamma_L^{1/2}G^r(E)\Gamma_R^{1/2},
\]

with the scattering matrix \(S(E)\) assembled in the channel basis.

For the cumulants, prefer the determinant form of the counting generating
function over hand-coded current and noise formulas, because it handles the
benchmark's *weighted* observables (charge and \(S_z\), per lead) with one
implementation and no separate derivation per observable:

\[
F(\boldsymbol\chi)=\int\frac{dE}{2\pi}\,
\ln\det\!\big[\mathbb 1+n(E)\big(S^\dagger(E)\Lambda^\dagger S(E)\Lambda-\mathbb 1\big)\big],
\qquad
\Lambda=\mathrm{diag}\big(e^{ik_c\chi_c}\big),
\]

with \(n=\mathrm{diag}(f_c)\) over channels \(c=(\alpha,\sigma)\) and per-channel
weights \(k_c\) matching whichever observable is being counted. Currents and the
zero-frequency covariance are the first and second derivatives at
\(\boldsymbol\chi=0\), taken analytically where the algebra is tractable and
otherwise by complex-step or extended-precision differentiation — **not** by a
plain finite difference, which would reintroduce the G3 problem inside the
reference solver.

Energy integration must be adaptive and validated: the integrand has resonances
of width \(\Gamma\) and thermal structure of width \(T\), and the two can be
orders of magnitude apart in exactly the regime the convergence gates use.

### 4.2 Validating the reference solver itself

A new implementation is not a reference until it has been checked. Before it
grades anything:

- unitarity of \(S(E)\) at every quadrature node, to machine precision;
- \(F(\mathbf 0)=0\) and reality of the resulting cumulants;
- reduction to the closed-form Breit–Wigner result for the spinless single
  resonant level, i.e. reproduction of the existing
  `test_single_resonant_level_matches_exact_scattering` integrand to machine
  precision rather than to that test's `1e-2`;
- the Johnson–Nyquist limit: at zero bias, \(S_{LL}=2TG\) with \(G\) the linear
  conductance obtained independently by differentiating the current. The
  coefficient is convention-dependent and must be pinned, not quoted: with
  \(S=\lim_t d\,\mathrm{Var}X/dt\) and no extra factor of two, the Büttiker
  integrand at \(f_L=f_R=f\) gives \(S=(T/\pi)\int dE\,\mathcal T(-\partial_Ef)\)
  against \(G=(1/2\pi)\int dE\,\mathcal T(-\partial_Ef)\), hence 2 and not the 4
  of the one-sided convention;
- the Poisson limit: at large bias and small transmission,
  \(S_{LL}\to|I_L|\) (the spec's convention has no extra factor of two), so the
  Fano factor tends to unity; and
- the flux periodicity and gauge invariance of section 3.3, which the reference solver
  must satisfy by construction and which therefore double as a check that the
  reference solver's own phase conventions are consistent.

Only after all six pass does the reference solver acquire the **analytic**
trust level used by this plan's reference-data policy.

### 4.3 What it can and cannot decide

It can decide, at \(U=0\):

- whether the second-order population kernel has the right \(\Gamma^2\)
  coefficient, by the residual-scaling gate below;
- whether the new noise is right, in the same asymptotic sense — this is the
  only independent check on the new noise that exists;
- whether the complex-amplitude repair is right, because the reference solver is exact at
  arbitrary flux and the flux dependence is a sharp, non-generic signature;
- whether unequal-temperature results are right, since the reference solver has no
  bandwidth cutoff and no closed-form-integral restriction, which also gives the
  `dband` convergence TODO an actual answer instead of a warning; and
- whether the energy and heat currents are right once section 3.3 lifts the
  `nan`, since the Landauer forms of both are available from the same \(S(E)\).

It cannot decide:

- anything at \(U\neq0\) — for the deep-blockade cotunnelling regime a second
  analytic reference solver is needed, and the natural one is the Averin–Nazarov
  cotunnelling rate in the regime where sequential tunnelling is exponentially
  suppressed. Treat that as a complementary, separately provenanced reference,
  not as part of the \(U=0\) instrument;
- anything at splitting \(\lesssim\Gamma\) — there the population-only projection
  is outside its own validity, and disagreement with the exact result is
  *expected*. Those points are characterized as a known limitation and are
  explicitly not gates. They are the strongest available motivation for P3;
  and
- whether a *resummed* stationary solve is right at finite \(\Gamma\). Only the
  asymptotic statement below is a valid gate.

### 4.4 The residual-scaling gate

Comparing a truncated perturbative kernel against an exact result at a single
\(\Gamma\) requires an arbitrary tolerance. Comparing the *scaling* of the
residual does not, and is the primary gate.

For an observable \(O\), define \(R(\Gamma)=|O_{\rm RTD}(\Gamma)-O_{\rm exact}(\Gamma)|\)
along a coupling sweep at fixed everything else, spanning at least two decades
with the splitting held at \(\ge5\Gamma_{\max}\) throughout. Fit \(\log R\) against
\(\log\Gamma\) and require the measured exponent to match the derived one.

The expected exponent must be **derived per observable and per regime**, not
assumed. In the \(U=0\), resonance-accessible regime the sequential term is
\(O(\Gamma)\) and the truncation error is \(O(\Gamma^3)\), so the absolute
residual carries exponent 3 and the relative residual exponent 2. For a
truncation-order control, the same sweep run with the second-order block
disabled must show exponent 2, demonstrating that the \(\Gamma^2\) physics is
present rather than merely tolerated. Noise carries its own derived exponent,
which must be established before the sweep is run and not read off it.

Report the fitted exponent and its residual in the test failure message. A gate
that only says "not close enough" is much less useful here than one that says
"expected slope 3, measured 2.02", which localizes a missing second-order term
immediately.

### 4.5 Where the reference solver lives: test-only now, not an `Approach` ever

Decision: **test-only for this plan.** A public, user-facing non-interacting
reference is a defensible follow-up, but it must not be a `kerntype`, and it must
not be built before the reference solver has finished grading P1 and P2.

Three reasons, in decreasing order of weight.

**Independence is the reference solver's entire value.** Section 4 requires it to share no
code and no conventions with the diagrammatics. An `Approach` is constructed by
`Builder`, so it would inherit `StateIndexingDM`, `FunctionProperties`, the
`leads` object, and the rotated `Tba` amplitudes. Every one of those is a shared
layer in which a sign or phase convention could be wrong — and a convention bug
in a shared layer is exactly the class of bug that a comparison through that
layer cannot see. The reference solver must take a bare single-particle Hamiltonian,
bare \(\Gamma_\alpha\) matrices, chemical potentials, temperatures, and channel
weights as plain arrays, and construct its own basis. Anything that makes it more
convenient makes it worth less.

**It does not fit the contract.** `Approach` is a density-matrix interface:
`get_kern_size`, `prepare_kernel_handler`, `generate_kern`,
`generate_coupling_terms`, `generate_norm_vec`, `solve_kern`, `phi0`, `kern`.
NEGF has none of these. It works in the single-particle space, not the many-body
Fock space; there is no kernel, no stationary vector, and no `si` indexing.
Shipping it as a `kerntype` means either fabricating a kernel or scattering
`NotImplementedError` across the interface, and the second option leaks into
`aprclass.py` for every consumer that reasonably assumes an approach has a
`kern`.

**Its validity restriction is a trap in a public API.** It is exact at
\(U_1=U_2=U_{12}=0\) and wrong otherwise. A `kerntype='NEGF'` accepting a
`coulomb` argument would silently mislead; rejecting nonzero `coulomb` makes it
an approach that refuses most QmeQ models. Either way it does not belong in a
list whose other entries are general-purpose approximations.

Two smaller practical points. The reference solver wants extended-precision or complex-step
differentiation and adaptive quadrature over resonances that can be orders of
magnitude narrower than the thermal scale; `mpmath` is not currently a QmeQ
runtime dependency and would only be a `test` extra. And shipping means a public
API commitment, backend-parity questions, and a changelog entry for an instrument
whose purpose is to validate work that has not landed yet.

If it is later promoted — and the general \(U=0\) limit is genuinely useful
beyond RTD, since 1vN, Redfield and 2vN have the same validation gap — then:

- expose it as a function-style module, `qmeq.reference.noninteracting` or
  similar, never as a `kerntype`;
- keep the array-level core as the thing the tests grade against, and make the
  public entry point a thin wrapper over it, so promotion adds a convenience
  layer without putting `Builder` conventions between the reference solver and the kernel
  it judges; and
- pin the wrapper against the core in a test, so a future convenience-layer
  convention change cannot silently weaken the independence claim.

The reverse order — ship it publicly, then use the public form as the reference solver —
is the failure mode to avoid, because the coupling it introduces is invisible
once established.

## 5. Priorities and exit gates

Each priority is a sequence of small, reviewable changes. The next priority
starts only when the previous exit gate passes. Documentation-platform migration
is complete and out of scope; `legacy_docs/` is merely retained until separate
housekeeping removes it. Physics documentation changed by this work is still
part of the relevant P1-P4 change.

### P0 — Validate the current implementation

No production numerics change in P0. Its purpose is to establish what is
historical behavior, what is independently correct, and what is already known
to be wrong.

#### P0.1 Historical baselines

1. Keep the QmeQ-1.1 RTD bundle, the legacy bundle, and the pinned counting
   bundle separate: their provenance and trust levels are different.
2. Store real-amplitude RTDnoise internals and observables:
   `Lpm_first`, `Lpm_second`, and both historical `*_dot` fixture arrays,
   order-separated stationary states, current, and noise. The old suffix is an
   immutable bundle key only; runtime arrays now use `*_dz`.
3. Do not store complex-amplitude outputs as references. Preserve the models as
   live invariant/defect tests, because recording a known-wrong value would turn
   a bug into a compatibility contract.
4. Regeneration remains explicit; routine tests only read immutable NPZ data and
   validate manifest schema, source revision, scenario names, and array shapes.
5. This plan is coordination material, not fixture provenance. Code, tests,
   generators, and manifests must not link to this file or embed its phase/gap
   labels; they record durable source paths, revisions, equations, model inputs,
   array conventions, and trust classifications directly.

#### P0.2 Independent non-interacting oracle

The test-only NEGF solver remains independent of `qmeq.builder`,
`qmeq.approach`, many-body indexing, and RTD integrals. Its array-level core is
self-validated by unitarity, Breit-Wigner reduction, Johnson-Nyquist, Poisson and
symmetric-barrier limits, gauge invariance, flux periodicity, conservation, and
integration-window convergence.

The QmeQ bridge is tested separately so convenience cannot weaken independence:

- QmeQ `tleads` are translated by
  \(g=\sqrt{2\pi}\,t^\ast\);
- the complex width matrix is checked directly, including phases; and
- a public-input resonant-level model is solved both by QmeQ RTDnoise and the
  oracle, pinning current and noise normalization end to end.

Use this oracle as an **exact \(U=0\), wide-band test oracle**, not as a
replacement implementation and not as evidence for \(U\neq0\).

#### P0.3 Validation gates

1. Both forced backends load every immutable bundle in separate processes.
2. First- and second-order population blocks are separable in tests; agreement
   of the sum alone is insufficient.
3. Column-sum zero holds per lead and per retained order.
4. Counted current agrees with the ordinary current, including vanishing
   imaginary part where the physical result is real.
5. Equilibrium current vanishes; covariance is symmetric; aggregate noise equals
   the independently recomputed sum of lead-resolved entries. The test must not
   compare an assignment to the same expression that produced it.
6. The historical fixed-`lpm_h` derivative is checked over a step sweep with
   Richardson extrapolation, recording the pre-P1 accuracy floor.
7. Real-amplitude historical scenarios are unchanged on Python and Cython.
8. The complex-flux RTDnoise defect is a strict expected failure or a direct
   reproducer: it may not disappear unnoticed during unrelated work.
9. At \(U=0\), coupling sweeps distinguish exponent 2 from exponent 3 for
   current; the corresponding expected exponent for noise is derived before it
   becomes a gate.
10. Add at least one interacting deep-blockade current/noise check from an
    independently derived cotunnelling limit. The NEGF oracle cannot cover it.

P0 exit gate: all non-xfail validation passes on both backends; the NEGF bridge
tests pass; every expected failure names a known defect and the priority that
will close it; no production RTD/RTDnoise file has changed.

**Status: complete (2026-08-24).** The counting bundle is pinned to source
commit `aa1af46dd687c271505d28dbfb7ccce03a8a1739` and the permanent annotated
tag `reference/rtdnoise-source`. Attribution is retained in the manifest's
provenance metadata; neutral names are used for tests, paths, and CI selectors.

The implemented gates cover all ten items above. Measured calibration points:

- the fixed `lpm_h=1e-8` second-order derivative has a relative error of about
  (2.6\times10^{-8}) in the double-dot control; decreasing the step to
  (10^{-9}) makes roundoff worse, while two independent Richardson pairs
  agree within (10^{-6}) relative;
- in the generic-flux (U=0) sweep, the NEGF current residual scales as
  approximately (Gamma^3) with the existing RTD off-diagonal correction and
  approaches (Gamma^2) without it; the RTDnoise noise residual scales as
  (Gamma^{1.99}), as expected when an order-(Gamma^2) counting correction
  is omitted;
- the complex-flux reproducer agrees with ordinary RTD within (10^{-6}) for
  real amplitudes, but has a (0.69) relative kernel mismatch at flux
  (pi/2), so the known defect cannot disappear silently; and
- the interacting particle-hole-symmetric Anderson-dot check agrees with the
  independently derived elastic-cotunnelling current within (0.02\%\) and
  bidirectional-Poisson noise within (1\%\).

The complete focused matrix passes in separate forced processes: 156 tests on
Python and 156 tests with the compiled Cython backend active. Production files
under `qmeq/approach/` are unchanged. P1 may begin from this baseline.

### P1 — Minimal counting-resolved off-diagonal correction

P1 changes only what is necessary to carry QmeQ's existing first-order
coherence-elimination correction into the counting kernel. It does not unify
the second-order traversal, repair general complex amplitudes, add a compiled
RTDnoise path, or change the full stationary unknown.

1. Write the convention sheet for first-order \(dn\) and \(nd\) blocks: endpoint
   layout, branch sign, lead label, signed transferred charge, and Laplace
   dependence. Derive
   \[
   W_{\rm corr}(\boldsymbol\chi,z)=
   W_{dn}^{(1)}(\boldsymbol\chi,z)G_{nn}^{(0)}(z)
   W_{nd}^{(1)}(\boldsymbol\chi,z)
   \]
   and its first/second counting and first/mixed Laplace derivatives.
2. Extract the existing first-order coherence traversal into a reusable emitter.
   Its zero-field consumer must reproduce existing `Wdn`/`Wnd` arrays before
   any counting consumer is enabled.
3. Add a transfer-resolved consumer for that emitter. Apply the real projection
   per transfer sector, and pin exact reduction to
   `add_off_diag_corrections` at \(\boldsymbol\chi=0\).
4. Add the correction to the effective counting kernel and remove
   RTDnoise's rejection of `off_diag_corrections=True`.
   `off_diag_corrections=False` remains a supported compatibility mode.
5. Prefer analytic \(\chi\) derivatives. For \(z\), use the analytic
   \(\partial_zG_{nn}^{(0)}=-[G_{nn}^{(0)}]^2\). Differentiate the
   first-order blocks analytically; keep numerical differentiation localized to
   the explicit complex direct/exchange integral functions, with one
   scale-aware centered rule tested against an independent higher-order
   stencil.
6. Make clamp engagement visible in diagnostics and exclude clamped points from
   accuracy gates.

P1 exit gate:

- at zero counting field, the corrected RTDnoise population kernel and
  stationary current equal default RTD within the established numerical floor;
- at \(U=0\), current and noise with `off_diag_corrections=True` pass the
  derived NEGF residual-scaling gates for real amplitudes;
- `off_diag_corrections=False` reproduces every historical counting fixture;
- counted and ordinary currents agree, conservation and equilibrium tests pass,
  and both forced backends agree; and
- no P2 complex-amplitude expected failure is weakened or relabelled as passing.

**Status: implementation present; exit gate met (2026-08-24, after the second derivative audit).** The shared
`qmeq.approach.rtd_blocks` traversal now serves ordinary RTD and the
counting-resolved correction without extending the historical `RtdMatrix`
selector. Transfer sectors use
\(q=N_{\rm final}-N_{\rm initial}\), positive into the quantum dot. Their
zero-field sum reproduces ordinary RTD's kernel and current to the numerical
floor. The correction's block and bare-resolvent derivatives are analytic;
every explicit direct/exchange integral derivative uses the scale-aware
centered rule described in section 2.3. All live derivative quantities use the
`_dz` suffix.

The second audit confirmed three corrections: first-order Laplace derivatives
carry the per-lead chain-rule factor `1/T`; the population-coherence derivative
contains both the Fermi and `phi` channels and reduces to the diagonal kernel;
and structurally zero coordinate pairs more than one charge apart are skipped
without weakening the single-electron selection rule.

The former real-amplitude noise scaling claim is not an exit gate. Extending
the \(T=1\) sweep toward weaker coupling gives successive local residual slopes
2.33, 2.17, and 2.08, approaching an \(O(\Gamma^2)\) term. The three reopened
conditions are now closed as follows.

**Finite-\(z\) projection and bare-resolvent orientation — closed.** The stored
projection was wrong, not merely unpinned. The zero-field
Schur product is purely imaginary and its Laplace derivative purely real, so
`1j*product_dz.imag` selected the identically zero channel and
`coherence_correction_dz` was zero to machine precision. The correction and its
derivative are one analytic object, \(W_{\rm corr}(z)=-i\,W_{dn}(z)G_{nn}(z)
W_{nd}(z)\), which also satisfies the counting path's convention of real
kernels and purely imaginary `_dz` arrays — required for a real noise. The
composition is now gated against an independent finite-\(z\),
transfer-resolved reference built from the term decomposition, itself anchored
element by element against the untouched historical value expressions. The
correction's derivative enters the noise only at \(O(\Gamma^3)\), which is
why no pre-existing gate could see any of this. The bare-resolvent orientation
\(G_{nn}(z)=1/(\Delta E+z)\) now follows from
[LeijnseWegewijs2008, Eqs. (19), (49)] after mapping QmeQ's energy-like
continuation and ordered RTD coherence slots. The finite-\(z\) gate retains the
opposite orientation as a strict negative control.

**Derivative step — closed.** The absolute `max(1, ...)` floor is replaced by a
pure fraction of the model's own energy scale. The energy-rescaling covariance
test now sweeps \(\lambda\) from \(7\) down to \(10^{-3}\), far below the
removed floor, and fails if the floor is restored.

**Superseded historical values — closed.** No table measured from the corrected
tree is used. The three affected non-Markovian observables are checked for
finiteness only; `Lpm_first_dot` is compared as an exact per-lead rescaling of
the pinned array, which reproduces the historical bundle *and* the precise form
of the defect, including in the unequal-temperature scenarios where no single
global factor can match. Their physics is gated by the unequal-temperature
non-interacting reference and by rescaling covariance.

**The \(O(\Gamma^2)\) noise term is identified and is expected.** It is RTD's
own finite-`dband` wide-band truncation. Its coefficient falls roughly as
\(1/\)`dband` — measured \(4.7\times10^{-4}\), \(6.0\times10^{-5}\),
\(7.9\times10^{-6}\) at `dband` \(10^4\), \(10^5\), \(10^6\) — and the
exponent recovers toward 3 as `dband` grows. Two candidates are excluded by
measurement: converging the Matsubara/Ozaki pole count from 190 to 397 to 1121
leaves the exponent decaying toward 2 (while the second-order kernel then
matches ordinary RTD's to \(1.2\times10^{-10}\)), and zeroing
`coherence_correction_dz` moves the residual by under 3% without moving the
exponent. The non-Markovian term cancels better than 99.9% of the Markovian
\(O(\Gamma^2)\) error at both temperatures tested.

**The cubic claim is therefore recovered, not withdrawn.** At a converged
`dband` the corrected-noise residual is cubic again over the original coupling
window: exponent 2.62 at `dband` \(10^5\), 3.26 at \(10^7\), 3.28 at
\(10^8\) for \(T=1\), and 2.99 / 3.01 / 3.01 for \(T=0.5\) — converged in
`dband` and stable in temperature. The exponent assertion is not widened at the
contaminated `dband`; it is moved to the converged one, with the \(1/\)`dband`
scaling and the Markovian cancellation fraction as the cheap companions.
`dband` convergence is now a documented requirement for noise, not only for
unequal temperatures.

This work is tagged `1.2.0.dev2`. The reopened conditions are closed and the
fourth finding is expected rather than a defect.

The localized generic-complex-amplitude defect was repaired on 2026-08-31.
RTDnoise now completes the `eta0=-1` value and Laplace-derivative diagrams from
the conjugate and negative-conjugate partners of the independent `eta0=+1`
sector, retaining each partner's counting coordinates. Generic-flux
transfer-sector reality, RTD/RTDnoise kernel agreement, cubic current/noise
residuals against the independent \(U=0\) solver, orbital rephasing, and
\(2\pi\)-periodicity pass on both selected backends. P2 remains open for the
arbitrary-system matrix and traversal unification below; this repair does not
qualify complex-amplitude energy/heat currents.

### P2 — Complex amplitudes and arbitrary population-space systems

P2 removes assumptions that are harmless only for real amplitudes or special
topologies.

1. **Completed 2026-08-31:** derive the conjugate partner of every second-order
   diagram directly from the Leijnse-Wegewijs branch/orientation rules and
   derive its counting transfer from Emary's vertex factor. Do not infer the
   transfer label from the summed real kernel.
2. **Completed 2026-08-31:** repair RTDnoise's `eta0=-1` construction so the
   whole amplitude product is conjugated with the derived partner endpoints and
   labels. The existing
   observation \(2\operatorname{Re}S(+)\simeq W_{dd,\rm RTD}^{(2)}\) is a value
   check, not sufficient evidence for the transfer-resolved mapping.
3. **Completed 2026-08-31:** gate the repair with
   per-transfer conjugation identities, reality of the summed population
   kernel, RTD/RTDnoise second-order agreement, counted/ordinary current
   agreement, NEGF current/noise residuals at generic flux, gauge invariance,
   and independent \(2\pi\) periodicity.
4. **Completed 2026-08-31:** generalize the scenario matrix: more than two
   orbitals, more than two leads,
   rank-deficient and dense lead couplings, complex hopping, non-collinear
   fields, unequal lead temperatures, and interacting cases for structural
   invariants. “Arbitrary” means no hard-coded DQD, two-terminal, spin, or
   real-gauge assumption; it does not mean exact validation at arbitrary
   interaction.
5. Only after those semantics are pinned, introduce the immutable diagram record
   and one traversal for physical and counting-resolved population kernels.
   Activate it in shadow mode, compare each transfer-resolved block, then route
   production through it; delete legacy duplication in a later change.
6. Derive the missing complex energy/heat-current terms or retain the current
   `nan` plus warning. Energy/heat support has a separate gate and must not
   block correct particle current/noise.
7. **Completed 2026-08-31:** derive the bare-resolvent orientation
   \(G_{nn}(z)=1/(\Delta E+z)\) from [LeijnseWegewijs2008, Eqs. (19), (49)] and
   QmeQ's ordered RTD coherence packing. The finite-\(z\) reference retains the
   opposite orientation as a negative control.
8. Profile before adding a compiled counting evaluator.

P2 exit gate: all complex-amplitude defect reproducers pass on both backends;
real-amplitude fixtures are unchanged; generic-flux \(U=0\) current and noise
pass the NEGF scaling gates; arbitrary-system structural tests pass; and a
single validated traversal serves RTD and RTDnoise, or any remaining duplication
has an explicit measured reason.

### P3 — Full-coherence RTD

P3 promotes populations and every same-charge coherence to the stationary
unknown. It is not a different diagram theory: it consumes the records,
transfer labels, and scalar evaluators validated in P0-P2, but assembles them
on the complete `dm0` Liouville space instead of eliminating the `nn` sector.

#### P3.0 Approximation and safety contract

The two target truncations are `L0 + W1` and `L0 + W1 + W2`, where `W1` is
`O(Gamma)` and `W2` is `O(Gamma^2)`. Keep two statements distinct:

1. the kernel is truncated at a stated order; and
2. the stationary null vector of that truncated kernel is solved exactly.

The second statement resums higher powers through the solve and must not be
described as an order-by-order density matrix. Before the second-order solver
is enabled, document whether one full Liouvillian is the intended approximation
for well-separated states or whether only a derived slow subspace should retain
coherences.

Full coherence removes the bare inverse-splitting elimination, so no energy-
splitting clamp is permitted. It does not guarantee a unique state. Diagnostics
must expose nullity, a condition estimate or smallest nonzero singular value,
stationary residual, trace error, Hermiticity error, and the minimum density-
matrix eigenvalue. Positivity is diagnostic information, not an automatic
repair.

Legacy `RTD`/`pyRTD` and `off_diag_corrections=False` RTDnoise behavior remain
compatibility boundaries. The coherent solver is a new opt-in approach; it must
reject `off_diag_corrections` rather than silently ignore or double-count it.

#### P3.1 Canonical full-Liouville architecture

Use one canonical Python topology generator. A suitable internal dependency is:

```text
StateIndexingDM -> Liouville adapter -> diagram generator
                                      -> scalar evaluator -> assembler -> solver
```

Diagram generation must not write directly into matrices. Each immutable
record carries at least:

- initial, intermediate, and final Liouville states;
- perturbative order and direct/exchange topology;
- ordered vertices with branch, electron/hole orientation, lead, and many-body
  transition;
- reservoir contraction pairing, tunnelling product, and fermionic sign;
- propagator energies, integral kind, and explicit Laplace dependence;
- lead attribution and signed transfer at every reservoir vertex; and
- an identity/multiplicity that makes conjugate and symmetry partners auditable.

The evaluator maps a validated record to a scalar; the assembler maps endpoints
to the canonical packed-real coordinates. `dd`, `dn`, `nd`, and `nn` are
derived projections used for diagnostics, never separate topology rules.
Cython may later evaluate lowered record tables, but must not become a second
hand-written topology generator.

Particle current is generated by marking a reservoir vertex on the same record
stream and evaluating `I_r = <1| W^I_r |rho_stat>`. Its population projection
must reproduce the legacy current sign and normalization. Energy and heat
currents require separate derivations and must fail clearly until available.

#### P3.2 First-order full-coherence solver

1. Assemble `L0 + W1` in `ndm0r` with the normal QmeQ trace constraint.
2. Expose `L0`, lead-resolved `W1`, total kernel, stationary state, and
   diagnostics separately.
3. Generate the lead-resolved marked particle-current kernel and opt-in
   zero-frequency covariance from the same records.
4. Differentiate the full Liouvillian directly in counting and Laplace
   variables; do not reuse the population Schur complement.
5. Keep the approach name provisional until its validation gate passes.

First-order gates:

- exact single-level and Pauli reductions where coherence blocks vanish;
- comparisons with 1vN and Redfield only in explicitly derived common limits;
- trace and Hermiticity preservation for a basis of Hermitian trial states;
- equilibrium zero current and stationary charge conservation;
- rephasing covariance and unitary covariance inside an exactly degenerate
  same-charge subspace;
- regular exact- and near-degenerate behavior when the stationary state is
  unique;
- counted current equal to the marked-vertex current and covariance symmetric
  under lead exchange; and
- no movement in any population-only compatibility fixture.

#### P3.3 Complete second-order coherence blocks

1. Permit arbitrary same-charge initial and final Liouville states in every
   four-vertex template.
2. Obtain `dd`, `dn`, `nd`, and `nn` from the same generated records, retaining
   transfer labels and analytic Laplace derivatives.
3. Expand legacy symmetry shortcuts into explicit records with checked identity
   and multiplicity.
4. Hand-enumerate minimal sectors covering every branch/orientation and both
   direct/exchange topologies; verify signs and conjugate partners without
   using the production generator as its own oracle.
5. Compare selected entries with a deliberately slow independent enumerator or
   symbolic calculation carrying explicit provenance.
6. Require trace and Hermiticity preservation blockwise before any stationary
   solve is attempted.

P3.3 exits only when every topology counter is exercised, independent spot
checks pass, and the `dd` projection still reproduces the P2 population engine.

#### P3.4 Second-order full-coherence solver and counting

1. Assemble and solve `L0 + W1 + W2` with one trace constraint and the P3.0
   diagnostics.
2. Keep `W1`, `W2`, and their lead-resolved forms observable; do not retain only
   their sum.
3. Add coupling sweeps that distinguish retained kernel orders from the implicit
   resummation of the stationary solve.
4. Generate particle current consistently through second order.
5. Generate non-Markovian current/noise covariance from derivatives of the full
   Liouvillian, and compare its large-splitting Schur reduction with the P1
   corrected population result.
6. Until energy/heat-current coherence terms are derived, raise a clear
   `NotImplementedError` rather than returning population-only approximations.

#### P3.5 Profiling hand-off and public API

Profile diagram generation, integral evaluation, assembly, memory, and solve
time, and record the representative workloads and hot-path boundaries that P4
will use. Do not add a second topology generator for performance.

Once validated, make counting opt-in on the population and coherent RTD
approaches. Keep `RTDnoise`/`pyRTDnoise` only as compatibility aliases with
their sequential and consistently order-truncated diagnostics. Selecting
counting must not change the zero-field kernel, stationary state, or ordinary
current.

Document Liouville-space and diagram-count scaling, approach validity,
near-degenerate behavior, public names, and unsupported observables before
exposure.

#### P3 validation matrix and invariants

Use small systems first so individual records remain inspectable:

| Regime | Primary assertion |
| --- | --- |
| Single-level dot | No coherence sector; analytic and legacy reduction |
| Symmetry-decoupled multilevel dot | Coherence blocks vanish; Pauli recovered |
| Well-separated double dot | Converges to population elimination as splitting grows |
| Exact/near degeneracy | Smooth state/current; no inverse-splitting clamp |
| Dark state and weakly broken dark state | Correct nullity and continuous leakage |
| Avoided crossing | Basis-covariant smooth crossover |
| Complex tunnel loop | Gauge-invariant and flux-periodic observables |
| Equilibrium | Zero lead currents and stationary conservation |
| Coherent counting | Smooth derivatives and counted/marked-current agreement |
| Coulomb blockade | Stable cotunnelling with independent fourth-order checks |

For every retained order and backend require trace preservation, Hermiticity
preservation, scale-aware stationary residual, unit trace, stationary current
conservation, equilibrium, relabeling covariance, gauge covariance, degenerate-
subspace covariance, correct nullity under disconnected states, expected
`Gamma` scaling, independently finite-differenced counting/Laplace derivatives
on small systems, covariance symmetry, and no effect from `countingleads=None`.

P3 exit gate: both coherent truncations use canonical `dm0` indexing; all four
Liouville blocks come from one topology stream; no full-coherence clamp exists;
diagnostics expose nonuniqueness and conditioning; the marked current and
counting derivatives agree; exact and near-degenerate `U=0` models agree with
NEGF in the derived regime; independent interacting checks pass where
applicable; legacy fixtures remain unchanged; the Python implementation and
existing compiled RTD compatibility projection pass; and release documentation
and installed-artifact tests pass before P4.

### P4 — Profile-driven Cython evaluator

P4 adds compiled execution only after P2 and P3 have removed the semantic
duplication. It is not a line-by-line `c_RTDnoise.pyx` port. Python remains the
single topology generator; Cython consumes a lowered, typed representation of
the same immutable records and must support both population/counting and
full-coherence assembly.

1. Profile representative P2 and P3 workloads separately for topology
   generation, direct/exchange integral evaluation, counting/Laplace
   derivatives, matrix insertion, memory, and the stationary solve. Compile
   only costs that materially affect end-to-end runtime.
2. Define a stable lowered record table using typed numeric arrays and explicit
   enum/integer tags. Its round trip to the auditable Python records must be
   checked before it is used for production assembly.
3. Port the measured scalar hot paths, including the Laplace-dependent
   direct/exchange integrals and their `_dz` evaluation when they dominate.
   Preserve the same branch-stabilized limits and derivative conventions as
   the Python evaluator.
4. Implement one compiled evaluator/assembler for the shared record stream.
   Parameterize its endpoint layout and output tensors rather than creating
   separate handwritten RTD, RTDnoise, and coherent diagram traversals.
5. Route the compiled backend only after per-record, per-transfer, per-order,
   kernel, stationary-state, current, and noise parity passes in fresh forced
   Python/Cython processes. Confirm the selected implementation with
   `qmeq.get_backend_status()`.
6. Benchmark serial and optional-OpenMP builds, memory use, and installed wheel
   and sdist behavior. Do not ship a compiled path whose measured speed-up does
   not justify its maintenance and packaging cost.
7. Keep `RTDnoise` and `pyRTDnoise` as compatibility aliases. Selecting
   counting or the compiled backend must not change the zero-field kernel,
   stationary state, ordinary current, diagnostics, or supported-observable
   contract.

P4 exit gate: the compiled path contains no independent topology rules; all
lowered-record and observable parity gates pass for real and complex amplitudes,
arbitrary-system structural cases, and the P3 coherence matrix; both forced
backends and installed artifacts report the expected implementation; optional
OpenMP and serial builds are correct; and representative benchmarks show and
document a material end-to-end benefit. If profiling finds no such benefit,
P4 exits with the compiled evaluator deliberately unshipped and the evidence
recorded instead of adding a maintenance-only backend.

## 6. Test matrix

| Model / regime | Assertion | Reference |
| --- | --- | --- |
| Spinless single resonant level | Current and noise, exactly | Analytic (existing test, tightened) |
| \(U=0\) DQD, real amplitudes | Residual exponent for \(I\), \(I^{S_z}\), \(S\), \(S^{S_z}\) | Reference solver |
| \(U=0\) DQD, generic flux | Same, plus \(2\pi\) periodicity and gauge invariance | Reference solver + metamorphic |
| \(U=0\) DQD, unequal \(T\) | Same, and `dband` independence | Reference solver |
| \(U=0\), rank-one lead coupling | Dark-mode structure; nullity diagnostics correct | Reference solver + structural |
| \(U=0\), splitting \(\lesssim\Gamma\) | Characterized as a known limitation, **not** a gate | Reference solver, informational |
| Deep Coulomb blockade, \(U\neq0\) | Cotunnelling current and Fano factor | Averin–Nazarov, independent |
| Zero bias, any \(U\) | \(S=2TG\) in this convention, \(G\) from a separate conductance derivative | Thermodynamic invariant |
| Large bias, small transmission | Fano factor \(\to1\) | Analytic limit |
| `off_diag_corrections=False` | Every pinned counting fixture | Historical regression |
| Real-amplitude fixtures across P1/P2 | Unmoved | Characterization |
| Python through P3; both backends after P4 | Parity in separate processes | Structural |

Tolerances scale with matrix norm, machine precision, summation count, and
solver conditioning, and are never widened to make a failing test pass. Where
the gate is a fitted exponent, the failure message reports the fit.

## 7. Downstream benchmark interface

The immediate consumer is the `dqd-transport` task in
`harbor-framework/terminal-bench-science` (PR 933), where a frontier model has
now passed, and RTD noise is the intended difficulty increase. Two consequences
for this plan, neither of which changes its content but both of which constrain
its outputs:

1. **Three-way agreement.** The task carries a standalone reference solver that
   does not import QmeQ. Adding RTD noise there means that solver, QmeQ, and the
   \(U=0\) reference solver must agree at every graded operating point in the \(U=0\)
   subset, and the first two must agree everywhere. Nothing should be graded
   against a value that only one implementation has ever produced.
2. **Accuracy budget.** The task grades at \(10^{-6}\) relative. P0 measured the
   old fixed-step derivative floor and P1 removed that scheme. The production
   full-kernel derivative now agrees with an independent five-point stencil far
   inside the grading budget. This remains a gate at every intended operating
   point; if a singular or poorly conditioned point violates it, change the
   operating point or grading contract rather than widening the physics gate.

Lifting the benchmark's real-amplitude restriction (section 2.2) depends on P2,
and generic-flux operating points should not be added to the task before that
phase's exit gate passes.

## 8. Risks

| Risk | Mitigation |
| --- | --- |
| `Re` placed wrongly under the counting field | Section 3.2 derivation, exact zero-field reduction pinned, reference solver gate on the noise |
| A characterization fixture is mistaken for proof | P0 assigns historical, characterization, analytic, or structural trust explicitly; only the latter two grade correctness |
| The P1 correction grows into a general rewrite | Reuse only the first-order coherence emitter in P1; diagram-record unification is a P2 deliverable |
| Complex repair gets the summed kernel right but transfer labels wrong | Derive the conjugate partner and Emary counting label first; gate per-transfer tensors and NEGF noise, not only `Wdd` |
| Complex changes move shipped numbers | P0 stores only real-amplitude fixtures; generic-flux defects are live tests, not frozen references |
| Reference solver itself wrong | Six independent self-validation checks before it grades anything; disjoint from the diagrammatics by construction |
| Reference solver's independence eroded by convenience | Section 4.5: array-level signature, no `Builder`/`approach` imports, test-only; any public promotion is a wrapper over the unchanged core, pinned by test |
| Residual-scaling fit misread | Exponent derived per observable before the sweep; truncation-order control run; fit reported in failures |
| Unification changes numbers | P2 shadow mode first, activation and deletion in separate changes |
| Numerical Laplace differentiation silently limits the benchmark | Scale-aware centered rule, independent five-point full-kernel gate, and analytic derivatives where the closed form is simple |
| Energy current blocks the noise work | Explicitly gated separately in P2; `nan` may persist |
| Scope creep into full coherence before P3 | Coherence stays eliminated through P2; P3 begins only after its hand-off gates pass |
| A direct `c_RTDnoise` port creates a third topology implementation | P4 compiles only the lowered shared record evaluator after P2/P3 unification |

## 9. Definition of done

- `RTDnoise` accepts `off_diag_corrections=True` and its noise comes from the
  same effective kernel as default `RTD`'s stationary state;
- `off_diag_corrections=False` reproduces every historical counting value;
- P0 establishes immutable historical characterization, independent analytic
  gates, structural invariants, derivative convergence, and explicit
  known-defect reproducers before production changes;
- one diagram traversal eventually serves the physical and counting-resolved
  population kernels in P2, and the duplicated `..._lpm` loops are gone;
- Laplace derivatives are analytic for the first-order blocks and bare
  coherence propagator; explicit complex direct/exchange integrals use one
  scale-aware centered rule checked by a test-only five-point reference;
- no real projection discards a physical imaginary part in the particle-current
  path, and observables are gauge-invariant and flux-periodic for arbitrary
  complex amplitudes;
- the \(U=0\) reference solver exists, is self-validated, shares no code with the
  diagrammatics, and its QmeQ bridge pins
  \(g=\sqrt{2\pi}\,t^\ast\) directly before it gates currents, spin currents,
  charge noise, and spin noise by derived residual exponent;
- the deep-blockade cotunnelling regime has an independent analytic check;
- the near-degeneracy clamp is observable in the diagnostics and never fires in
  a graded regime;
- P4 either ships one profile-justified compiled evaluator for population,
  counting, and coherent assembly with forced-backend parity, or records that
  compilation has no material benefit and deliberately leaves it unshipped;
  the separate documentation migration is not part of these gates; and
- the approach-validity documentation states the truncation, the validation
  envelope, and what the reference solver does not certify.
