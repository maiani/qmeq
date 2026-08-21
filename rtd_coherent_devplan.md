# Development plan: full-coherence RTD

Status: analysis and implementation plan only. No new RTD functionality is
implemented by this document.

## 1. Objective and safety boundary

The long-term objective is to add charge-conserving, nonsecular RTD approaches
whose stationary unknown contains populations and same-charge coherences:

\[
\rho = \{\rho_{ab}: N_a=N_b\}.
\]

The two target truncations are

\[
\mathcal L^{[1]} = L_0 + W^{(1)},
\qquad
\mathcal L^{[2]} = L_0 + W^{(1)} + W^{(2)},
\]

where \(W^{(1)}\) is second order in the tunnelling Hamiltonian
(\(O(\Gamma)\)) and \(W^{(2)}\) is fourth order in the tunnelling Hamiltonian
(\(O(\Gamma^2)\)). The implementation must generate all population/coherence
blocks from the same Liouville-space rules rather than maintaining separate
`dd`, `dn`, `nd`, and `nn` diagram implementations.

The same engine must unify RTD particle-current counting statistics with the
physical kernel. Counting is an opt-in view of the same diagram records, not a
separate approach: lead-transfer labels and Laplace-energy dependence must be
retained so the first two zero-frequency cumulants, including lead-resolved
covariance, can be derived for both the population projection and the
full-coherence solver through `O(Gamma^2)`.

The existing `RTD`/`pyRTD` approach is a compatibility boundary. Its default
effective population kernel, public inputs, stationary populations, and
particle/energy/heat currents must remain reproducible. The current approach
must not silently change to full-coherence semantics. Existing population-only
`RTDnoise`/`pyRTDnoise` results with `off_diag_corrections=False` are a second
compatibility boundary and must remain reproducible while their implementation
is absorbed into the shared RTD engine.

This project is not an opportunity to add unrelated physics, redesign all
QmeQ indexing, or change the meaning of `off_diag_corrections`. Arbitrary
higher cumulants, energy/heat-current noise, and complex-amplitude RTD energy
current remain separate projects. Full-coherence particle-current noise and
the removal of RTDnoise's separate kernel implementation are explicit goals of
this plan.

## 2. Analysis of the current implementation

### 2.1 Existing population RTD path

The pure-Python implementation is concentrated in
`qmeq/approach/base/RTD.py`; `qmeq/approach/base/c_RTD.pyx` is a separate,
performance-oriented implementation of the same behavior. `ApproachPyRTD`
currently:

1. sizes the stationary problem to `si.npauli`;
2. generates the lead-resolved first- and second-order population kernel in
   `Wdd`;
3. optionally generates real-coordinate representations of `Wdn^(1)` and
   `Wnd^(1)`;
4. constructs a diagonal `Lnn_inv` containing a clamped inverse bare energy
   splitting; and
5. adds the effective correction `Wdn^(1) Lnn_inv Wnd^(1)` to `Wdd` before
   solving.

`Lnn_inv` stores the inverse used by the elimination, not the full coherence-
sector Liouvillian. The pre-Phase-1 name was `Lnn`; that ambiguous name must not
be reintroduced in the new engine.

`KernelHandlerRTD` in `qmeq/approach/kernel_handler.py` inserts values into
the separate arrays and uses population-specific symmetry shortcuts for the
four-vertex kernel. In particular, `add_element_2nd_order` fills four
population-kernel entries from one evaluated contribution. Those shortcuts
cannot simply be generalized to arbitrary Liouville endpoints: their signs,
multiplicities, and endpoint flips need to be represented explicitly and then
revalidated.

The existing particle current is evaluated from the lead-resolved population
kernel using an anticommutator with dot number. The energy current additionally
uses `WE1` and `WE2` and has known limitations for complex tunnel amplitudes.
These formulas cannot be assumed valid after replacing the stationary vector
by the full density matrix.

### 2.2 Reusable density-matrix infrastructure

`StateIndexingDM` in `qmeq/indexing.py` already provides the required
same-charge state pairs, Hermiticity reduction, conjugation metadata,
`npauli`, `ndm0`, and the packed real dimension `ndm0r`. The general
`KernelHandler` already knows how to map complex matrix elements into QmeQ's
packed real representation, and first-order 1vN/Redfield approaches solve in
that representation.

The new RTD approaches should therefore use `StateIndexingDM` and `ndm0r`.
They should not introduce a second complex-vector ordering, public or private.
The ordered `(ket, bra, charge)` view used by the diagram generator must be the
canonical, shared description of the packed `dm0` layout rather than an
RTD-local copy of it; see section 2.4.

### 2.3 Current regression surface

The current tests cover final RTD currents, energy currents, Python/Cython
parity, many-body construction, cutoff warnings, Ozaki sizing, and a
roundoff-scale tunnel-phase regression. At the start of this planning work the
focused command

```bash
QMEQ_BACKEND=python pytest -q qmeq/tests/test_rtd_regressions.py \
    qmeq/tests/test_builder.py -k 'RTD or rtd'
```

passed 12 tests (23 deselected), and the corresponding forced-Cython command
also passed 12 tests. This is a useful baseline, but it does not freeze the
individual kernel blocks or diagram contributions and is not sufficient to
protect a structural refactor.

### 2.4 The shared packed-real representation

The coherent RTD work is not the first consumer of `ndm0r`. Lindblad, Redfield,
1vN and their electron-phonon variants already solve in it: `Approach.get_kern_size`
returns `si.ndm0r`, the shared `generate_kern` loop writes the bare Liouvillian
through `KernelHandler.set_energy`, and every coupling term reaches the kernel
through `KernelHandler.set_matrix_element`. Phase 3 does not invent `L0`; it
inherits an existing one.

That representation is currently defined only by its uses. An audit of the
working tree found:

- the offset `bbpi = ndm0 + bbp - npauli`, the `>= ndm0` inclusion test and the
  `maptype=3` conjugation sign open-coded at 11 sites in
  `qmeq/approach/kernel_handler.py` and 8 in
  `qmeq/approach/c_kernel_handler.pyx`, spread over `KernelHandler`,
  `KernelHandlerMatrixFree`, `KernelHandlerNoise`, `KernelHandlerRTD` and
  `KernelHandlerRTDnoise`;
- no divergence between the two backends. `set_matrix_element` is
  element-for-element identical in the `.py` and `.pyx` copies, so this is
  duplication, not drift, and no shipped result is currently suspect;
- no direct test of the mapping. It is exercised only end-to-end through physics
  fixtures, so an index or conjugation error surfaces as a wrong current rather
  than as a failing adapter test; and
- integer `maptype` arguments at Python call sites where the Cython side has
  named accessors, for example `get_ind_dm0(a, ap, acharge, maptype=3)` against
  `get_ind_dm0_conj`;
- an undocumented sign divergence between the assembled kernel and the
  matrix-free path. `KernelHandlerMatrixFree` writes its imaginary rows with
  the opposite sign, measured as exactly
  `diag(+1 on [0, ndm0), -1 on [ndm0, ndm0r))`. A per-row sign leaves the
  stationary null space untouched, which is why `mfreeq=True` and
  `mfreeq=False` agree on solutions and why nothing caught it; it does mean
  `dphi0_dt` is not the packed time derivative, so any future time propagation
  or residual diagnostic must not assume it is; and
- a trace rule that is not the obvious one. Under `indexing='ssq'` a stored
  population index stands for a whole symmetry multiplet, so the trace is the
  multiplicity-weighted population sum rather than `sum(phi0[0:npauli])`.
  `generate_norm_vec` already accumulates those multiplicities correctly. This
  matters for the coherent solver's trace constraint and for any independent
  normalization check written against the layout.

The consequence for this project is specific. There is no written specification
of the layout to write diagram assembly against, and an RTD-private view would
become a sixth parallel copy of the same arithmetic carrying harder semantics:
arbitrary same-charge Liouville endpoints instead of the fixed population and
first-order patterns the existing handlers cover. Reconcile the representation
before expanding onto it, and do it while the QmeQ 1.1 corpus that pins every
current consumer is freshly established.

### 2.5 Existing RTD counting path

`ApproachPyRTDnoise` subclasses the Python RTD approach but duplicates the
first- and second-order population diagram loops in counting-resolved forms.
It stores transfer-resolved `Lpm_first` and `Lpm_second` tensors plus finite-
difference Laplace derivatives, solves auxiliary order-separated stationary
problems, and evaluates non-Markovian cumulants with the shared projected-
pseudoinverse machinery. There is no compiled RTDnoise evaluator.

The separation is scientifically visible: RTDnoise rejects
`off_diag_corrections=True`, so its stationary kernel differs from default RTD.
Adding the zero-field correction only after constructing counting kernels would
not fix this. The effective coherence contribution must carry counting-field
and Laplace dependence before differentiation. The rewrite must therefore
preserve vertex transfer labels and propagator-energy dependence in the generic
records instead of trying to reconstruct them from an assembled zero-field
matrix.

## 3. Theoretical decisions required before code changes

The primary reference is Leijnse and Wegewijs, *Phys. Rev. B* **78**, 235424
(2008), [doi:10.1103/PhysRevB.78.235424](https://doi.org/10.1103/PhysRevB.78.235424),
[arXiv:0807.4027](https://arxiv.org/abs/0807.4027). It explicitly treats the
complete molecular density matrix through fourth order in the tunnelling
Hamiltonian. The implementation must not proceed from topology names alone;
the following convention sheet must first be derived and reviewed against the
paper and the existing QmeQ equations/code.

### 3.1 Convention sheet

Create a developer note, kept with the implementation, that fixes:

- the definition and sign of `L0`, the stationary equation, and QmeQ's packed
  real representation;
- kernel index order `W[final, initial]`;
- chronological versus diagrammatic vertex order;
- upper/lower Keldysh-branch sign conventions;
- electron/hole orientation and reservoir contraction direction;
- many-body tunnelling-amplitude order and conjugation;
- fermion-parity signs, including exchange signs;
- direct and exchange irreducible four-vertex topologies;
- propagator denominators, infinitesimal/Laplace prescription, and energy
  arguments supplied to the existing `D` and `X` integrals;
- lead attribution for a kernel and for a current vertex; and
- factors arising from conjugate diagrams and the current population-only
  symmetry shortcuts.

Every generator test should refer to a named rule from this sheet. This is more
auditable than treating the legacy nested loops as the specification.

### 3.2 Perturbative bookkeeping

Two different statements must remain distinct:

1. the kernel is truncated at `O(Gamma)` or `O(Gamma^2)`; and
2. the stationary null vector of that truncated kernel is solved exactly.

The latter generally contains an implicit resummation of higher powers of
`Gamma`. That may be a useful and standard solver definition, particularly in
a quasidegenerate subspace, but it must be documented rather than described as
a globally order-by-order density matrix. Tests should include coupling-scale
studies that distinguish retained orders from accidental higher-order
behavior.

Before phase 5, decide and document how the target equations map onto the
paper's perturbative treatment of secular/quasidegenerate and nonsecular
components. In particular, establish whether solving one full
`L0 + W^(1) + W^(2)` matrix for all same-charge coherences is the intended
approximation for well-separated states, or whether an order-consistent
projection/elimination is required outside the slow subspace. Do not resolve
this policy implicitly in code.

### 3.3 Degeneracies and uniqueness

The new solver removes the explicit bare `1/(Ea-Eb)` elimination singularity,
but a coupled Liouvillian can still have multiple or ill-conditioned stationary
states. Define diagnostics for:

- nullity greater than one;
- the smallest nonzero singular value or an equivalent condition estimate;
- residual norm after trace normalization;
- trace and Hermiticity errors; and
- minimum density-matrix eigenvalue as a diagnostic, not an automatic repair.

No ad hoc energy-splitting clamp belongs in the full-coherence kernel.

### 3.4 Current definition

Particle current should be generated by a lead-resolved current kernel with a
marked current vertex and evaluated as

\[
I_r = \langle\!\langle \mathbb 1 | W^{I_r} |\rho_{\rm stat}\rangle\!\rangle.
\]

The sign and normalization must reproduce the legacy anticommutator result in
the population projection. Energy and heat currents require a separate
derivation and gate; they must not be inferred by padding the legacy `WE1` and
`WE2` matrices with zero coherence blocks.

### 3.5 Counting fields and non-Markovian derivatives

Fix the counting convention on the same vertex records used for the kernel.
For every counted lead, derive the zero-field objects required by the first two
cumulants:

\[
\partial_{\chi_r}W,\quad
\partial_{\chi_r}\partial_{\chi_s}W,\quad
\partial_z W,\quad
\partial_z\partial_{\chi_r}W.
\]

For the legacy eliminated-coherence projection, apply these derivatives to the
complete effective kernel

\[
W_{\mathrm{eff}}(\boldsymbol\chi,z)=W_{dd}(\boldsymbol\chi,z)
\quad+W_{dn}(\boldsymbol\chi,z)G_{nn}(\boldsymbol\chi,z)
W_{nd}(\boldsymbol\chi,z),
\]

using an explicitly derived perturbative expansion of `Gnn`; do not dress only
`Wdd`. For the full-coherence solver, differentiate the full Liouvillian
directly, without Schur elimination. Decide analytically which derivatives of
the free and interacting coherence propagators contribute at each retained
order before implementing them. Finite differences may be an independent test
oracle, but not the sole production definition.

## 4. Proposed architecture

Use one canonical Python diagram description and generator first. Do not begin
with parallel hand-written Python and Cython generators.

A suitable internal layout is:

```text
qmeq/approach/base/rtd/
    __init__.py
    conventions.py       # enums and sign/orientation definitions
    liouville.py         # dm0 adapter and Liouville-state transitions
    diagram.py           # immutable two-/four-vertex diagram records
    generator.py         # topology enumeration and validity rules
    integrals.py         # adapter to the existing scalar RTD integrals
    assembler.py         # lead-resolved packed-real kernel assembly
    current.py           # marked-vertex/current-kernel generation
    counting.py          # chi/z derivatives and zero-frequency cumulants
```

The exact split may be adjusted, but the dependency direction should be:

```text
StateIndexingDM -> Liouville adapter -> diagram generator
                                      -> scalar evaluator -> assembler -> solver
```

The generator must not write directly into NumPy matrices. It should emit an
immutable record containing at least:

- initial, intermediate, and final Liouville states;
- perturbative order and direct/exchange topology;
- ordered vertex records (branch, electron/hole orientation, lead, and
  many-body transition);
- reservoir contraction pairing;
- tunnelling-amplitude factors and fermionic sign;
- propagator energy differences and integral kind; and
- the lead attribution used by the kernel/current observable;
- signed particle transfer at every reservoir vertex; and
- explicit Laplace-energy dependence needed for analytic `z` derivatives.

The evaluator maps an already validated record to a scalar. The assembler maps
its endpoints to `dm0` packed-real coordinates. Whether an entry is `dd`,
`dn`, `nd`, or `nn` is then a derived property used only for diagnostics and
block-level tests.

For performance, stable diagram records may later be lowered to structure-of-
arrays buffers consumed by Cython. Cython should accelerate scalar evaluation
and accumulation from the same generated records; it should not independently
reimplement the topology rules unless profiling proves that unavoidable.

## 5. Staged implementation plan

Each phase is a reviewable stopping point. A phase starts only after the prior
phase's exit gate passes.

### Phase 0 — Freeze legacy behavior and provenance

No RTD production code changes.

1. Add characterization tests for small, deterministic models covering:
   - a single-level Anderson dot;
   - a nondegenerate double dot with real amplitudes;
   - a double-orbital model with complex amplitudes;
   - equal and unequal lead temperatures;
   - `off_diag_corrections` both false and true;
   - `Builder` and `BuilderManyBody`; and
   - every indexing mode supported by QmeQ 1.1 RTD. QmeQ 1.1 supports only
     `charge`; also characterize and preserve its explicit fallback from a
     requested spin symmetry to `charge` rather than claiming unsupported
     symmetry-reduced coverage.
2. Capture, separately, lead-resolved `Wdd^(1)`, `Wdd^(2)`, total `Wdd`,
   `Wdn^(1)`, `Wnd^(1)`, the inverse coherence propagator currently called
   `Lnn_inv` (stored under the provenance key `inverse_Lnn`), `WE1`, `WE2`,
   `phi0`, and all currents.
3. Add invariant tests that do not depend on golden numbers: column trace
   preservation, stationary residual, normalization, lead-current
   conservation, equilibrium zero current, and structural zeros.
4. Store reference values outside Python source. Use a JSON manifest for
   provenance, scenario-to-array mappings, shapes, and dtypes, with a
   compressed `.npz` archive for lossless real/complex multidimensional
   arrays. Every fixture must have metadata with:
   generating script, QmeQ revision, backend, Python/NumPy/SciPy versions,
   model parameters and units, array ordering, tolerance rationale, and whether
   the value is analytic, independently calculated, or merely a legacy
   characterization value.
5. Never regenerate expected data as part of a test. Generation is an explicit
   maintainer command, and a fixture change requires a physics explanation.
6. Every new RTD characterization fixture in Phase 0 must be generated by the
   pristine QmeQ 1.1 source at commit
   `96cc51076458b11f7db81a5d7d8df04c30bf8384`. Current QmeQ must never replace
   those values. Where an audited post-1.1 bug fix intentionally changed a
   result, retain the 1.1 value and document a fixed, field-specific comparison
   envelope and its reason in the manifest.

Exit gate: the new characterization suite passes for both forced backends in
separate processes, and at least one reviewer can reproduce the fixture set
from its provenance record.

Completed baseline status (2026-08-20): the provenance fixtures are generated
from pristine commit `96cc51076458b11f7db81a5d7d8df04c30bf8384`, which
reports QmeQ 1.1 and immediately precedes the modernization branch's first
first-parent commit. It covers Pauli, Lindblad, Redfield, 1vN, 2vN, and RTD,
plus the Pauli, Lindblad, Redfield, and 1vN electron-phonon variants. The RTD
matrix covers a single-level equilibrium model, a nondegenerate coherent
double dot with `off_diag_corrections` both enabled and disabled, complex
tunnelling amplitudes, unequal lead temperatures, `BuilderManyBody`, and the
documented spin-symmetry-to-`charge` fallback. Every RTD fixture separately
records first-order, second-order, and coherence-elimination contributions,
all population/coherence blocks, the stationary state, and all currents. Its
JSON manifest and compressed array archive live under
`qmeq/tests/data/qmeq_11/` and are regenerated only by
`scripts/reference_data/generate_qmeq_11_references.py` from the pristine
historical tree.

The suite checks provenance and metadata, block decomposition, column trace
preservation, stationary residual, normalization, particle- and energy-current
conservation, equilibrium zero current, heat-current identity, structural
zeros, and isolation of the off-diagonal correction. Two clean generations
from the detached 1.1 tree produced array-identical archives. The audited
post-1.1 complex-integral branch correction has field-specific comparison
envelopes in the manifest; the fixture values themselves remain from 1.1.
The selected compiled electron-phonon Lindblad comparison is the repository's
only strict expected failure and documents its existing P1 backend-parity gap
inline in the test; its historical pure-Python kernel remains covered.

Phase 0 is closed. No production solver or coherent-RTD functionality was
changed while establishing this baseline.

### Infrastructure checkpoint before Phase 1

Do not begin the Liouville adapter immediately. Complete these maintenance
items first so the coherent-RTD work does not grow on top of duplicate or
source-checkout-only regression infrastructure:

1. Migrate or retire the golden dictionaries in `qmeq/tests/data_builder.py`,
   `qmeq/tests/data_builder_elph.py`, and `qmeq/tests/data_counting.py`. Preserve
   the builder and electron-phonon values exactly and label them `legacy`:
   their generating revision and environment were never recorded, so they
   must not be relabeled as QmeQ 1.1. Counting statistics do not exist in QmeQ
   1.1, so retain the distinct Simon Wozny source provenance documented by its
   generator.
2. Reuse the external JSON/NPZ schema rather than creating another format.
   Update the builder and counting tests to load with `allow_pickle=False`,
   validate provenance/shape/dtype metadata, and keep all regeneration explicit
   and maintainer-only.
3. Add an installed-artifact CI check that builds both wheel and sdist, confirms
   the reference files are present, installs each artifact outside the source
   tree, asserts the requested backend, and runs the focused QmeQ 1.1 reference
   suite. The manual wheel/sdist inspection completed in Phase 0 is evidence,
   not a substitute for the automated gate.

Exit gate: the three targeted Python data modules are gone; one external
provenance schema covers the QmeQ 1.1 corpus, the legacy builder snapshots,
and the separately sourced counting-statistics data; and source-tree,
installed-wheel, and installed-sdist focused checks pass. The compiled
electron-phonon Lindblad parity failure may remain as the repository's
single strict, documented `xfail`: it is isolated from RTD and must neither be
broadened nor used to mask a new indexing regression.

Completed checkpoint status (2026-08-20): the builder and electron-phonon
literals were migrated losslessly as 100 arrays in the `legacy` bundle without
running a solver. The 15 counting-statistics arrays were migrated separately
with their recorded Simon source commit. Both use the same validated bundle
loader and schema as the 182-array QmeQ 1.1 corpus, so future snapshots can be
added as another manifest/archive directory. The old Python data modules and
their source-writing helpers were removed. Maintainer-only generators now live
under `scripts/reference_data/`, outside pytest's package, while reusable model
construction remains in a neutrally named test support module. A focused CI
job builds and checks wheel and sdist, installs each outside the source tree,
asserts the forced backend, and runs the external-reference suites. This
checkpoint is closed; Phase 1 may begin in a separate change.

The following TODO items are not prerequisites for Phase 1 and should remain
separate changes:

- structured warnings and stationary-state diagnostics must be designed and
  implemented before Phase 3 introduces a new solver;
- the consolidated approach-validity and failure-mode documentation should be
  completed with the convention sheet before Phase 2 evaluates diagrams;
- the legacy RTD near-degeneracy warning requires a defensible physics
  threshold rather than an infrastructure-only change; and
- complex-amplitude RTD energy/heat currents, unequal-temperature convergence
  tooling, and `BuilderManyBodyElPh` are independent shipped-feature gaps.
  RTDnoise reconciliation is not a Phase 1 prerequisite, but is part of Phases
  2 through 8 because the generic records must carry its information from their
  introduction.

### Phase 1 — Reconcile and specify the packed-real representation

Still no diagram refactor. Phase 1 has two parts and two gates: the second is a
refactor of shipped code and must not be merged with the first.

#### Phase 1a — Specify and test the Liouville adapter

No production code changes. The existing handlers are the oracle.

1. Implement an immutable `LiouvilleState(ket, bra, charge)` view over
   `StateIndexingDM`, together with a written specification of the packed `dm0`
   layout: real/imaginary partition, the `ndm0`/`npauli` offset, the inclusion
   predicate, the conjugation sign, and the sector ordering. Give the
   `maptype` integers named accessors on the Python side to match the Cython
   ones.
2. Test round trips between every included Liouville state and QmeQ's packed
   `dm0` real coordinates, including conjugate pairs and excluded/symmetry-
   mapped elements. Include empty and removed charge sectors, sectors with no
   coherences, and the smallest nonempty sectors so zero-sized adapters cannot
   be hidden by the normal reference models. Cover the `sz` and `ssq` indexing
   modes that Lindblad, Redfield and 1vN accept even though coherent RTD will
   only use `charge`.
3. Define trace and Hermitian-conjugation operations in both representations.
4. Add randomized small-sector tests showing that packed assembly agrees with
   explicit complex-matrix action, and differential tests asserting that the
   adapter and `KernelHandler.set_matrix_element` produce identical `ndm0r`
   matrices from the same complex kernel on the same `si`. This uses the
   already-pinned first-order approaches as a free oracle and introduces no new
   physics claim.

Exit gate: indexing, conjugation, trace and complex-to-real kernel conversion
are fully tested without importing the RTD generator, and the layout has a
specification a reviewer can check diagrams against.

#### Phase 1b — Migrate existing handlers onto the specification

A behavior-preserving refactor of shipped code, reviewable on its own.

1. Route the duplicated offset, inclusion and conjugation arithmetic in
   `kernel_handler.py` and `c_kernel_handler.pyx` through the single definition
   from phase 1a, so the two backends agree by construction rather than by
   inspection.
2. Change no sign, prescription or kernel value. Normalize only conventions
   that no observable, published equation or stored fixture can see: the
   integer `maptype` arguments, the two different exclusion idioms
   (`== -1` against `>= ndm0`), and the unguarded `-1` sentinel. Leave the
   insertion convention of rule L6 alone. Lindblad's `1j*fct` against 1vN's
   `fct` is not a physics difference; it is compensation for the `-i` that L6
   builds into `set_matrix_element`, and that `-i` is load-bearing because it
   is what lets the bare Liouvillian and the tunnelling kernel share one
   insertion path (rule L7). Removing it would force an explicit `-1j` at
   every 1vN, Redfield and RTD call site and buy nothing.
3. Leave `StateIndexingDMc` and the 2vN kernel size out of scope.
4. Guard the change with the QmeQ 1.1 corpus, the legacy builder snapshots and
   the counting-statistics bundle, run for both forced backends in separate
   processes. Every affected approach — Pauli, Lindblad, Redfield, 1vN, legacy
   RTD, RTDnoise and the electron-phonon variants — must reproduce its fixtures
   bit-for-bit where the arithmetic is unchanged.

Exit gate: one definition of the packed-real layout, no fixture moves, both
forced backend suites green, and the documented electron-phonon Lindblad
`xfail` neither broadened nor newly masking anything. If this gate cannot be
met without touching a fixture, stop and treat the discrepancy as a finding
before continuing to phase 2.

Completed status (2026-08-20): both parts are closed.

Phase 1a added `qmeq/approach/dm_layout.py`, which states the layout as nine
numbered rules with an immutable `LiouvilleState` view and a reference
`DensityMatrixLayout`. `qmeq/tests/test_dm_layout.py` pins each rule, every test
naming the rule it covers, across `charge`/`sz`/`ssq`, a sector with no
coherences, and layouts with empty leading and trailing sectors. Two conventions
had to be recovered by measurement rather than read off the source: the packed
kernel implements `rho -> -i (W rho)` (agreement `3.9e-16`), which makes
Lindblad's `1j*fct` a convention artifact and not a physics difference; and
`set_energy` is exactly `set_matrix_element` with a real value, so `L0` is
inherited rather than built by a new approach.

Two claims in the plan's own audit were wrong and are corrected above. The trace
is multiplicity-weighted, not a plain population sum, which only shows under
`ssq`; `generate_norm_vec` was already right. And `KernelHandlerMatrixFree`
negates its imaginary rows relative to the assembled kernel, which is harmless
for a null-space solve but means `dphi0_dt` is not the packed time derivative.

Phase 1b routed the duplicated arithmetic in both handlers through one named
`imag_offset`, replaced the partner test with the equivalent `i >= npauli`,
named the `EXCLUDED` sentinel in both backends, made insertion at an uncarried
endpoint a no-op, and gave `maptype` 2 and 3 named accessors. Unsupported
`maptype` values now raise instead of returning `None`.

Making those two silent paths loud exposed two live defects that the green
suite had been hiding, both instances of one problem: `StateIndexingDM`,
`StateIndexingDMc` and `StateIndexingPauli` are siblings with overlapping but
unequal interfaces, no shared base defining the `dm0` contract, and names that
suggest inheritance that does not exist. `Builder.get_phi0` was asking
`StateIndexingDMc` for a conjugation map it does not have and discarding the
`None`; and the electron-phonon approaches hold `si` and `si_elph` as
*different* classes, so a helper added only to `StateIndexingDM` broke all six
electron-phonon reference tests. Both are fixed, with regression tests. Treat
this as a standing hazard for Phase 2 onward: at any call site in
`qmeq/approach/elph/`, the local name `si` may be a `StateIndexingDMc`.

Verification: ruff clean; pure-Python 489 passed / 18 skipped and forced-Cython
488 passed / 18 skipped / 1 xfailed, in separate processes, with the
electron-phonon Lindblad parity `xfail` unchanged and no reference fixture
moved. Hot-path cost measured at 0.359 s against a 0.350 s baseline for twenty
1vN solves, within noise.

The canonical Phase 1 conventions are recorded in `dm_layout.py` and pinned by
`test_dm_layout.py`. The ignored `new_docs/` tree is intentional: it is the
working prototype for the documentation system intended eventually to replace
Sphinx. It is not yet a Phase 1 deliverable because the user guide, Python and
Cython API coverage, examples, and release build still have to be migrated.
Until that separate migration is complete, the existing Sphinx documentation
remains the shipped, warning-clean documentation gate.

Readability pass (2026-08-21): a follow-up change, separate from the phase 1
gates above and carrying no behaviour change, addressed the naming that made
this area hard to audit.

The pure-Python electron-phonon Pauli and 1vN approaches no longer bind the name
`si` to `self.si_elph`. Those modules hold two indexing objects of *different*
classes, and in `elph/pauli.py` the shared local name meant `StateIndexingDMc`
in one method and `StateIndexingDM` in the adjacent one. Any Phase 2 code
reached from `qmeq/approach/elph/` must still assume either class may appear;
the rename makes which one visible at the call site instead of invisible.

The "no index" sentinel is named `NO_INDEX` at 61 sites across both backends,
and `QMEQ_STRICT_INDEX=1` turns a missed guard into an `IndexError` rather than
a silent skip. That converts the one-off probe used above into a standing check:
the full suite passes under it, so no shipped path is missing an `is_included`
guard. Use it when adding diagram assembly, which will insert at arbitrary
same-charge endpoints and therefore has more ways to get this wrong than the
fixed patterns the current approaches use.

RTD's eight destination arrays are selected by the `RtdMatrix` enum instead of a
bare trailing integer, at 61 call sites. The compiled mirror `RtdMatrixC` is
declared `cpdef` so the two copies are compared in the suite rather than trusted.
Two asymmetries surfaced and are documented rather than changed: the `Lnn_inv`
selector is valid only in the compiled `add_matrix_element`, and the array is
two-dimensional in pure Python against a bare diagonal in Cython.

`Lnn` is renamed `Lnn_inv` throughout, resolving the ambiguity section 2.1 of
this document warned about. The provenance-locked fixture key `inverse_Lnn` is
unchanged.

Verification: ruff clean, Sphinx `-W --keep-going` succeeded, and three separate
full-suite legs passed — pure Python 493, forced Cython 493 plus the unchanged
electron-phonon Lindblad `xfail`, and pure Python under `QMEQ_STRICT_INDEX=1`
493. No reference fixture moved.

### Phase 2 — Generic two-vertex engine in shadow mode

1. Write the convention sheet and enumerate all irreducible two-vertex
   diagrams for arbitrary initial and final Liouville states.
2. Initially project generator output onto `dd` and compare it element by
   element and lead by lead with legacy `Wdd^(1)`.
3. Project onto `dn` and `nd` and reproduce the legacy first-order elimination
   ingredients before using them in a new solver.
4. Generate `nn` and verify trace/Hermiticity preservation of the complete
   first-order kernel.
5. Attach signed lead-transfer and analytic Laplace-derivative metadata to the
   same records. In population projection, reproduce `Lpm_first` and
   `Lpm_first_dot` from RTDnoise without a second traversal.
6. Keep the legacy implementation active. Run the generic engine only in tests
   or an internal comparison mode until all block tests pass.

Exit gate: exact structural equality and justified numerical agreement with
all available legacy first-order blocks and sequential RTDnoise derivatives,
plus independent conservation tests on the complete kernel.

### Phase 3 — First-order full-coherence solver

1. Add an internal approach using `ndm0r` and the normal QmeQ trace constraint.
2. Assemble `L0 + W^(1)` from the generic engine; expose separate `L0`,
   lead-resolved `W1`, and total-kernel diagnostics for testing.
3. Do not apply `off_diag_corrections` in this approach. Reject that option
   clearly if supplied rather than silently ignoring or double counting it.
4. Add the lead-resolved first-order particle-current kernel and opt-in
   zero-frequency current/noise covariance from derivatives of that same
   kernel. Its population-only reduction must agree with both Pauli counting
   and `current_noise_first`.
5. Keep the public approach name provisional until the validation gate. A
   clear eventual naming pair would be `RTD1Coherent`/`pyRTD1Coherent`, but it
   must be checked against QmeQ's selected-backend naming convention.

Validation matrix:

- Pauli agreement when the generated off-diagonal blocks vanish by construction;
- comparison with both 1vN and Redfield only in explicitly derived matching
  conventions/limits; disagreement elsewhere is not automatically a bug;
- exact single-level reduction;
- trace and Hermiticity preservation for arbitrary Hermitian trial states;
- equilibrium zero current and stationary charge conservation;
- invariance under rephasing of dot many-body states and consistent lead/orbital
  tunnel-amplitude gauge transformations;
- basis covariance inside an exactly degenerate same-charge subspace;
- regular behavior at exact and near degeneracy;
- current from the first counting derivative equals the explicitly marked-
  vertex current, with covariance symmetric under lead exchange; and
- Python/Cython-selected API parity once a compiled evaluator exists.

Exit gate: all invariants pass, the comparator regimes are documented, the
stationary solution diagnostics are exposed, and no legacy RTD fixture moves.

### Phase 4 — Generic four-vertex engine, population projection only

1. Derive direct and exchange topology templates from the reviewed convention
   sheet.
2. Expand every legacy symmetry shortcut into explicit generated diagrams.
3. Reuse the existing `integralD`/`integralX` numerical functions through a
   narrow adapter; do not change special-function behavior during this phase.
4. Generate only the `dd` projection initially and compare:
   - individual diagram records on hand-enumerated toy sectors;
   - per-topology and per-lead partial sums;
   - every `Wdd^(2)` matrix element;
   - every population-projected `Lpm_second` element and required Laplace
     derivative from RTDnoise;
   - the full legacy effective kernel after applying the unchanged legacy
     first-order elimination; and
   - stationary populations and currents from the phase-0 fixtures.
5. Test traversal-order determinism and repeated-run stability. Where OpenMP
   reduction order prevents bitwise equality, use a separately justified
   tolerance and retain exact structural-count tests.

Exit gate: the shadow engine reproduces legacy `Wdd^(2)`, `Lpm_second`, and
their required Laplace derivatives across the reference matrix without
changing the production path. This is the main authorization point for
replacing reusable legacy routines.

### Phase 5 — Replace reusable population routines without behavior change

Only after phase 4, route legacy `RTD` and RTDnoise through the generic engine
incrementally:

1. diagram topology/enumeration;
2. scalar-integral argument construction;
3. lead-resolved assembly;
4. current-kernel plumbing where it is provably equivalent; and
5. counting-field and Laplace derivatives, first with
   `off_diag_corrections=False`, without changing any existing RTDnoise result.

Keep a temporary test-only legacy oracle until both forced-backend full suites
and slow RTD examples pass. Remove duplicate routines only after the new path
has survived at least one clean comparison cycle; do not delete the oracle in
the same change that first activates the replacement.

Exit gate: phase-0 and Simon counting data are unchanged within fixed
tolerances, all backend tests pass, and both old nested implementations can be
removed in separately reviewable cleanups. Public approach names remain
unchanged at this gate.

### Phase 6 — Complete second-order coherence blocks

1. Permit arbitrary same-charge initial and final Liouville states in the
   four-vertex templates.
2. Obtain `dn`, `nd`, and `nn` as projections of the same generated records.
3. Retain counting transfers and analytic Laplace derivatives for every block,
   including the ingredients needed to differentiate the legacy
   eliminated-coherence effective kernel consistently.
4. Add block-resolved diagnostics and tests for Hermiticity and trace
   preservation before solving a stationary problem.
5. Hand-enumerate minimal sectors that exercise every branch/orientation and
   both direct/exchange topologies. Verify signs and conjugate partners without
   relying on the generator itself as the oracle.
6. Compare selected entries with an independent, deliberately slow reference
   enumerator or symbolic notebook whose conventions and revision are stored
   with the test provenance.

Exit gate: the complete `W^(2)` satisfies the formal invariants, all topology
coverage counters are nonzero where expected, and independent spot checks pass.

### Phase 7 — Second-order full-coherence solver

1. Assemble and solve `L0 + W^(1) + W^(2)` in `ndm0r` with one trace
   constraint and the diagnostics defined in section 3.3.
2. Keep `W1`, `W2`, and their lead-resolved forms separately observable for
   debugging; do not retain only the final sum.
3. Add coupling-scale tests to characterize the exact solution of the
   truncated Liouvillian and expose unintended lower-order contamination.
4. Add the particle-current kernel consistently through second order.
5. Add opt-in non-Markovian current/noise covariance through second order from
   the full Liouvillian derivatives. In parallel, validate counting-resolved
   Schur elimination against default legacy RTD so its noise uses the same
   corrected stationary kernel rather than RTDnoise's old population-only one.
6. Gate energy and heat currents behind their completed derivation and tests.
   Until then, raise a clear `NotImplementedError` for those observables in the
   new approach rather than returning population-only approximations.

Exit gate: all formal invariants and physical validation cases pass, the
solver remains regular at exact degeneracy when a unique stationary state
exists, counted current agrees with the explicit current kernel, and both the
legacy RTD and population-only RTDnoise compatibility fixtures remain
unchanged in their stated modes.

### Phase 8 — Backend, performance, documentation, and public API

1. Profile before compiling. Record diagram counts, generation time, integral
   time, assembly time, memory, and solve time for representative systems.
2. Lower stable diagram tables to typed arrays and compile only measured hot
   paths. Maintain one topology generator if feasible.
3. Validate pure-Python and compiled implementations in separate processes and
   assert `qmeq.get_backend_status()['active']` for the selected implementation.
4. Make `countingleads` opt-in on `RTD`/`pyRTD` and on the new coherent RTD
   approaches. Keep `RTDnoise`/`pyRTDnoise` as compatibility aliases rather
   than separate kernel implementations; preserve the documented sequential
   and consistently fourth-order-truncated diagnostic outputs.
5. Add builder validation, imports, API docs, approach-validity documentation,
   examples, and an `[Unreleased]` changelog entry only when the feature is
   ready to expose.
6. Document scaling: the Liouville dimension grows as the sum of squared
   charge-sector sizes, and a generic four-vertex enumeration can grow much
   faster. Add explicit resource diagnostics or limits if measurements justify
   them.

The broader Sphinx-to-`new_docs/` migration is tracked separately. Phase 8 must
write new RTD material so it can be carried into that successor, but completing
the documentation-platform replacement is not a condition for landing the RTD
rewrite. Sphinx remains the release gate until the successor has equivalent
coverage and a replacement clean-build check.

Exit gate: both forced backend suites, the slow Python examples, the explicitly
selected `notebook` group in an environment that permits Jupyter local sockets,
warning-clean docs, and relevant artifact-installed tests pass. Public aliases
are added without changing uncounted `RTD`/`pyRTD` behavior, and selecting
counting cannot change the zero-field stationary kernel or ordinary current.

## 6. Physical validation suite

Use small systems first so individual diagrams remain inspectable.

| Model/regime | Primary assertion | Reference type |
| --- | --- | --- |
| Single-level Anderson dot | No coherence sector; new and legacy population results agree | Analytic/legacy |
| Single-level counting | Current and noise agree with the exact scattering limit | Analytic integral |
| Symmetry-decoupled multilevel dot | Coherence blocks vanish; Pauli limit recovered | Analytic structural |
| Well-separated double dot | Full solver approaches the controlled population-eliminated result as splitting grows | Asymptotic sweep |
| Nearly degenerate orbitals | Smooth stationary state/current through `Delta E = 0`; no inverse-splitting clamp | Invariant and convergence |
| Exact dark state | Expected trapped subspace/current suppression and correct nullity diagnostics | Analytic first-order model |
| Slightly broken dark state | Continuous leakage-current crossover | High-precision numerical reference |
| Avoided crossing | Basis-covariant smooth crossover | Parameter sweep |
| Complex tunnel loop | Gauge-invariant observables under allowed rephasings | Metamorphic test |
| Equilibrium, including degeneracy | Zero particle current and stationary charge conservation | Thermodynamic invariant |
| Coherent double-dot counting | Counting derivatives remain smooth through degeneracy and counted current equals explicit current | Independent finite-difference/metamorphic |
| Coulomb blockade | Sequential suppression and stable cotunnelling contribution | Independent fourth-order spot checks |

For the splitting crossover, compare dimensionless ratios rather than a fixed
energy threshold. Sweep `Delta E/Gamma_eff` over several decades and define
`Gamma_eff` explicitly from the same tunnelling-rate convention used by QmeQ.
The test should demonstrate convergence for large splitting and regularity,
not forced agreement, for `Delta E` comparable to or below the broadening.

## 7. Mandatory invariant and metamorphic tests

For each perturbative order and each backend:

- `trace(W @ rho) == 0` for a basis of Hermitian trial states;
- the assembled action maps Hermitian matrices to Hermitian matrices;
- the stationary solution has unit trace within a scale-aware tolerance;
- the stationary residual is small relative to the kernel norm;
- `sum_r I_r == 0` in stationarity;
- each lead current vanishes in equilibrium within a justified tolerance;
- relabeling many-body states only permutes the result;
- rephasing basis states transforms `rho` covariantly and leaves observables
  invariant;
- unitary rotations within exactly degenerate retained subspaces give covariant
  density matrices and invariant observables;
- removing a zero-coupled state does not alter the connected-sector result,
  while correctly changing nullity diagnostics;
- scaling all tunnel amplitudes verifies the expected `Gamma` and `Gamma^2`
  kernel coefficients before the stationary solve;
- first and mixed counting/Laplace derivatives agree with high-precision finite
  differences of an independently assembled `W(chi,z)` on small systems;
- lead covariance is symmetric, aggregate counting equals the sum of its
  lead-resolved entries, and counted current equals the ordinary current; and
- setting `countingleads=None` does not alter the zero-field kernel, stationary
  state, or any uncounted observable.

Tolerances must scale with matrix norm, machine precision, summation count, and
solver conditioning. They must not be widened until a failing test passes.

## 8. Reference-data policy

Reference data has three trust levels, recorded per quantity:

1. **analytic** — closed-form or exact small-model result;
2. **independent** — produced by a separate slow enumerator, symbolic
   calculation, or independently implemented published equation; and
3. **characterization** — generated by the current QmeQ implementation solely
   to prevent accidental legacy drift.

Characterization data is not evidence that the old value is physically
correct. New coherence-block correctness must not be established by data
generated from the same generator being tested. Each fixture should state its
trust level, provenance, conventions, units, and regeneration procedure.

## 9. Review and change-management strategy

Prefer small changes with one scientific claim each:

1. tests and reference generator only;
2. Liouville adapter and packed-real specification only;
3. behavior-preserving migration of the existing kernel handlers onto it;
4. two-vertex shadow generator;
5. first-order solver;
6. four-vertex `dd` shadow generator;
7. legacy routing with no numerical change;
8. second-order coherence blocks;
9. second-order solver, particle current, and full-coherence counting;
10. legacy RTD/RTDnoise shared routing and compatibility aliases; and
11. compiled acceleration and public documentation.

Do not combine reference-data updates with a kernel behavior change unless the
old reference is independently demonstrated to be wrong and the change
contains that evidence. Do not update generated C files; `.pyx`/`.pxd` remain
canonical compiled sources.

## 10. Principal risks and mitigations

| Risk | Mitigation |
| --- | --- |
| Hidden sign/conjugation error | Reviewed convention sheet, hand-enumerated diagrams, gauge and Hermiticity tests |
| Legacy drift during extraction | Phase-0 block fixtures and shadow-mode comparison before routing production code |
| Double counting conjugate/symmetry-related diagrams | Explicit diagram identity and multiplicity; topology-count tests |
| Uncontrolled perturbative interpretation | Separate kernel-order tests from exact stationary solve; coupling-scale studies |
| Singular/multiple stationary states | Nullity, conditioning, and residual diagnostics; no arbitrary state selection |
| False validation against 1vN/Redfield | Compare only derived common limits and document differing prescriptions |
| Python/Cython divergence | One canonical generator and typed lowering; forced-backend parity |
| Performance or memory explosion | Profile diagram generation/evaluation separately before compilation; publish scaling limits |
| Incorrect full-space current | Marked-current-vertex derivation and population-projection regression |
| Counting kernel drifts from physical kernel | One diagram-record stream; zero-field and derivative identity tests |
| Incorrect counting-resolved coherence elimination | Derived product/inverse derivatives plus finite-difference oracle |
| RTDnoise compatibility drift | Preserve Simon fixtures and order-separated diagnostics until alias migration is complete |
| Scope creep into energy-current noise | Explicit non-goal and feature gate |

## 11. Definition of done

The feature is complete only when:

- legacy `RTD` and `pyRTD` reproduce the phase-0 block and observable fixtures;
- the generic engine produces all `dd`, `dn`, `nd`, and `nn` blocks at both
  retained orders from one set of topology rules;
- first- and second-order coherent solvers use canonical `dm0` indexing;
- trace preservation, Hermiticity preservation, stationary charge
  conservation, equilibrium, gauge covariance, and degeneracy tests pass;
- solver diagnostics expose nonuniqueness, conditioning, residual, trace, and
  positivity information without artificially repairing the state;
- the second-order `dd` projection reproduces legacy RTD before any new
  coherence block is enabled;
- particle current uses a derived full-Liouville current kernel;
- `RTD`/`pyRTD` and coherent RTD expose opt-in particle-current noise from the
  same diagram records and zero-field kernel used for their stationary state;
- first and second counting derivatives, Laplace derivatives, and mixed
  derivatives are derived and tested through both retained orders;
- default legacy RTD noise includes the counting-resolved coherence correction,
  while the historical `off_diag_corrections=False` RTDnoise results and Simon
  reference data remain reproducible;
- `RTDnoise`/`pyRTDnoise` are compatibility aliases, not independent kernel
  implementations, and retain their sequential and order-truncated diagnostics;
- every shipped observable is either fully derived and tested or fails clearly
  as unsupported;
- pure-Python and compiled paths pass separately with confirmed backend status;
- slow Python examples, explicitly selected notebooks, and documentation pass
  their separate release gates; and
- the theory/convention note, reference provenance, API documentation, and
  changelog accurately describe the implemented approximation and its known
  limits.
