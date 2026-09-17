# Lamb shift in the Lindblad approach

The Lindblad approach of QmeQ describes the quantum dot by a master equation of
Gorini-Kossakowski-Sudarshan-Lindblad form,

$$
\dot{\rho} = -i\left[H_{QD}+H_{LS},\rho\right] + \mathcal{D}[\rho],
$$

where $\mathcal{D}$ is the dissipator built from the jump operators
$T^{l}_{ba}$ of Appendix F of the
[QmeQ paper](https://doi.org/10.1088/1361-648X/aa9c15), and $H_{LS}$ is the
Lamb shift, i.e. the renormalisation of the many-body energies of the dot caused by
the coupling to the leads. It is included by default and dropped with
`principal_part='omit'`.

The many-body eigenstates of the dot are labelled following the QmeQ convention: the
states $b$, $b'$, $b''$ carry $N$ electrons, the states
$a$ carry $N-1$, and the states $c$ carry $N+1$ electrons.
All expressions are written in the eigenbasis of $H_{QD}$, in which the
tunneling amplitudes are the rotated matrix elements $T^{l}_{ba}$ (these
contain the amplitude factor $1/\sqrt{2\pi}$, so that
$\sum_{j}|t_{lj}|^{2}=\Gamma_{l}/2\pi$).

## Lamb shift Hamiltonian

Beyond the secular approximation, i.e. keeping all energy differences
$\omega_{bb'}=E_b-E_{b'}$ of the spectrum and not only those of neighbouring
levels, the Lamb shift of lead $l$ is

$$
(H_{LS}^{l})_{bb'} = \sum_{a}T^{l}_{ba}T^{l}_{ab'}\,
                     w\!\left(x^{a}_{b},x^{a}_{b'}\right)
                   + \sum_{c}T^{l}_{bc}T^{l}_{cb'}\,
                     w^{h}\!\left(\tilde{x}^{c}_{b},\tilde{x}^{c}_{b'}\right),
$$

with $H_{LS}=\sum_{l}H_{LS}^{l}$ and the scaled transition energies
$x^{a}_{b}=(E_b-E_a-\mu_l)/T_l$, $\tilde{x}^{c}_{b}=(E_c-E_b-\mu_l)/T_l$.
The two sums are the particle and the hole
contribution, i.e. the two terms of the anticommutator of the tunneling operators.

The principal value factors are the odd Fourier transforms of the lead correlation
functions,

$$
\Lambda_{l}(E) = \mathcal{P}\!\!\int_{D_-}^{D_+}\!\!\mathrm{d}\omega\,
                 \frac{f\big((\omega-\mu_{l})/T_{l}\big)}{\omega-E}
               \approx \mathrm{Re}\,\psi\!\left(\frac{1}{2}
                       + i\frac{E-\mu_{l}}{2\pi T_{l}}\right)
                 - \ln\frac{D}{2\pi T_{l}},
$$

and $\tilde{\Lambda}_{l}(E)=\Lambda_{l}(E)|_{\mu_l\to-\mu_l}$, where
$\psi$ is the digamma function. The right hand side is the standard wide-band
expansion, the same approximation that `itype=1` uses for the principal parts of the
1vN, Redfield and RTD kernels.

## Derivation with the corrected source indices

Use Nathan-Rudner Eq. (D7) together with **corrected D8**, Eq. (6) of the
[2021 erratum](https://doi.org/10.1103/PhysRevB.104.119901). Both the original
main-text Eq. (34) and the original D8 contain misprints. Writing the outer
states as $b,b'$ and the intermediate as $k$, the source says

$$
p=E_k-E_b,\qquad q=E_{b'}-E_k,\qquad
F(p,q)=-2\pi\gamma\,\mathcal P\int\frac{d\omega}{\omega}
                     g(\omega-p)g(\omega+q).
$$

These arguments follow directly by inserting the energy eigenoperators into
D1. Exchanging $b,b'$ sends $(p,q)$ to $(-q,-p)$ and conjugate-transposes
the spectral matrix product, proving Hermiticity. Reversing both arguments
also preserves Hermiticity, so that property alone cannot choose the physical
expression. The corrected indices agree with D3 and perturbation theory.

For one normal lead channel, strip the tunnel amplitudes out of its two
spectra. At bath energy $\omega$ they are
$A(\omega)=1-f_\alpha(\omega)$ and $C(\omega)=f_\alpha(-\omega)$,
where $f_\alpha(\epsilon)=f((\epsilon-\mu_\alpha)/T_\alpha)$ and
$f(x)=1/(1+e^x)$. The Hermitian bath quadratures diagonalize into these two
channels. Their square-root products select $A$ for $M_{ba}M^*_{b'a}$ and
$C$ for $M^*_{cb}M_{cb'}$, not the occupations of the opposite jump.

For the **lower intermediate** $a$, put $\Delta_i=E_{b_i}-E_a$.
D7 samples $A(\omega+\Delta_i)$. With $v=-\omega/T$ and
$x_i=(\Delta_i-\mu)/T$, its weight is

$$
w_<(x_1,x_2)=+\mathcal P\int\frac{dv}{v}
             \sqrt{[1-f(x_1-v)][1-f(x_2-v)]}.
$$

For the **upper intermediate** $c$, put $\Delta_i=E_c-E_{b_i}$.
D7 samples $C(\omega-\Delta_i)=f_\alpha(\Delta_i-\omega)$.
With $v=\omega/T$ and $y_i=(\Delta_i-\mu)/T$, the weight is

$$
w_>(y_1,y_2)=-\mathcal P\int\frac{dv}{v}
             \sqrt{f(y_1-v)f(y_2-v)}.
$$

These formulas use QmeQ's $t=M/\sqrt{2\pi}$ normalization; in the
rate-amplitude convention $M$ each weight has an additional $1/(2\pi)$.
The two weights are dimensionless, and $t t^*$ supplies the energy unit.
The $T$ in $d\omega$ cancels the denominator; $\mu$ is included in each
scaled argument, not in the integration variable.

## One cutoff-free correction, two energy orientations

Define $S(x)=\operatorname{Re}\psi(1/2+ix/(2\pi))$ and

$$
\delta_f(u_1,u_2)=\frac12\mathcal P\int\frac{dv}{v}
   [\sqrt{f(u_1-v)}-\sqrt{f(u_2-v)}]^2.
$$

Expanding the square, then changing $v\to-v$ for the lower intermediate,
gives the minimal wide-band implementation (common bandwidth constant omitted):

$$
\boxed{w_<(x_1,x_2)=\tfrac12[S(x_1)+S(x_2)]+\delta_f(-x_1,-x_2)},
$$
$$
\boxed{w_>(y_1,y_2)=\tfrac12[S(y_1)+S(y_2)]+\delta_f(y_1,y_2)}.
$$

Thus `func_ule_shift` has one Fermi branch. Its lower-intermediate arguments
are $(E_a-E_b+\mu)/T$ and $(E_a-E_{b'}+\mu)/T$; its upper-intermediate
arguments are $(E_c-E_b-\mu)/T$ and $(E_c-E_{b'}-\mu)/T$.
**$S$ is even, but $\delta_f$ is not.** If a separate hole correction is
introduced, it obeys $\delta_h(x_1,x_2)=-\delta_f(-x_1,-x_2)$ and must
be SUBTRACTED for the lower intermediate. Adding $\delta_f(x)$ below and
$\delta_h(y)$ above passes every diagonal test but gives the wrong ULE.

The correction vanishes for equal arguments, is symmetric under argument
exchange, and is quadratic in a small mismatch. These structural checks
are necessary, not sufficient. Tests evaluate the two direct geometric
principal-value integrals independently at asymmetric arguments, nonzero
chemical potential, and with complex tunneling amplitudes.

## Independent sign anchors

Second-order Rayleigh-Schrödinger perturbation theory fixes the diagonal:

$$
\delta E_b^{<}=\sum_a |t_{ba}|^2\mathcal P\int d\epsilon\,
 \frac{1-f_\alpha(\epsilon)}{E_b-E_a-\epsilon},\qquad
\delta E_b^{>}=\sum_c |t_{cb}|^2\mathcal P\int d\epsilon\,
 \frac{f_\alpha(\epsilon)}{E_b-E_c+\epsilon}.
$$

Both reduce to $S((\Delta-\mu)/T)-\ln(D/(2\pi T))$ in a symmetric
wide band. The opposite signs in the geometric integrals above do **not**
mean opposite bandwidth shifts: the occupied and empty spectra have their
nonzero tails on opposite sides of the principal-value pole. Both physical
contributions have the same negative logarithm. Hence the bandwidth term
is the fermionic anticommutator, not their difference.

## The bandwidth constant

[`func_lambshift`](../api/specfunc.md) returns only the digamma term and
drops the bandwidth constant $-\ln(D/2\pi T_{l})$. The constant is the same for
every pair of states, so its contribution to $H_{LS}$ is proportional to

$$
\sum_{a}T^{l}_{ba}T^{l}_{ab'} + \sum_{c}T^{l}_{bc}T^{l}_{cb'}
= \left(X_{l}X_{l}^{\dagger}+X_{l}^{\dagger}X_{l}\right)_{bb'}
= \delta_{bb'}\sum_{j}|t_{lj}|^{2},
$$

where $X_{l}=\sum_{j}t_{lj}d_{j}$ is the tunneling operator of lead $l$.
The anticommutator of the fermion operators makes this exactly proportional to the
identity within each charge sector, so the dropped constant only shifts all states of
a charge sector by the same amount and cancels in the commutator
$[H_{LS},\rho]$. This holds for a symmetric wide band; the residual asymmetry
$\ln|D_-/D_+|$ between the particle and hole factors is beyond the accuracy of
the digamma approximation itself.

## Contribution to the kernel

QmeQ stores the kernel $K$ of the master equation such that the free evolution
of a coherence is $K_{(bb'),(bb')}=E_b-E_{b'}$ (see
[`KernelHandler.set_energy`](../api/approach.md)), i.e. $K=iL$
for $\dot{\rho}=L\rho$. Writing out the commutator,

$$
i\dot{\rho}_{bb'} \supset \sum_{b''}\left[(H_{LS})_{bb''}\rho_{b''b'}
                                - \rho_{bb''}(H_{LS})_{b''b'}\right],
$$

the Lamb shift therefore adds

$$
K_{(bb'),(b''b')} \mathrel{+}= (H_{LS})_{bb''}, \qquad
K_{(bb'),(bb'')} \mathrel{-}= (H_{LS})_{b''b'} ,
$$

which is what [`ApproachLindblad.generate_coupling_terms`](../api/approach.md)
inserts. Because the two terms come from a commutator with a Hermitian operator,
they cancel in the sum over the diagonal rows of the kernel, so the Lamb shift is
trace preserving; the test suite asserts this.

The sign convention is fixed by second-order perturbation theory: for a single level
$\varepsilon$ coupled to one lead, the occupied state is shifted by
$\delta E_1 = |t|^{2}\,\Lambda(\varepsilon)$ and the empty state by
$\delta E_0 = |t|^{2}\,\tilde{\Lambda}(-\varepsilon)$, which is the same
convention in which the real parts of the 1vN and Redfield principal parts
([`func_1vN`](../api/specfunc.md)) enter their kernels. The test suite
verifies that the level renormalisation extracted from $H_{LS}$ agrees with the
1vN factors.

## Switching the Lamb shift on and off

`principal_part` selects the shift and `bandwidth` the dissipator; they are
independent, and `itype` does not select the shift. See
[Transport integration options](transport-options.md) for the cross-approach table.

| Option | Lamb shift | Dissipator |
|---|---|---|
| `bandwidth='finite'` | unchanged | outside-band transitions dropped |
| `bandwidth='infinite'` (default) | unchanged | infinite bandwidth |
| `principal_part='digamma'` (default) | wide-band digamma form | unchanged |
| `principal_part='quad'` | integrated over the band | unchanged |
| `principal_part='omit'` | **neglected** | unchanged |

`'quad'` evaluates the arithmetic principal values over the actual band,
while retaining the wide-band geometric correction. It converges to the
wide-band result, but is not the exact ULE for a finite spectral band: that
would require band support inside both square roots and the physical
finite-band particle/removal denominators.

```python
system = qmeq.Builder(nsingle, hsingle, coulomb, nleads, tleads,
                      mulst, tlst, dband, kerntype='Lindblad',
                      bandwidth='infinite', principal_part='digamma')
system.solve()
shifts = system.appr.HLS.sum(axis=0)   # HLS is lead-resolved
```

The electron-phonon Lindblad approach reuses the electron-lead part of the kernel and
therefore also picks up the Lamb shift of the leads. The phonon baths do not
contribute a Lamb shift; their `itype_ph` flag is unrelated and unchanged.

## Limitations

* The phonon-induced Lamb shift of the electron-phonon variants is not included.
* Adding the Lamb shift makes the kernel stiffer. For weakly coupled models whose
  currents are many orders of magnitude below the level spacing, check the solution
  (for example with `symq=False`, or by comparing the lead currents) before trusting
  the last digits.
