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
already contain the factor $1/2\pi$, so that
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

## Which shift belongs to this dissipator

The two-argument weight is

$$
w(x_1,x_2) = \frac{1}{2}\left[\Lambda(x_1)+\Lambda(x_2)\right]
           + \delta w(x_1,x_2).
$$

The arithmetic mean alone is the principal-value part of the second-order lead
self-energy, and it is the shift that accompanies a *Bloch-Redfield* kernel. It is
not the shift that accompanies the dissipator QmeQ actually uses. The jump
operators of the Lindblad approach dress each tunneling matrix element with the
square root of an occupation factor and keep one operator per lead rather than one
per Bohr frequency
([Kirsanskas, Franckie & Wacker 2018](https://doi.org/10.1103/PhysRevB.97.035432)),
the construction later derived as a controlled weak-coupling approximation by
[Nathan & Rudner 2020](https://doi.org/10.1103/PhysRevB.102.115109). The Hermitian
shift generated alongside those jump operators carries the same square roots, and
its weight is

$$
w(x_1,x_2) = -\mathcal{P}\!\!\int\!\mathrm{d}v\,
             \frac{\sqrt{J(x_1-v)J(x_2-v)}}{v},
$$

with $J=f$ on the particle family and $J=1-f$ on the hole family. Subtracting the
arithmetic mean, which is the same object at coincident arguments, leaves

$$
\delta w(x_1,x_2) = \frac{1}{2}\mathcal{P}\!\!\int\!\mathrm{d}v\,
   \frac{\left[\sqrt{J(x_1-v)}-\sqrt{J(x_2-v)}\right]^{2}}{v},
$$

computed by `func_ule_shift`. Three properties matter in practice. The difference of
square roots cancels the bandwidth logarithm, so $\delta w$ is cutoff free even
though neither weight is on its own. It vanishes identically when $x_1=x_2$, so the
two shifts agree on every diagonal element, where the level shift is ordinary
second-order perturbation theory and not a matter of choice; they differ only off
the diagonal, which is where a nonsecular generator has an error bar anyway. And
$\delta w$ is **not** even in its arguments, so unlike $\Lambda$ the hole family
may not be obtained by reversing the chemical potential: its arguments are written
in the direction the hole family runs.

$H_{LS}$ is Hermitian and block diagonal in the charge, as it must be because
the charge of the total system is conserved. For a single spinless level, or for any
charge sector that is one-dimensional or uniformly shifted (for example a
spin-degenerate orbital without a magnetic field), the Lamb shift reduces to a
constant within each charge sector, drops out of the commutator and leaves the
currents unchanged. It matters whenever coherences between states of the same charge
matter, which is exactly the regime in which the Lindblad approach differs from the
Pauli master equation.

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

`'quad'` evaluates the same principal values as `'digamma'` over the actual
band, so it keeps the bandwidth constant the digamma form drops and converges
to `'digamma'` as `dband` grows. The geometric correction $\delta w$ is cutoff
free either way.

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
