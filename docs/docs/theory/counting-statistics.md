# Zero-frequency current counting statistics

QmeQ calculates the first two zero-frequency cumulants of the particle current:
the mean current $I$ and noise $S$. The implementation follows the
counting-field formulation of
[Emary, Phys. Rev. B 80, 235306 (2009)](https://arxiv.org/abs/0902.3544). It was
developed by Simon Wozny in his
[QmeQ fork](https://github.com/si8881wo/qmeq); his
[example notebook](https://github.com/si8881wo/qmeq-noise-example) is adapted
as tutorial 7 in this documentation.

## First-order formula

For Pauli, Lindblad, Redfield, and 1vN, QmeQ assigns an independent counting
field to every selected lead and decomposes the Markovian kernel as

$$
K(\boldsymbol\chi) = K
+ \sum_i\left[(e^{i\chi_i}-1)J_{i,+}
+(e^{-i\chi_i}-1)J_{i,-}\right] .
$$

The order of $i$ is exactly the order supplied in `countingleads`.
Setting all selected fields equal gives the aggregate result retained in
`current_noise`.

Let $|0\rangle\!\rangle$ be the normalized stationary state,
$\langle\!\langle\tilde 0|$ the approach's trace vector, and

$$
Q = 1-|0\rangle\!\rangle\langle\!\langle\tilde 0|,
\qquad R = Q K^+ Q,
$$

where $K^+$ is the Moore--Penrose pseudoinverse. With
$K_i^{(1)}=i(J_{i,+}-J_{i,-})$ and
$K_{ij}^{(2)}=-\delta_{ij}(J_{i,+}+J_{i,-})$, QmeQ evaluates

$$
I_i = -i\langle\!\langle\tilde 0|K_i^{(1)}|0\rangle\!\rangle,
$$

$$
S_{ij} = -\langle\!\langle\tilde 0|
    \left[K_{ij}^{(2)}-K_i^{(1)}RK_j^{(1)}
    -K_j^{(1)}RK_i^{(1)}\right]
    |0\rangle\!\rangle.
$$

The physical kernel is checked for exactly one stationary state. A disconnected
model with a non-unique stationary state raises `LinAlgError` instead of
returning a basis-dependent cumulant. The same Python counting-kernel and
pseudoinverse code runs after either the pure-Python or compiled first-order
solver.

## API and conventions

Counting is opt-in. `None` is the default and performs no counting-kernel
allocation. A nonempty iterable must contain unique integer lead indices:

```python
import numpy as np
import qmeq

system = qmeq.Builder(
    nsingle=1,
    hsingle={(0, 0): 0.0},
    nleads=2,
    tleads={(0, 0): 0.1, (1, 0): 0.1},
    mulst={0: 1.0, 1: -1.0},
    tlst={0: 0.2, 1: 0.2},
    dband={0: 100.0, 1: 100.0},
    kerntype="Pauli",
    countingleads=[0],
)
system.solve()
current, noise = system.current_noise
covariance = system.current_noise_matrix
```

`current_noise` is `[I, S]`. The current uses QmeQ's convention: positive
means particle flow from the counted lead into the dot. Consequently its first
entry equals the sum of the corresponding entries in `system.current`.
`current_noise_matrix` is the symmetric matrix $S_{ij}$ ordered as
`countingleads`. The aggregate quantities obey

$$
I = \sum_i I_i, \qquad S = \sum_{ij} S_{ij}.
$$

For arbitrary real channel weights $w_i$, the weighted current and noise
are therefore

```python
leads = np.asarray(system.countingleads)
weights = np.asarray([0.5, -0.5])
weighted_current = weights @ system.current[leads]
weighted_noise = weights @ system.current_noise_matrix @ weights
```

For spin-resolved channels ordered as up and down, `weights = [0.5, -0.5]`
directly gives the $S_z$ current and noise. Equivalently, its noise can be
reconstructed from three aggregate calculations as
$(2S_\uparrow+2S_\downarrow-S_{\uparrow+\downarrow})/4$.
Changing `system.countingleads` before another solve changes the counted
aggregate; assigning `None` disables it again.

QmeQ uses natural units $\hbar=k_\mathrm{B}=|e|=1$. Thus particle current
and zero-frequency particle-number noise have units of inverse time (energy in
these units). To obtain charge cumulants, multiply $I$ by the signed
electron charge according to the desired electrical-current convention and
$S$ by $e^2$.

## RTD results and approximation order

Use `kerntype='pyRTDnoise'` for Real Time Diagrammatic counting statistics.
`kerntype='RTDnoise'` is a documented alias for that same pure-Python
implementation:

```python
system = qmeq.Builder(
    # model parameters as above
    kerntype="RTDnoise",
    countingleads=[0],
    off_diag_corrections=True,
)
system.solve()
full = system.current_noise
full_covariance = system.current_noise_matrix
sequential = system.current_noise_first
sequential_covariance = system.current_noise_matrix_first
truncated = system.current_noise_o4trunc
```

The arrays mean:

* `current_noise` is `[I, S]` from the full fourth-order RTD kernel and its
  stationary state.
* `current_noise_first` is the sequential `[I, S]` result.
* `current_noise_matrix` and `current_noise_matrix_first` are the
  corresponding lead-resolved covariance matrices.
* `current_noise_o4trunc` is
  `[I_sequential, I_fourth_order, S_sequential, S_fourth_order]`, where the
  latter member of each pair is evaluated with the consistent fourth-order
  truncation.

Here *fourth order* always means fourth order in the tunnelling Hamiltonian
$H_T$, equivalently second order in the tunnel rate $\Gamma$. It
does not mean fourth order in $\Gamma$. RTD noise includes the
non-Markovian energy-derivative terms of Emary's formulation.

## Limitations

Only the first two zero-frequency particle-current cumulants are implemented.
There are no arbitrary higher cumulants or energy-current noise. The auxiliary
`current_noise_o4trunc` order decomposition has no matrix-valued companion.
Counting is not implemented for 2vN, electron-phonon approaches, or matrix-free
solvers. RTD counting includes the same first-order coherence-elimination
correction as ordinary RTD when `off_diag_corrections=True` (the default).
The compatibility mode `False` remains available and reproduces the historical
RTDnoise kernel. The counted correction has been validated against exact
non-interacting transport for real tunnel amplitudes. Generic complex tunnel
amplitudes remain outside the validated RTDnoise domain because the separate
second-order population traversal still drops an imaginary channel.

The published unequal-temperature RTD integrals retain `dband` as a finite
wide-band regulator. Thermal-bias RTD counting calculations must therefore be
repeated with increasing `dband` until every reported current and noise
cumulant converges. QmeQ emits `RTDBandwidthWarning` when the cutoff is below
the conservative diagnostic ratio described in
[Transport integration options](transport-options.md).
