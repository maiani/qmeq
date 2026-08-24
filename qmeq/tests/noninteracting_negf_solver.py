"""Exact non-interacting transport reference for validating QmeQ's approaches.

It is *deliberately independent* of ``qmeq.approach`` and ``qmeq.builder``: it
takes a bare single-particle Hamiltonian and bare channel coupling vectors as
plain arrays and builds its own scattering problem. Nothing here may import
from the solver or builder packages, because a convention bug in a shared layer
is exactly the class of bug a comparison through that layer cannot detect.

Validity: exact at zero interaction (:math:`U=0`) only, at all orders in
:math:`\\Gamma`, for arbitrary complex hopping, arbitrary non-collinear Zeeman
fields, arbitrary bias, and unequal lead temperatures. It says nothing about
any interacting model.

Conventions, fixed here and pinned by ``test_noninteracting_negf_solver.py``:

* :math:`e=\\hbar=k_B=1`.
* The tunnelling Hamiltonian is
  :math:`H_T=\\sum_{c j}(g_{cj}\\,\\gamma_c^\\dagger d_j + \\mathrm{h.c.})` with
  :math:`2\\pi` absorbed into :math:`g`, so the level-width matrix of channel
  :math:`c` is :math:`\\Gamma_c=u_cu_c^\\dagger` with
  :math:`(u_c)_j=g_{cj}^{\\ast}` -- the *conjugate* amplitude; see
  :meth:`NoninteractingModel.amplitude_matrix`. This is QmeQ's own
  normalisation of ``tleads`` up to
  :math:`g=\\sqrt{2\\pi}\\,t^{\\ast}`.  The conjugation follows from QmeQ
  storing the electron-adding matrix element as ``tleads`` whereas ``g``
  multiplies :math:`\\gamma_c^\\dagger d_j` in the convention above.
* Currents are **positive inward** (into the dot).
* The zero-frequency noise is
  :math:`S=\\lim_{t\\to\\infty}\\mathrm d\\,\\mathrm{Var}X(t)/\\mathrm dt`, with
  **no** additional factor of two.

Cumulants come from the Levitov-Lesovik-Klich determinant formula. Its
derivatives are extracted by a Cauchy contour integral rather than by a finite
difference, so there is no step size to tune.
"""

from dataclasses import dataclass
import math
import numbers

import numpy as np
from scipy.integrate import quad_vec

def fermi(energy, mu, temperature):
    """Fermi function, written to avoid overflow at large ``|energy - mu|``."""
    x = (np.asarray(energy, dtype=float) - mu) / temperature
    # expit(-x) without importing scipy.special, branch-free in both tails.
    out = np.empty_like(x)
    positive = x > 0
    out[positive] = np.exp(-x[positive]) / (1.0 + np.exp(-x[positive]))
    out[~positive] = 1.0 / (1.0 + np.exp(x[~positive]))
    return out


@dataclass(frozen=True)
class NoninteractingModel:
    """A non-interacting dot wired to independent reservoir channels.

    Parameters
    ----------
    hamiltonian : (n, n) complex
        Single-particle Hamiltonian ``h`` with
        ``H = sum_{jk} h[j, k] d_j^dagger d_k``. Must be Hermitian.
    couplings : (n, nchannels) complex
        Column ``c`` holds the physical tunnelling amplitudes ``g_{cj}`` of
        channel ``c``, exactly as they appear in ``H_T``. The vector entering
        the level-width matrix is their conjugate; see
        :meth:`amplitude_matrix`.
    mu, temperature : (nchannels,) float
        Per-channel chemical potential and temperature. Channels of the same
        physical lead simply repeat that lead's values.
    lead_of_channel : (nchannels,) int
        Which lead each channel belongs to, used only to aggregate currents.
    """

    hamiltonian: np.ndarray
    couplings: np.ndarray
    mu: np.ndarray
    temperature: np.ndarray
    lead_of_channel: np.ndarray

    def __post_init__(self):
        h = np.asarray(self.hamiltonian, dtype=complex)
        a = np.asarray(self.couplings, dtype=complex)
        if h.ndim != 2 or h.shape[0] != h.shape[1]:
            raise ValueError("hamiltonian must be square.")
        if not np.allclose(h, h.conj().T, atol=1e-12, rtol=0.0):
            raise ValueError("hamiltonian must be Hermitian.")
        if a.ndim != 2 or a.shape[0] != h.shape[0]:
            raise ValueError("couplings must have one row per dot state.")
        for name in ("mu", "temperature", "lead_of_channel"):
            value = np.asarray(getattr(self, name))
            if value.shape != (a.shape[1],):
                raise ValueError(f"{name} must have one entry per channel.")
        if np.any(np.asarray(self.temperature) <= 0.0):
            raise ValueError("temperature must be positive.")

    @property
    def nstates(self):
        return np.asarray(self.hamiltonian).shape[0]

    @property
    def nchannels(self):
        return np.asarray(self.couplings).shape[1]

    @property
    def nleads(self):
        return int(np.max(self.lead_of_channel)) + 1

    def amplitude_matrix(self):
        """``A`` with ``Gamma_c = outer(A[:, c], conj(A[:, c]))``, i.e. ``conj(g)``.

        Contracting the reservoir in
        ``H_T = sum_{cj} (g_{cj} gamma_c^dagger d_j + h.c.)`` gives a self-energy
        proportional to ``conj(g_{cj}) g_{ck}``, so the vector that enters the
        level-width matrix is the **conjugate** of the tunnelling amplitude.

        This is not cosmetic. With ``A = g`` instead, the combination that
        survives in an observable is ``a_{alpha 1} - a_{12} - a_{alpha 2}``,
        which is *not* invariant under rephasing the dot states, so the answer
        depends on an unphysical phase choice. With ``A = conj(g)`` it is
        ``-(a_{alpha 1} + a_{12} - a_{alpha 2})``, minus the oriented plaquette
        flux, which is invariant. ``test_gauge_invariance_of_observables`` pins
        this.
        """
        return np.conj(np.asarray(self.couplings, dtype=complex))

    def width_matrix(self):
        """Total level-width matrix ``sum_c Gamma_c = A A^dagger``."""
        a = self.amplitude_matrix()
        return a @ a.conj().T

    def scattering_matrix(self, energy):
        """Unitary channel scattering matrix ``S(E) = 1 - i A^dagger G^r A``."""
        h = np.asarray(self.hamiltonian, dtype=complex)
        a = self.amplitude_matrix()
        identity = np.eye(self.nstates, dtype=complex)
        inverse_g = energy * identity - h + 0.5j * self.width_matrix()
        resolvent = np.linalg.solve(inverse_g, a)
        return np.eye(self.nchannels, dtype=complex) - 1.0j * a.conj().T @ resolvent

    def occupations(self, energy):
        """Per-channel Fermi occupations at ``energy``."""
        return np.array([
            fermi(energy, mu, temperature)
            for mu, temperature in zip(self.mu, self.temperature)
        ])

    def energy_window(self, margin_temperature=40.0, margin_width=200.0):
        """A finite integration window that contains all spectral weight.

        Widened by ``margin_temperature`` thermal widths and
        ``margin_width`` half-widths around every chemical potential and every
        eigenvalue of the effective Hamiltonian. ``integration_is_converged``
        checks that the window is in fact wide enough.
        """
        width = np.linalg.eigvalsh(self.width_matrix())
        scale = max(float(np.max(width)), 0.0)
        levels = np.linalg.eigvalsh(np.asarray(self.hamiltonian, dtype=complex))
        anchors = np.concatenate([np.asarray(self.mu, dtype=float), levels])
        margin = (
            margin_temperature * float(np.max(self.temperature))
            + margin_width * scale
        )
        return float(np.min(anchors) - margin), float(np.max(anchors) + margin)


def _qmeq_matrix(values, nrow, ncol, name, hermitian=False):
    """Convert QmeQ's dictionary/list/array matrix inputs without importing QmeQ.

    QmeQ uses ``{(row, col): value}`` or ``[[row, col, value], ...]`` for
    sparse inputs.  For ``hsingle`` an off-diagonal entry denotes its Hermitian
    partner too, as it does for QmeQ with its default ``herm_hs=True``.
    """
    if isinstance(values, np.ndarray):
        matrix = np.asarray(values, dtype=complex)
        if matrix.shape != (nrow, ncol):
            raise ValueError(f"{name} array must have shape {(nrow, ncol)}.")
        if not hermitian:
            return matrix.copy()
        # QmeQ's dense hsingle input is read from its upper triangle when
        # herm_hs=True, then completed by conjugation.
        return np.triu(matrix) + np.triu(matrix, k=1).conj().T
    matrix = np.zeros((nrow, ncol), dtype=complex)
    if isinstance(values, dict):
        entries = ((row, col, value) for (row, col), value in values.items())
    elif isinstance(values, list):
        entries = values
    elif values is None:
        entries = ()
    else:
        raise TypeError(f"{name} must be a QmeQ-style dict, list, or array.")
    for row, col, value in entries:
        if not (0 <= row < nrow and 0 <= col < ncol):
            raise ValueError(f"{name} entry {(row, col)} is out of range.")
        matrix[row, col] += value
        if hermitian and row != col:
            matrix[col, row] += np.conj(value)
    return matrix


def _qmeq_lead_array(values, nleads, name):
    """Convert QmeQ's scalar/dictionary/list lead parameters to an array."""
    if isinstance(values, numbers.Number):
        return np.full(nleads, values, dtype=float)
    if isinstance(values, dict):
        result = np.zeros(nleads, dtype=float)
        for lead, value in values.items():
            if not 0 <= lead < nleads:
                raise ValueError(f"{name} entry {lead} is out of range.")
            result[lead] = value
        return result
    result = np.asarray(values, dtype=float)
    if result.shape != (nleads,):
        raise ValueError(f"{name} must have one entry per lead.")
    return result


def model_from_qmeq(nsingle, hsingle=None, coulomb=None, nleads=0,
                    tleads=None, mulst=None, tlst=None, herm_hs=True):
    """Build an exact NEGF model from QmeQ's non-interacting input vocabulary.

    This is a convenience adapter, not a QmeQ ``kerntype``: it deliberately
    does not accept a ``Builder`` or import QmeQ's construction code.  Pass the
    same ``nsingle``, ``hsingle``, ``nleads``, ``tleads``, ``mulst`` and ``tlst``
    values used to define a QmeQ model.  Each QmeQ lead becomes one reservoir
    channel; use separate lead labels for independently occupied channels.

    ``coulomb`` is accepted solely to make accidental use on an interacting
    QmeQ definition fail loudly.  It must be empty or contain only exact zero
    entries.  The adapter supports QmeQ's dictionary, list, and dense-array
    forms, and mirrors ``herm_hs`` for sparse ``hsingle`` input.
    """
    if not isinstance(nsingle, numbers.Integral) or nsingle <= 0:
        raise ValueError("nsingle must be a positive integer.")
    if not isinstance(nleads, numbers.Integral) or nleads <= 0:
        raise ValueError("nleads must be a positive integer.")
    if coulomb is not None:
        if isinstance(coulomb, dict):
            interaction_values = coulomb.values()
        elif isinstance(coulomb, np.ndarray):
            interaction_values = np.asarray(coulomb).ravel()
        else:
            interaction_values = (entry[-1] for entry in coulomb)
        if any(value != 0 for value in interaction_values):
            raise ValueError("NEGF is exact only for coulomb=0.")
    hamiltonian = _qmeq_matrix(
        hsingle, nsingle, nsingle, "hsingle", hermitian=herm_hs
    )
    if not np.allclose(hamiltonian, hamiltonian.conj().T, atol=1e-12, rtol=0.0):
        raise ValueError("hsingle must define a Hermitian Hamiltonian.")
    # QmeQ indexes tleads as (lead, state) and stores the electron-adding
    # matrix element.  Here g multiplies gamma^dagger d, so g=sqrt(2*pi)*t^*;
    # NEGF stores channels as columns.
    tleads_matrix = _qmeq_matrix(tleads, nleads, nsingle, "tleads")
    return NoninteractingModel(
        hamiltonian=hamiltonian,
        couplings=np.sqrt(2.0 * np.pi) * tleads_matrix.T.conj(),
        mu=_qmeq_lead_array(mulst, nleads, "mulst"),
        temperature=_qmeq_lead_array(tlst, nleads, "tlst"),
        lead_of_channel=np.arange(nleads),
    )


def _log_det_contour(model, energy, weights, radius, nnodes):
    """``ln det M(lambda)`` on a circle of radius ``radius`` in complex lambda.

    The Levitov-Lesovik-Klich kernel is

    ``M(lambda) = 1 + n (S^dagger e^{-i lambda Q} S e^{+i lambda Q} - 1)``,
    ``Q = diag(weights)``,

    written with ``e^{-i lambda Q}`` rather than a complex conjugate so that it
    is an analytic function of ``lambda`` and its Taylor coefficients are the
    cumulants. ``M(0) = 1``, so for a small enough radius ``det M`` has no zero
    inside the contour; the phase is unwrapped along the circle and the winding
    is checked, which is what detects a radius that is too large.
    """
    scattering = model.scattering_matrix(energy)
    occupation = model.occupations(energy)
    identity = np.eye(model.nchannels, dtype=complex)
    lam = radius * np.exp(2.0j * np.pi * np.arange(nnodes) / nnodes)

    values = np.empty(nnodes, dtype=complex)
    for index, value in enumerate(lam):
        phases = np.exp(1.0j * value * weights)
        # S^dagger diag(1/p) S diag(p)
        scaled = (scattering / phases[:, np.newaxis]) * phases[np.newaxis, :]
        dressed = scattering.conj().T @ scaled
        kernel = identity + occupation[:, np.newaxis] * (dressed - identity)
        sign, logabs = np.linalg.slogdet(kernel)
        if sign == 0:
            raise FloatingPointError(
                f"det M vanished at E={energy!r} on the counting contour; "
                "reduce radius."
            )
        values[index] = logabs + np.log(sign)

    closed = np.concatenate([values.imag, values.imag[:1]])
    unwrapped = np.unwrap(closed)
    winding = unwrapped[-1] - unwrapped[0]
    if abs(winding) > 1e-6:
        raise FloatingPointError(
            f"ln det M wound by {winding:.3g} around the counting contour at "
            f"E={energy!r}; det M has a zero inside it. Reduce radius."
        )
    return values.real + 1.0j * unwrapped[:nnodes]


def _cumulant_integrand(model, energy, weights, order, radius, nnodes):
    """Taylor coefficients of the CGF at one energy, as cumulants.

    The Cauchy extraction: with ``F(lambda) = sum_m a_m lambda^m`` sampled on a
    circle of radius ``r``, ``a_m`` is the ``m``-th DFT mode divided by ``r^m``,
    and ``C_m = (-i d/dlambda)^m F|_0 = (-i)^m m! a_m``.
    """
    logdet = _log_det_contour(model, energy, weights, radius, nnodes)
    modes = np.fft.fft(logdet) / nnodes
    orders = np.arange(order + 1)
    factorials = np.array([math.factorial(int(m)) for m in orders], dtype=float)
    taylor = modes[: order + 1] / radius**orders
    return ((-1.0j) ** orders) * factorials * taylor / (2.0 * np.pi)


def cumulants(model, weights, order=2, radius=0.35, nnodes=32,
              window=None, epsabs=1e-13, epsrel=1e-11):
    """Zero-frequency cumulants of the weighted transferred charge.

    ``weights[c]`` is the amount credited to the counted observable when one
    electron moves from channel ``c`` into the dot. Charge counting at lead
    ``alpha`` uses ``1`` on that lead's channels and ``0`` elsewhere; an
    :math:`S_z` count uses ``+1/2`` and ``-1/2`` on its spin-up and spin-down
    channels.

    Returns ``result[m]`` for ``m = 0 .. order``: ``result[0]`` is zero by
    construction, ``result[1]`` is the current, and ``result[2]`` is the
    zero-frequency noise in the ``d Var/dt`` convention.
    """
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (model.nchannels,):
        raise ValueError("weights must have one entry per channel.")
    low, high = model.energy_window() if window is None else window
    levels = np.linalg.eigvalsh(np.asarray(model.hamiltonian, dtype=complex))
    breakpoints = sorted(
        float(point)
        for point in np.concatenate([np.asarray(model.mu, dtype=float), levels])
        if low < point < high
    )
    value, _ = quad_vec(
        lambda energy: _cumulant_integrand(
            model, energy, weights, order, radius, nnodes
        ),
        low, high, points=breakpoints or None,
        epsabs=epsabs, epsrel=epsrel, quadrature="gk21",
    )
    return value


def current_analytic(model, weights, window=None, epsabs=1e-13, epsrel=1e-11):
    """First cumulant in closed form, as a cross-check on :func:`cumulants`.

    Derived independently of the contour extraction:
    :math:`I=\\int\\frac{dE}{2\\pi}\\,\\mathrm{tr}[n(Q-S^\\dagger QS)]` with
    :math:`Q=\\mathrm{diag}(w_c)`, which reduces to the Landauer form
    :math:`\\sum_{c'}w_{c'}\\sum_c|S_{c'c}|^2(f_{c'}-f_c)`.
    """
    weights = np.asarray(weights, dtype=float)
    low, high = model.energy_window() if window is None else window

    def integrand(energy):
        scattering = model.scattering_matrix(energy)
        occupation = model.occupations(energy)
        probability = np.abs(scattering) ** 2
        return float(
            np.sum(
                weights[:, None]
                * probability
                * (occupation[:, None] - occupation[None, :])
            )
        ) / (2.0 * np.pi)

    value, _ = quad_vec(
        lambda energy: np.array([integrand(energy)]),
        low, high, epsabs=epsabs, epsrel=epsrel, quadrature="gk21",
    )
    return float(value[0])


def lead_weights(model, lead, observable="charge", spin_of_channel=None):
    """Counting weights for charge or ``S_z`` transfer at one lead."""
    weights = np.zeros(model.nchannels, dtype=float)
    for channel in range(model.nchannels):
        if model.lead_of_channel[channel] != lead:
            continue
        if observable == "charge":
            weights[channel] = 1.0
        elif observable == "spin":
            if spin_of_channel is None:
                raise ValueError("spin counting needs spin_of_channel.")
            weights[channel] = 0.5 if spin_of_channel[channel] == 0 else -0.5
        else:
            raise ValueError(f"Unknown observable {observable!r}.")
    return weights
