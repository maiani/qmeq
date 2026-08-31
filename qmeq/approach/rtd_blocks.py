"""Reusable population-coherence blocks for real-time diagrammatics.

The RTD population solver stores Hermitian coherences as separate real and
imaginary coordinates.  This module is the boundary between that packed
storage and block algebra that can also be reused by counting-statistics and
coherence-retaining approaches.

For a block element taking an initial density-matrix coordinate into a final
one, the counting transfer is the final dot charge minus the initial dot
charge.  Thus a positive transfer denotes an electron entering the dot.  At
zero counting field the effective population correction is

``Re(Wdn) G Im(Wnd) + Im(Wdn) G Re(Wnd)``,

where ``G`` is the bare coherence propagator.  Equivalently, it is the real
part of ``-1j * Wdn G Wnd`` when the two stored channels are combined as
``Re + 1j*Im``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ..specfunc.specfunc import diff_fermi
from ..specfunc.specfunc import diff_phi
from ..specfunc.specfunc import fermi_func
from ..specfunc.specfunc import phi
from ..wrappers.mytypes import complexnp, doublenp

if TYPE_CHECKING:
    from .base.RTD import ApproachPyRTD
    from ..indexing import StateIndexingDM, StateIndexingDMc


@dataclass(frozen=True)
class CoherenceCorrection:
    """Counting-resolved effective population kernel and its Laplace derivative."""

    kernel: np.ndarray
    laplace_derivative: np.ndarray


@dataclass(frozen=True)
class FirstOrderTerm:
    """One Keldysh contribution to a first-order population/coherence block.

    A first-order block element is a sum of contributions

    ``amplitude * tunnel_product * (pi*f(u) + eta*1j*phi(u))``,

    where ``u = (E - mu)/T`` is the scaled energy of the *intermediate*
    propagator segment.  The bracket is one analytic function of ``u``
    ([LeijnseWegewijs2008, Eq. (C4)] for ``phi``; see :func:`phi`), because

    ``pi*f(u) = pi/2 - Im psi(1/2 + 1j*u/(2*pi))``

    while ``phi(u) = -Re psi(1/2 + 1j*u/(2*pi)) + log(D/(2*pi))``, so

    ``pi*f(u) + 1j*phi(u) = pi/2 - 1j*psi(1/2 - 1j*u/(2*pi)) + 1j*log(D/(2*pi))``.

    ``eta`` is the sign of the ``1j*phi`` channel relative to ``amplitude``,
    which is the coefficient of ``pi*f``.  It also fixes how the Laplace energy
    enters: ``z`` shifts the intermediate propagator energy uniformly, and the
    two Keldysh partners of a pair carry that segment with opposite orientation,
    so in the *coded* scaled arguments the shift appears as
    ``u -> u + eta*z/T``.

    Attributes
    ----------
    amplitude : int
        Sign multiplying ``tunnel_product`` for this contribution.
    eta : int
        ``+1`` or ``-1``; the relative sign of the ``1j*phi`` channel.
    scaled_energy : float
        ``u`` at zero Laplace energy.
    """

    amplitude: int
    eta: int
    scaled_energy: float


def first_order_block_value(
        tunnel_product: complex, temperature: float,
        band_plus: float, band_minus: float,
        terms: tuple[FirstOrderTerm, ...],
        laplace: float = 0.0) -> complex:
    """Evaluate one first-order block element at Laplace energy ``laplace``.

    At ``laplace == 0`` this must reproduce the historical ``temp1 + 1j*temp2``
    expressions in :mod:`qmeq.approach.base.RTD` exactly; that equality is what
    pins the term decomposition, and it is asserted by
    ``test_finite_laplace_energy_reproduces_the_stored_blocks``.  Nonzero
    ``laplace`` is the independent finite-``z`` reference used to gate the
    analytic derivative and the counting-resolved composition.
    """
    total = 0.0 + 0.0j
    for term in terms:
        u = term.scaled_energy + term.eta*laplace/temperature
        total += term.amplitude*(
            np.pi*fermi_func(u) + term.eta*1j*phi(u, band_plus, band_minus)
        )
    return tunnel_product*total


def first_order_block_derivative(
        tunnel_product: complex, temperature: float,
        terms: tuple[FirstOrderTerm, ...]) -> complex:
    """Analytic ``d/dz`` of :func:`first_order_block_value` at ``z = 0``.

    Differentiating ``amplitude*(pi*f + eta*1j*phi)`` under ``u -> u + eta*z/T``
    gives ``(amplitude/T)*(eta*pi*f'(u) + 1j*phi'(u))``: the ``1j*phi'`` channel
    is governed by ``amplitude`` alone, while the real ``pi*f'`` channel carries
    ``eta``.  The latter cancels only when the two partners of a pair share
    ``u`` -- the diagonal limit -- which is why the *diagonal* first-order
    Laplace derivative is purely imaginary and the off-diagonal blocks' is not.
    The ``log(D/(2*pi))`` term of ``phi`` is ``z``-independent and drops out.
    """
    total = 0.0 + 0.0j
    for term in terms:
        total += term.amplitude*(
            term.eta*np.pi*diff_fermi(term.scaled_energy)
            + 1j*diff_phi(term.scaled_energy)
        )
    return tunnel_product*total/temperature


def generate_population_coherence_blocks(approach: ApproachPyRTD) -> None:
    """Fill the first-order ``nd``/``dn`` blocks and bare propagator.

    The approach supplies the diagram formulas; this function owns the common
    traversal so population RTD, RTD noise, and future coherent consumers do
    not independently reimplement which density-matrix coordinates are
    visited.
    """
    si = approach.si
    statesdm = si.statesdm
    handler = approach.kernel_handler

    for charge in range(si.ncharge):
        for state in statesdm[charge]:
            if handler.is_unique(state, state, charge):
                approach.generate_col_nondiag_kern_1st_order_nd(state, charge)

    for charge in range(si.ncharge):
        for state in statesdm[charge]:
            for partner in statesdm[charge]:
                if state == partner:
                    continue
                approach.generate_col_nondiag_kern_1st_order_dn(
                    state, partner, charge
                )
                approach.generate_row_inverse_Liouvillian(
                    state, partner, charge
                )


def zero_field_coherence_correction(approach: ApproachPyRTD) -> np.ndarray:
    """Return the existing lead-resolved RTD population correction."""
    dn = approach.ReWdn + 1j * approach.ImWdn
    nd = approach.ReWnd + 1j * approach.ImWnd
    propagator = approach.Lnn_inv
    correction = np.empty(approach.Wdd.shape, dtype=doublenp)
    nd_total = np.sum(nd, axis=0)
    for lead in range(approach.si.nleads):
        correction[lead] = (-1j * dn[lead] @ propagator @ nd_total).real
    return correction


def counting_resolved_coherence_correction(
        approach: ApproachPyRTD) -> CoherenceCorrection:
    """Compose the transfer-resolved population correction.

    The returned arrays have shape
    ``(lead_dn, lead_nd, transfer_dn, transfer_nd, population, population)``.
    Transfer axes use the historical RTDnoise convention in which the Python
    indices ``-1``, ``0`` and ``1`` denote the corresponding signed charge.

    The Laplace derivative follows the product rule.  The free molecular line
    is ``Pi0(z_LW) = 1j*(z_LW - L)^-1``
    [LeijnseWegewijs2008, Eq. (49)].  On the ordered coherence
    ``|a><b|``, ``L`` has eigenvalue ``dE = E[a] - E[b]``.  QmeQ's energy-like
    continuation is ``z_LW = -z``; after extracting the line's ``-1j`` into
    ``W_corr = -1j*Wdn*G*Wnd``, the stored resolvent is therefore
    ``G(z) = (dE + z)^-1`` and ``G'(0) = -G(0)^2``.  The RTD coherence axis
    stores the two ordered partners separately, so conjugation changes ``dE``
    to ``-dE`` without changing the common ``+z`` orientation.  See
    ``docs/docs/conventions/rtd-kernels.md`` for the packed-coordinate mapping.

    The projection is not a choice.  ``product`` is purely imaginary and
    ``product_dz`` purely real, so ``W_corr(z) = -1j*Wdn(z) G(z) Wnd(z)`` keeps
    every nonzero channel of both while matching the counting path's convention
    of real kernels and purely imaginary ``_dz`` arrays, on which the reality of
    the noise depends ([Emary2009, Eqs. (40)-(41)]).  Taking ``.imag`` of the
    derivative instead keeps the identically zero channel.
    """
    si = approach.si
    nleads = si.nleads
    npauli = approach.get_kern_size()
    ncoherences = approach.Lnn_inv.shape[0]
    shape = (nleads, nleads, 3, 3, npauli, npauli)
    correction = np.zeros(shape, dtype=complexnp)
    derivative = np.zeros(shape, dtype=complexnp)

    population_charge, coherence_charge = _coordinate_charges(si)
    dn = approach.ReWdn + 1j * approach.ImWdn
    nd = approach.ReWnd + 1j * approach.ImWnd
    dn_dz = approach.ReWdn_dz + 1j * approach.ImWdn_dz
    nd_dz = approach.ReWnd_dz + 1j * approach.ImWnd_dz
    propagator = approach.Lnn_inv
    propagator_dz = -(propagator @ propagator)

    dn_resolved = np.zeros((nleads, 3, npauli, ncoherences), dtype=complexnp)
    nd_resolved = np.zeros((nleads, 3, ncoherences, npauli), dtype=complexnp)
    dn_dz_resolved = np.zeros_like(dn_resolved)
    nd_dz_resolved = np.zeros_like(nd_resolved)

    for population in range(npauli):
        for coherence in range(ncoherences):
            transfer = population_charge[population] - coherence_charge[coherence]
            forward = (
                dn[:, population, coherence], dn_dz[:, population, coherence],
            )
            backward = (
                nd[:, coherence, population], nd_dz[:, coherence, population],
            )
            if abs(transfer) > 1:
                # Charge sectors more than one electron apart are simply not
                # connected by a first-order vertex, so the block is
                # structurally zero and carries no transfer to resolve.  Any
                # nonzero entry here would be a genuine violation.
                _reject_multi_electron_transfer(
                    transfer, *forward, *backward
                )
                continue
            dn_resolved[:, transfer, population, coherence] = forward[0]
            dn_dz_resolved[:, transfer, population, coherence] = forward[1]

            reverse_transfer = -transfer
            nd_resolved[:, reverse_transfer, coherence, population] = backward[0]
            nd_dz_resolved[:, reverse_transfer, coherence, population] = (
                backward[1]
            )

    for lead_dn in range(nleads):
        for lead_nd in range(nleads):
            for transfer_dn in (-1, 0, 1):
                left = dn_resolved[lead_dn, transfer_dn]
                left_dz = dn_dz_resolved[lead_dn, transfer_dn]
                for transfer_nd in (-1, 0, 1):
                    right = nd_resolved[lead_nd, transfer_nd]
                    right_dz = nd_dz_resolved[lead_nd, transfer_nd]
                    product = left @ propagator @ right
                    product_dz = (
                        left_dz @ propagator @ right
                        + left @ propagator_dz @ right
                        + left @ propagator @ right_dz
                    )
                    correction[
                        lead_dn, lead_nd, transfer_dn, transfer_nd
                    ] = (-1j*product).real
                    derivative[
                        lead_dn, lead_nd, transfer_dn, transfer_nd
                    ] = -1j*product_dz

    return CoherenceCorrection(correction, derivative)


def _coordinate_charges(
        si: StateIndexingDM | StateIndexingDMc,
) -> tuple[np.ndarray, np.ndarray]:
    """Return dot charges for packed population and coherence coordinates."""
    population_charge = np.empty(si.npauli, dtype=int)
    coherence_charge = np.empty(si.ndm0r - si.npauli, dtype=int)
    seen_coherences = np.zeros(coherence_charge.size, dtype=bool)

    for charge in range(si.ncharge):
        for state in si.statesdm[charge]:
            population = si.get_ind_dm0(state, state, charge)
            population_charge[population] = charge
        for state in si.statesdm[charge]:
            for partner in si.statesdm[charge]:
                if state == partner:
                    continue
                index = si.get_ind_dm0(state, partner, charge)
                if index < si.npauli:
                    continue
                real_index = index - si.npauli
                coherence_charge[real_index] = charge
                seen_coherences[real_index] = True
                imag_index = real_index + si.ndm0 - si.npauli
                coherence_charge[imag_index] = charge
                seen_coherences[imag_index] = True

    if not np.all(seen_coherences):
        raise RuntimeError("Could not assign a charge to every coherence coordinate.")
    return population_charge, coherence_charge


def _reject_multi_electron_transfer(
        transfer: int, *blocks: np.ndarray) -> None:
    """Raise only if a multi-electron coordinate pair carries an amplitude."""
    if not any(np.any(block) for block in blocks):
        return
    raise ValueError(
        "A first-order population-coherence block changed the dot charge "
        f"by {transfer}; only single-electron transfers are supported."
    )
