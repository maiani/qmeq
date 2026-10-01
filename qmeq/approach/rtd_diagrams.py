r"""Population diagrams of the RTD kernel, as immutable records.

RTD and RTDnoise evaluate the same two- and four-vertex diagrams with
population endpoints. RTD assembles the real stationary kernel ``Wdd``.
RTDnoise keeps each diagram's counting labels and Laplace derivative. This
module owns the enumeration that both share: the generators yield one record
per diagram, and each approach evaluates the records it receives.

Each generator yields its records in a fixed order, documented on it, and
the floating-point accumulation into the assembled arrays follows that order.
Two consumers that perform the same arithmetic per record therefore assemble
identical arrays, bit for bit, whatever else they compute.

Diagram notation follows [LeijnseWegewijs2008, Eqs. (61)-(65)], with QmeQ's
local vertex numbering ``0..3`` from right (earliest) to left (latest):
``p`` is a Keldysh index, ``eta`` an electron-hole index, ``r`` a lead. The
tunnelling amplitudes obey two rules:

1. ``t = Tba[r, f, i]`` if the vertex raises the charge from ``i`` to ``f``,
   else ``t = Tba[r, i, f].conj()``; and
2. ``t`` is conjugated once more when its vertex has ``p = -1``.
"""

from itertools import product
from typing import NamedTuple

import numpy as np

DIRECT = 'direct'
"""Topology of a four-vertex diagram contracting vertices (0, 3) and (1, 2)."""

EXCHANGE = 'exchange'
"""Topology of a four-vertex diagram contracting vertices (0, 2) and (1, 3)."""

GAIN = 'gain'
"""Two-vertex diagram that moves weight into the row population."""

LOSS = 'loss'
"""Two-vertex diagram that moves weight out of the row population."""


class FirstOrderDiagram(NamedTuple):
    """One two-vertex diagram of a row of the population kernel.

    The row population is ``|b><b|``. ``other`` is the state one charge away
    that the diagram passes through: the column state of a gain diagram, and
    the intermediate state of a loss diagram, whose column is ``b`` itself.
    ``lower`` records whether ``other`` has one electron fewer than ``b``.

    ``pair`` is the ``get_ind_dm1`` index of the transition between ``b`` and
    ``other``, which addresses the precomputed golden-rule factors
    ``paulifct``. ``energy`` is the transition energy ``E_upper - E_lower``.
    ``gamma`` is ``|T|**2`` as the counting kernel uses it, a complex number
    with zero imaginary part. It is formed from ``Tba[lead, other, b]``,
    except for an upper loss diagram, which uses ``Tba[lead, b, other]``;
    the two agree up to the roundoff of the basis rotation.
    """

    kind: str
    lead: int
    row: int
    column: int
    other: int
    lower: bool
    pair: int
    energy: float
    gamma: complex


class SecondOrderDiagram(NamedTuple):
    """One independent four-vertex population diagram.

    The enumeration fixes the outer Keldysh indices ``p0 = p3 = +1`` and the
    outer electron-hole index ``eta0 = +1``; in the paper's numbering these
    are ``p1 = p4 = eta1 = +1``. Two families of diagrams are implied by each
    record rather than enumerated:

    - **outer Keldysh flips.** Flipping ``p0``, ``p3`` or both moves the
      column from ``initial`` to ``initial_flipped`` and the row from
      ``final_state`` to ``flipped_state``. The four entries carry the signs
      ``+, -, +, -`` of ``KernelHandlerRTD.add_element_2nd_order``.
    - **the inverted partner.** Inverting every Keldysh and electron-hole
      index gives the ``eta0 = -1`` diagram. Its value at zero Laplace energy
      is the complex conjugate, its Laplace derivative the negative complex
      conjugate, and it keeps the record's counting labels
      [LeijnseWegewijs2008, Eqs. (B1)-(B3), (D1)-(D3)], [Emary2009, Eq. (31)].
      RTD adds it by taking twice the real part of every contribution;
      RTDnoise completes it after the traversal.

    ``r0`` contracts the outer vertex pair and ``r1`` the inner one in a
    direct diagram. In an exchange diagram ``r0`` contracts vertices 0 and 2,
    ``r1`` vertices 1 and 3. :attr:`lead` is the lead whose stationary kernel
    receives the diagram, the one contracted with the last vertex.

    ``tunnel_product`` is the four-amplitude product without the exchange
    sign; evaluators multiply an exchange diagram by ``-1``. ``E1``, ``E2``
    and ``E3`` are the propagator energies, and ``T1, mu1`` and ``T2, mu2``
    the temperatures and chemical potentials of ``r0`` and ``r1``. ``D`` is
    the full bandwidth of ``r0``. ``states`` are the many-body states
    ``(a0, a1, a2, a3)`` the diagram visits, from the initial population on.
    """

    topology: str
    r0: int
    r1: int
    eta1: int
    p1: int
    p2: int
    states: tuple
    tunnel_product: complex
    E1: float
    E2: float
    E3: float
    T1: float
    T2: float
    mu1: float
    mu2: float
    D: float
    initial: int
    initial_flipped: int
    final_state: int
    final_charge: int
    flipped_state: int
    flipped_charge: int

    @property
    def lead(self):
        """Lead whose stationary kernel receives the diagram."""
        return self.r0 if self.topology == DIRECT else self.r1


def counting_labels(topology, eta0, eta1, p1, p2):
    """Transferred charges of the four outer-Keldysh insertions of a diagram.

    Returns ``(q0, q1)`` for each of the entries ``(p0, p3) = (+, +), (+, -),
    (-, +), (-, -)``, in that order. ``q0`` is the charge counted at the
    contraction of ``r0`` and ``q1`` at that of ``r1``; each is the vertex
    factor ``eta*(p - p')/2`` of its contraction, with QmeQ's ``eta = -xi``
    against Emary's electron-hole index [Emary2009, Eq. (31)].
    """
    if topology == DIRECT:
        # eta0*(p0 - p3)/2 and eta1*(p1 - p2)/2.
        inner = eta1 * (p1 - p2)//2
        return (
            (eta0 * (1 - 1)//2, inner),
            (eta0 * (1 + 1)//2, inner),
            (eta0 * (-1 - 1)//2, inner),
            (eta0 * (-1 + 1)//2, inner),
        )
    # eta0*(p0 - p2)/2 and eta1*(p1 - p3)/2.
    return (
        (eta0 * (1 - p2)//2, eta1 * (p1 - 1)//2),
        (eta0 * (1 - p2)//2, eta1 * (p1 + 1)//2),
        (eta0 * (-1 - p2)//2, eta1 * (p1 - 1)//2),
        (eta0 * (-1 - p2)//2, eta1 * (p1 + 1)//2),
    )


def first_order_diagrams(appr, b, bcharge):
    """Yield the two-vertex diagrams of the population-kernel row of ``b``.

    Gain diagrams come first, from the lower then the upper charge sector,
    followed by the loss diagrams in the same order; within each block the
    lead index runs fastest.
    """
    si = appr.si
    nleads, statesdm = si.nleads, si.statesdm
    E, Tba = appr.qd.Ea, appr.leads.Tba

    acharge = bcharge-1
    ccharge = bcharge+1
    bb = si.get_ind_dm0(b, b, bcharge)

    for a in statesdm[acharge]:
        aa = si.get_ind_dm0(a, a, acharge)
        ba = si.get_ind_dm1(b, a, acharge)
        dE = E[b] - E[a]
        for l in range(nleads):
            gamma = Tba[l, a, b] * Tba[l, a, b].conj()
            yield FirstOrderDiagram(GAIN, l, bb, aa, a, True, ba, dE, gamma)
    for c in statesdm[ccharge]:
        cc = si.get_ind_dm0(c, c, ccharge)
        cb = si.get_ind_dm1(c, b, bcharge)
        dE = E[c] - E[b]
        for l in range(nleads):
            gamma = Tba[l, c, b] * Tba[l, c, b].conj()
            yield FirstOrderDiagram(GAIN, l, bb, cc, c, False, cb, dE, gamma)
    for bm in statesdm[acharge]:
        ba = si.get_ind_dm1(b, bm, acharge)
        dE = E[b] - E[bm]
        for l in range(nleads):
            gamma = Tba[l, bm, b] * Tba[l, bm, b].conj()
            yield FirstOrderDiagram(LOSS, l, bb, bb, bm, True, ba, dE, gamma)
    for bp in statesdm[ccharge]:
        cb = si.get_ind_dm1(bp, b, bcharge)
        dE = E[bp] - E[b]
        for l in range(nleads):
            gamma = Tba[l, b, bp] * Tba[l, b, bp].conj()
            yield FirstOrderDiagram(LOSS, l, bb, bb, bp, False, cb, dE, gamma)


def second_order_diagrams(appr, a0, charge):
    """Yield the independent four-vertex diagrams with initial population ``a0``.

    Amplitude products below the cutoffs are skipped: an exact zero for the
    first vertex, ``1e-10*max(T)`` for the first two and ``1e-20*max(T)**2``
    for all four, with ``T`` the lead temperatures. The direct diagram of an
    intermediate configuration precedes its exchange partner.
    """
    si = appr.si
    statesdm, Tba, E = si.statesdm, appr.leads.Tba, appr.qd.Ea
    tlst, mulst, dlst = appr.leads.tlst, appr.leads.mulst, appr.leads.dlst
    nleads = si.nleads

    t_cutoff1 = 0.0
    t_cutoff2 = 1e-10*max(tlst)
    t_cutoff3 = 1e-20*max(tlst)**2
    indx0 = si.get_ind_dm0(a0, a0, charge)

    for r0, r1 in product(range(nleads), range(nleads)):
        T1, T2 = tlst[r0], tlst[r1]
        mu1, mu2 = mulst[r0], mulst[r1]
        D = np.abs(dlst[r0, 1]) + np.abs(dlst[r0, 0])
        leads = (T1, T2, mu1, mu2, D)
        # N1 = (N0, N0 + 1), a1- = a0
        for a1p in statesdm[charge+1]:
            t = Tba[r0, a1p, a0]
            if abs(t) == t_cutoff1:
                continue
            indx1 = si.get_ind_dm0(a1p, a1p, charge + 1)
            E1 = E[a1p] - E[a0]
            # eta1 = 1, p1 = 1: N2 = (N0, N0 + 2), a2- = a0
            for a2p in statesdm[charge+2]:
                t1 = t * Tba[r1, a2p, a1p]
                if abs(t1) < t_cutoff2:
                    continue
                E2 = E[a2p] - E[a0]
                # p2 = 1: N3 = (N0, N0 + 1), a3- = a2-
                for a3p in statesdm[charge+1]:
                    t2D = t1 * Tba[r1, a2p, a3p].conj() * Tba[r0, a3p, a0].conj()
                    t2X = t1 * Tba[r0, a2p, a3p].conj() * Tba[r1, a3p, a0].conj()
                    E3 = E[a3p] - E[a0]
                    states = (a0, a1p, a2p, a3p)
                    ends = (indx0, indx1, a0, charge, a3p, charge + 1)
                    if abs(t2D) > t_cutoff3:
                        yield SecondOrderDiagram(DIRECT, r0, r1, 1, 1, 1, states, t2D, E1, E2, E3, *leads, *ends)
                    if abs(t2X) > t_cutoff3:
                        yield SecondOrderDiagram(EXCHANGE, r0, r1, 1, 1, 1, states, t2X, E1, E2, E3, *leads, *ends)
                # p2 = -1: N3 = (N0 + 1, N0 + 2), a3+ = a2+
                for a3m in statesdm[charge+1]:
                    t2D = t1 * Tba[r1, a3m, a0].conj() * Tba[r0, a2p, a3m].conj()
                    t2X = t1 * Tba[r0, a3m, a0].conj() * Tba[r1, a2p, a3m].conj()
                    E3 = E[a2p] - E[a3m]
                    states = (a0, a1p, a2p, a3m)
                    ends = (indx0, indx1, a3m, charge + 1, a2p, charge + 2)
                    if abs(t2D) > t_cutoff3:
                        yield SecondOrderDiagram(DIRECT, r0, r1, 1, 1, -1, states, t2D, E1, E2, E3, *leads, *ends)
                    if abs(t2X) > t_cutoff3:
                        yield SecondOrderDiagram(EXCHANGE, r0, r1, 1, 1, -1, states, t2X, E1, E2, E3, *leads, *ends)
            # eta1 = 1, p1 = -1: N2 = (N0 - 1, N0 + 1), a2+ = a1+
            for a2m in statesdm[charge-1]:
                t1 = t * Tba[r1, a0, a2m]
                if abs(t1) < t_cutoff2:
                    continue
                E2 = E[a1p] - E[a2m]
                # p2 = 1: N3 = (N0 - 1, N0), a3- = a2-
                for a3p in statesdm[charge]:
                    t2D = t1 * Tba[r1, a1p, a3p].conj() * Tba[r0, a3p, a2m].conj()
                    t2X = t1 * Tba[r0, a1p, a3p].conj() * Tba[r1, a3p, a2m].conj()
                    E3 = E[a3p] - E[a2m]
                    states = (a0, a1p, a2m, a3p)
                    ends = (indx0, indx1, a2m, charge - 1, a3p, charge)
                    if abs(t2D) > t_cutoff3:
                        yield SecondOrderDiagram(DIRECT, r0, r1, 1, -1, 1, states, t2D, E1, E2, E3, *leads, *ends)
                    if abs(t2X) > t_cutoff3:
                        yield SecondOrderDiagram(EXCHANGE, r0, r1, 1, -1, 1, states, t2X, E1, E2, E3, *leads, *ends)
                # p2 = -1: N3 = (N0, N0 + 1), a3+ = a2+
                for a3m in statesdm[charge]:
                    t2D = t1 * Tba[r1, a3m, a2m].conj() * Tba[r0, a1p, a3m].conj()
                    t2X = t1 * Tba[r0, a3m, a2m].conj() * Tba[r1, a1p, a3m].conj()
                    E3 = E[a1p] - E[a3m]
                    states = (a0, a1p, a2m, a3m)
                    ends = (indx0, indx1, a3m, charge, a1p, charge + 1)
                    if abs(t2D) > t_cutoff3:
                        yield SecondOrderDiagram(DIRECT, r0, r1, 1, -1, -1, states, t2D, E1, E2, E3, *leads, *ends)
                    if abs(t2X) > t_cutoff3:
                        yield SecondOrderDiagram(EXCHANGE, r0, r1, 1, -1, -1, states, t2X, E1, E2, E3, *leads, *ends)
            # eta1 = -1, p1 = 1: N2 = (N0, N0), a2- = a0
            for a2p in statesdm[charge]:
                E2 = E[a2p] - E[a0]
                t1 = t * Tba[r1, a1p, a2p].conj()
                if abs(t1) < t_cutoff2:
                    continue
                # p2 = 1: N3 = (N0, N0 + 1), a3- = a0
                for a3p in statesdm[charge+1]:
                    t2D = t1 * Tba[r1, a3p, a2p] * Tba[r0, a3p, a0].conj()
                    E3 = E[a3p] - E[a0]
                    if abs(t2D) > t_cutoff3:
                        yield SecondOrderDiagram(
                            DIRECT, r0, r1, -1, 1, 1, (a0, a1p, a2p, a3p), t2D, E1, E2, E3,
                            *leads, indx0, indx1, a0, charge, a3p, charge + 1)
                # p2 = 1: N3 = (N0, N0 - 1), a3- = a0
                for a3p in statesdm[charge-1]:
                    t2X = t1 * Tba[r0, a2p, a3p].conj() * Tba[r1, a0, a3p]
                    E3 = E[a3p] - E[a0]
                    if abs(t2X) > t_cutoff3:
                        yield SecondOrderDiagram(
                            EXCHANGE, r0, r1, -1, 1, 1, (a0, a1p, a2p, a3p), t2X, E1, E2, E3,
                            *leads, indx0, indx1, a0, charge, a3p, charge - 1)
                # p2 = -1: N3 = (N0 - 1, N0), a3+ = a2+
                for a3m in statesdm[charge-1]:
                    t2D = t1 * Tba[r1, a0, a3m] * Tba[r0, a2p, a3m].conj()
                    E3 = E[a2p] - E[a3m]
                    if abs(t2D) > t_cutoff3:
                        yield SecondOrderDiagram(
                            DIRECT, r0, r1, -1, 1, -1, (a0, a1p, a2p, a3m), t2D, E1, E2, E3,
                            *leads, indx0, indx1, a3m, charge - 1, a2p, charge)
                # p2 = -1: N3 = (N0 + 1, N0)
                for a3m in statesdm[charge+1]:
                    t2X = t1 * Tba[r0, a3m, a0].conj() * Tba[r1, a3m, a2p]
                    E3 = E[a2p] - E[a3m]
                    if abs(t2X) > t_cutoff3:
                        yield SecondOrderDiagram(
                            EXCHANGE, r0, r1, -1, 1, -1, (a0, a1p, a2p, a3m), t2X, E1, E2, E3,
                            *leads, indx0, indx1, a3m, charge + 1, a2p, charge)
            # eta1 = -1, p1 = -1: N2 = (N0 + 1, N0 + 1), a2+ = a1+
            for a2m in statesdm[charge+1]:
                E2 = E[a1p] - E[a2m]
                t1 = t * Tba[r1, a2m, a0].conj()
                # p2 = 1: N3 = (N0 + 1, N0 + 2), a3- = a2-
                for a3p in statesdm[charge+2]:
                    t2D = t1 * Tba[r1, a3p, a1p] * Tba[r0, a3p, a2m].conj()
                    E3 = E[a3p] - E[a2m]
                    if abs(t2D) > t_cutoff3:
                        yield SecondOrderDiagram(
                            DIRECT, r0, r1, -1, -1, 1, (a0, a1p, a2m, a3p), t2D, E1, E2, E3,
                            *leads, indx0, indx1, a2m, charge + 1, a3p, charge + 2)
                # p2 = 1: N3 = (N0 + 1, N0)
                for a3p in statesdm[charge]:
                    t2X = t1 * Tba[r0, a1p, a3p].conj() * Tba[r1, a2m, a3p]
                    E3 = E[a3p] - E[a2m]
                    if abs(t2X) > t_cutoff3:
                        yield SecondOrderDiagram(
                            EXCHANGE, r0, r1, -1, -1, 1, (a0, a1p, a2m, a3p), t2X, E1, E2, E3,
                            *leads, indx0, indx1, a2m, charge + 1, a3p, charge)
                # p2 = -1: N3 = (N0, N0 + 1), a3+ = a2+
                for a3m in statesdm[charge]:
                    t2D = t1 * Tba[r1, a2m, a3m] * Tba[r0, a1p, a3m].conj()
                    E3 = E[a1p] - E[a3m]
                    if abs(t2D) > t_cutoff3:
                        yield SecondOrderDiagram(
                            DIRECT, r0, r1, -1, -1, -1, (a0, a1p, a2m, a3m), t2D, E1, E2, E3,
                            *leads, indx0, indx1, a3m, charge, a1p, charge + 1)
                # p2 = -1: N3 = (N0 + 2, N0 + 1), a3+ = a2+
                for a3m in statesdm[charge+2]:
                    t2X = t1 * Tba[r0, a3m, a2m].conj() * Tba[r1, a3m, a1p]
                    E3 = E[a1p] - E[a3m]
                    if abs(t2X) > t_cutoff3:
                        yield SecondOrderDiagram(
                            EXCHANGE, r0, r1, -1, -1, -1, (a0, a1p, a2m, a3m), t2X, E1, E2, E3,
                            *leads, indx0, indx1, a3m, charge + 2, a1p, charge + 1)
