"""Module containing RTD Approach."""

from itertools import product
from dataclasses import dataclass

import numpy as np
import itertools
import warnings

from ...wrappers.mytypes import doublenp
from ..._warnings import QmeqRuntimeWarning

from ..kernel_handler import RtdMatrix

from ...specfunc.specfunc import integralD
from ...specfunc.specfunc import integralX
from ...specfunc.specfunc import phi
from ...specfunc.specfunc import fermi_func
from ...specfunc.specfunc import delta_phi
from ...specfunc.specfunc import BW_Ozaki
from ...specfunc.specfunc import func_pauli
from ..aprclass import Approach
from ..kernel_handler import KernelHandlerRTD

_IMAGINARY_TUNNEL_PRODUCT_RTOL = 1e-12
_RTD_WIDEBAND_RATIO_WARNING = 1e3
_RTD_COHERENCE_RATIO_WARNING = 10.0


def _has_significant_imaginary_part(value):
    """Return whether a tunnel product has a physically relevant phase."""
    return abs(value.imag) > _IMAGINARY_TUNNEL_PRODUCT_RTOL*abs(value)


class RTDBandwidthWarning(QmeqRuntimeWarning):
    """Warning that an unequal-temperature RTD cutoff may be too small."""


class RTDCoherenceWarning(QmeqRuntimeWarning):
    """Warning that RTD's diagonal-density-matrix approximation is marginal."""


class RTDNoBroadeningWarning(QmeqRuntimeWarning):
    """Warning that RTD has no tunnel broadening for its active states."""


@dataclass(frozen=True)
class RTDCoherenceDiagnostics:
    """Validity diagnostic for RTD's eliminated same-charge coherences.

    ``gamma`` is a conservative golden-rule estimate: ``2*pi`` times the
    sum of squared many-body tunnel amplitudes out of each state, averaged
    over the two states in the closest same-charge pair.  It omits Fermi
    factors, so it is an upper-scale estimate of the coherence broadening.
    The diagonal approximation is flagged when the closest splitting is less
    than ten times this scale.

    ``charge``, ``state_a`` and ``state_b`` identify the closest pair, and
    ``clamped_coherences`` counts same-charge splittings below the inverse
    Liouvillian clamp used by RTD.
    """

    minimum_splitting: float
    gamma: float
    ratio: float
    threshold: float
    charge: int
    state_a: int
    state_b: int
    clamped_coherences: int


def _rtd_bandwidth_ratio(qd, leads):
    """Return the smallest lead cutoff divided by the largest transport scale."""
    temperatures = np.asarray(leads.tlst, dtype=float)
    chemical_potentials = np.asarray(leads.mulst, dtype=float)
    bandwidths = np.sum(np.abs(np.asarray(leads.dlst, dtype=float)), axis=1)
    scale = float(np.max(temperatures))
    energies = np.asarray(qd.Ea, dtype=float)
    statesdm = qd.si.statesdm
    for charge in range(qd.si.ncharge - 1):
        for lower in statesdm[charge]:
            for upper in statesdm[charge + 1]:
                transition = energies[upper] - energies[lower]
                scale = max(
                    scale,
                    float(np.max(np.abs(transition - chemical_potentials))),
                )
    return float(np.min(bandwidths) / max(scale, np.finfo(float).tiny))


def _warn_if_unequal_temperature_cutoff_is_small(qd, leads):
    """Warn when the finite RTD regulator is not safely in the wide-band regime."""
    temperatures = np.asarray(leads.tlst, dtype=float)
    unequal = not np.allclose(
        temperatures, temperatures[0], rtol=1e-12, atol=0.0
    )
    if not unequal:
        return
    ratio = _rtd_bandwidth_ratio(qd, leads)
    if ratio < _RTD_WIDEBAND_RATIO_WARNING:
        warnings.warn(
            "Unequal-temperature RTD uses the finite cutoff in dband as a "
            "numerical wide-band regulator. The current cutoff-to-transport-"
            f"scale ratio is {ratio:.3g}, below the conservative diagnostic "
            f"threshold {_RTD_WIDEBAND_RATIO_WARNING:.0f}. Repeat the "
            "calculation with larger dband values and require convergence of "
            "every reported observable.",
            RTDBandwidthWarning,
            stacklevel=3,
        )


def _warn_if_rtd_coherence_is_not_resolved(appr):
    """Warn when RTD's population-only treatment is outside its regime.

    The legacy RTD kernel eliminates same-charge coherences with a bare
    inverse energy difference.  For each many-body state we estimate its
    tunnel width from all adjacent-charge transitions, using
    ``Gamma = 2*pi*sum(|T|**2)``.  The coherence width is the mean of the two
    state widths.  A ten-to-one splitting-to-width margin is used as a
    conservative validity screen: the warning is diagnostic only and does
    not change the kernel.
    """
    si, energies, Tba = appr.si, np.asarray(appr.qd.Ea), np.asarray(appr.leads.Tba)
    widths = np.zeros(si.nmany, dtype=float)
    for charge, states in enumerate(si.statesdm):
        neighbours = []
        if charge:
            neighbours.extend(si.statesdm[charge - 1])
        if charge + 1 < si.ncharge:
            neighbours.extend(si.statesdm[charge + 1])
        for state in states:
            widths[state] = 2.0*np.pi*sum(
                np.sum(np.abs(Tba[:, state, other])**2) for other in neighbours
            )

    closest = None
    clamped_coherences = 0
    for states in si.statesdm:
        for i, state in enumerate(states):
            for other in states[i + 1:]:
                splitting = abs(float(energies[state] - energies[other]))
                if splitting < 1e-10:
                    clamped_coherences += 1
                gamma = 0.5*(widths[state] + widths[other])
                ratio = splitting/gamma if gamma > 0.0 else np.inf
                if closest is None or ratio < closest.ratio:
                    closest = RTDCoherenceDiagnostics(
                        splitting, gamma, ratio, _RTD_COHERENCE_RATIO_WARNING,
                        -1, state, other, 0,
                    )

    if closest is not None:
        # Recover the sector without relying on the global state ordering.
        for charge, states in enumerate(si.statesdm):
            if closest.state_a in states:
                closest = RTDCoherenceDiagnostics(
                    closest.minimum_splitting, closest.gamma, closest.ratio,
                    closest.threshold, charge, closest.state_a,
                    closest.state_b, clamped_coherences,
                )
                break
    appr.rtd_coherence_diagnostics = closest
    if closest is not None and closest.gamma == 0.0:
        if not getattr(appr, 'printed_warning_no_broadening', False):
            warnings.warn(
                "RTD found no tunnel broadening for the same-charge states "
                "used in its population kernel. The stationary kernel may be "
                "singular; check that tunnel amplitudes connect the intended "
                "states and leads.",
                RTDNoBroadeningWarning,
                stacklevel=3,
            )
            appr.printed_warning_no_broadening = True
        return
    if (closest is not None and closest.ratio < _RTD_COHERENCE_RATIO_WARNING
            and not getattr(appr, 'printed_warning_coherence', False)):
        warnings.warn(
            "RTD eliminates same-charge coherences although the closest "
            "intra-sector splitting is %.3g for states (%d, %d) in charge "
            "%d, the estimated coherence "
            "broadening is %.3g (ratio %.3g; required %.3g). The diagonal "
            "density-matrix approximation is not controlled at this point; "
            "use a coherence-retaining method or verify convergence away "
            "from the degeneracy." % (
                closest.minimum_splitting, closest.state_a, closest.state_b,
                closest.charge, closest.gamma, closest.ratio, closest.threshold),
            RTDCoherenceWarning,
            stacklevel=3,
        )
        appr.printed_warning_coherence = True


class ApproachPyRTD(Approach):

    kerntype = 'pyRTD'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.BW_Ozaki_expansion = 0
        self.Ozaki_poles_and_residues = None
        self.ImGamma = False
        self.printed_warning_ImGamma = False
        self.nsingle_warning_printed = False

    def get_kern_size(self):
        return self.si.npauli

    def restart(self):
        Approach.restart(self)
        self.rtd_coherence_diagnostics = None
        self.Wdd = None
        self.WE1 = None
        self.WE2 = None
        self.ReWnd = None
        self.ImWnd = None
        self.ReWdn = None
        self.ImWdn = None
        self.Lnn_inv = None

    def prepare_kernel_handler(self):
        self.kernel_handler = KernelHandlerRTD(self.si)

    def prepare_arrays(self):
        Approach.prepare_arrays(self)

        nleads, ndm1 = self.si.nleads, self.si.ndm1
        self.paulifct = np.zeros((nleads, ndm1, 2), dtype=doublenp)

        kern_size = self.get_kern_size()
        nleads = self.si.nleads
        self.Wdd = np.zeros((nleads, kern_size, kern_size), dtype=self.dtype)
        self.WE1 = np.zeros((nleads, kern_size, kern_size), dtype=self.dtype)
        self.WE2 = np.zeros((nleads, kern_size, kern_size), dtype=self.dtype)

        kh = self.kernel_handler
        kh.Wdd = self.Wdd
        kh.WE1 = self.WE1
        kh.WE2 = self.WE2

        self.W2 = np.zeros((nleads, kern_size, kern_size), dtype=self.dtype)

        self.generate_LN()
        self.LE = np.zeros((kern_size, kern_size), dtype=self.dtype)

        if self.funcp.off_diag_corrections:
            kern_size2 = 2 * self.si.ndm0 - 2 * self.si.npauli
            self.ReWnd = np.zeros((nleads, kern_size2, kern_size), dtype=self.dtype)
            self.ImWnd = np.zeros((nleads, kern_size2, kern_size), dtype=self.dtype)
            self.ReWdn = np.zeros((nleads, kern_size, kern_size2), dtype=self.dtype)
            self.ImWdn = np.zeros((nleads, kern_size, kern_size2), dtype=self.dtype)
            self.Lnn_inv = np.zeros((kern_size2, kern_size2), dtype=self.dtype)

            kh.ReWnd = self.ReWnd
            kh.ImWnd = self.ImWnd
            kh.ReWdn = self.ReWdn
            kh.ImWdn = self.ImWdn
            kh.Lnn_inv = self.Lnn_inv

        kh.set_matrix_list()

    def clean_arrays(self):
        Approach.clean_arrays(self)
        self.ImGamma = False
        self.paulifct.fill(0.0)

        self.Wdd.fill(0.0)
        self.WE1.fill(0.0)
        self.WE2.fill(0.0)
        self.W2.fill(0.0)
        self.LE.fill(0.0)

        if self.funcp.off_diag_corrections:
            self.ReWnd.fill(0.0)
            self.ImWnd.fill(0.0)
            self.ReWdn.fill(0.0)
            self.ImWdn.fill(0.0)
            self.Lnn_inv.fill(0.0)

    def generate_kern(self):
        r""" Generates all kernels including tunnel processes of orders :math:`t^2` and :math:`t^4`.

        The total kernel used to solve for :math:`\phi_0` is :math:`W =\sum_r W^r= W_{dd}^{(1)} + W_{dd}^{(2)}
        + W_{dn}^{(1)} (L_{nn})^{-1} W_{nd}^{(1)}`. The last term is ignored if `off_diag_corrections` is False.

        Parameters
        ----------
        self.kern : ndarray
            (Modifies) The total Kernel for the diagonal density matrix. Has npauli * npauli entries.
        self.Wdd :  ndarray
            (Modifies) The lead-resolved Kernel for the diagonal density matrix. Has
            nleads * npauli * npauli entries.

        """
        si, kh = self.si, self.kernel_handler
        ncharge, statesdm = si.ncharge, si.statesdm
        off_diag_corrections = self.funcp.off_diag_corrections

        _warn_if_unequal_temperature_cutoff_is_small(self.qd, self.leads)
        _warn_if_rtd_coherence_is_not_resolved(self)

        if (not np.all(np.isclose(self.leads.tlst, self.leads.tlst[0]))) or np.any(abs(self.leads.Tba.imag)>0):
            self.set_Ozaki_params()

        for bcharge in range(ncharge):
            for b in statesdm[bcharge]:
                if not kh.is_unique(b, b, bcharge):
                    continue
                self.generate_row_1st_order_kernel(b, bcharge)
                self.generate_col_diag_kern_2nd_order(b, bcharge)
                self.generate_row_1st_energy_kernel(b, bcharge)
                self.generate_row_2nd_energy_kernel(b, bcharge)

                if off_diag_corrections:
                    self.generate_col_nondiag_kern_1st_order_nd(b, bcharge)

        kern_size = self.get_kern_size()
        self.kern[:kern_size, :kern_size] += np.sum(self.Wdd, 0)

        if off_diag_corrections:
            for bcharge in range(ncharge):
                for b in statesdm[bcharge]:
                    for bp in statesdm[bcharge]:
                        if b == bp:
                            continue
                        self.generate_col_nondiag_kern_1st_order_dn(b, bp, bcharge)
                        self.generate_row_inverse_Liouvillian(b, bp, bcharge)
            self.add_off_diag_corrections()

    def add_off_diag_corrections(self):
        """
        Adds :math:`W_{dn}^{(1)} (L_{nn})^{-1} W_{nd}^{(1)}` to the total kernel.

        Parameters
        ----------
        self.kern : ndarray
            (Modifies) The total kernel

        self.Wdd : ndarray
            (Modifies) The lead-resolved diagonal kernel
        """
        kh = self.kernel_handler
        Wcorr = np.zeros(self.Wdd.shape, dtype=doublenp)
        for l in range(self.si.nleads):
            # Since off-diagonal kernels contain imaginary numbers we get two contributions to
            # the kernel
            Wcorr[l, :, :] += np.matmul(np.matmul(kh.ReWdn[l, :, :], kh.Lnn_inv[:, :]),
                                        np.sum(kh.ImWnd[:, :], 0))
            Wcorr[l, :, :] += np.matmul(np.matmul(kh.ImWdn[l, :, :], kh.Lnn_inv[:, :]),
                                        np.sum(kh.ReWnd[:, :], 0))
        self.Wdd += Wcorr
        kern_size = self.get_kern_size()
        self.kern[:kern_size, :kern_size] += np.sum(Wcorr, 0)

    def generate_LN(self):
        """Generates the diagonal of :math:`L_{N_{dot}}^+`

        Parameters
        ----------
        self.LN : ndarray
            (Modifies) the digononal entries of the anti-commutator Liouvillian for the number
            operator. The array has npauli entries.
        """
        charge_lst = []
        for charge in range(self.si.ncharge):
            for a in self.si.statesdm[charge]:
                if self.si.get_ind_dm0(a, a, charge, 2):
                    charge_lst.append(2 * charge)
        self.LN = np.array(charge_lst)

    def generate_current(self):
        r""" Calculates currents for the RTD approach.

        Charge current for reservoir r is evaluated as :math:`I^r = 1/2 \cdot Tr (L_{N_{dot}}^+ W^r \phi_0)`

        Energy current for reservoir r is evaluated as :math:`E^r = 1/2\cdot Tr (L_H^+ W^r \phi_0) - 1/2\cdot
        Tr(W_{E,1}^r \phi_0) + 1/2\cdot Tr(W_{E,2}^r \phi_0)`

        Heat current for reservoir r is evaluated as :math:`Q^r = E^r - \mu_r\cdot I^r`.

        Parameters
        ----------
        self.current : array
            (Modifies) Charge current in each lead.
        self.energy_current : array
            (Modifies) Energy charge current in each lead.
        self.heat_current : array
            (Modifies) Heat charge current in each lead.

        """
        kh = self.kernel_handler
        nleads = self.si.nleads
        LE = 2 * self.qd.Ea

        current = np.zeros(nleads, dtype=doublenp)
        energy_current = np.zeros(nleads, dtype=doublenp)

        for l in range(nleads):
            current[l] = 0.5 * np.dot(self.LN, np.dot(kh.Wdd[l, :, :], self.phi0))
            energy_current[l] = 0.5 * np.dot(LE, np.dot(kh.Wdd[l, :, :], self.phi0))
            energy_current[l] += -0.5 * np.sum(np.dot(kh.WE1[l, :, :], self.phi0))
            energy_current[l] += 0.5 * np.sum(np.dot(kh.WE2[l, :, :], self.phi0))

        self.current = current
        self.energy_current = energy_current
        self.heat_current = energy_current - current * self.leads.mulst

        if self.ImGamma:
            self.energy_current.fill(np.nan)
            self.heat_current.fill(np.nan)
            if not self.printed_warning_ImGamma:
                warnings.warn(
                    "Complex matrix elements are not supported for the RTD "
                    "energy current; energy_current and heat_current were "
                    "set to NaN.",
                    QmeqRuntimeWarning,
                    stacklevel=2,
                )
                self.printed_warning_ImGamma = True

        if self.si.nsingle == 0:
            if not self.nsingle_warning_printed:
                warnings.warn(
                    "No single-particle tunneling amplitudes (tleads) were "
                    "provided. RTD energy-current corrections use tleads; "
                    "specify BuilderManyBody.tleads_array and "
                    "BuilderManyBody.nsingle when possible. The corrections "
                    "can be neglected if no single-particle state is "
                    "connected to more than one lead.",
                    QmeqRuntimeWarning,
                    stacklevel=2,
                )
                self.nsingle_warning_printed = True

    def generate_fct(self):
        """
        Make factors used for generating the first order diagonal kernel :math:`W_{dd}^{(1)}`.

        Parameters
        ----------
        paulifct : array
            (Modifies) Factors used for generating Pauli master equation kernel.
        """
        E, Tba, si = self.qd.Ea, self.leads.Tba, self.si
        mulst, tlst, dlst = self.leads.mulst, self.leads.tlst, self.leads.dlst
        ncharge, nleads, statesdm = si.ncharge, si.nleads, si.statesdm

        itype = self.funcp.itype
        paulifct = self.paulifct
        for charge in range(ncharge-1):
            ccharge = charge+1
            bcharge = charge
            for c, b in itertools.product(statesdm[ccharge], statesdm[bcharge]):
                cb = si.get_ind_dm1(c, b, bcharge)
                Ecb = E[c]-E[b]
                for l in range(nleads):
                    xcb = (Tba[l, b, c]*Tba[l, c, b]).real
                    rez = func_pauli(Ecb, mulst[l], tlst[l], dlst[l, 0], dlst[l, 1], itype)
                    paulifct[l, cb, 0] = xcb*rez[0]
                    paulifct[l, cb, 1] = xcb*rez[1]

    def generate_row_1st_order_kernel(self, b, bcharge):
        """Generates a row in the first order diagonal kernel :math:`W_{dd}^{(1)}`.

        Parameters
        ----------
        b : int
            the state (row)

        bcharge : int
            charge of state b

        self.Wdd : ndarray
            (Modifies) The kernel connecting diagional density-matrix elements. This Kernel
            has npauli * npauli entries.
        """
        paulifct = self.paulifct
        si, kh = self.si, self.kernel_handler
        nleads, statesdm = si.nleads, si.statesdm

        acharge = bcharge-1
        ccharge = bcharge+1

        bb = si.get_ind_dm0(b, b, bcharge)
        for a in statesdm[acharge]:
            aa = si.get_ind_dm0(a, a, acharge)
            ba = si.get_ind_dm1(b, a, acharge)
            for l in range(nleads):
                fctm = -paulifct[l, ba, 1]
                fctp = paulifct[l, ba, 0]
                kh.set_matrix_element_dd(l, fctm, fctp, bb, aa, RtdMatrix.Wdd)
        for c in statesdm[ccharge]:
            cc = si.get_ind_dm0(c, c, ccharge)
            cb = si.get_ind_dm1(c, b, bcharge)
            for l in range(nleads):
                fctm = -paulifct[l, cb, 0]
                fctp = paulifct[l, cb, 1]
                kh.set_matrix_element_dd(l, fctm, fctp, bb, cc, RtdMatrix.Wdd)

    def generate_row_1st_energy_kernel(self, b, bcharge):
        r""" Generates a row of the first kernel for the barrier part of the energy current :math:`W_{E,1}`. This kernel
         is obtained from a diagram of the form
         :math:`L_{T,r}^+  L_{T, r_2}^-  (z-L_{dot} - L_r )^{-1}  L_T  (z-L_{dot} - L_r )^{-1} L_T \phi_0`,
         when contracting the first and third, as well as second and fourth, reservoir superoperators. Assumes
         that the wide-band limit is valid and that all products of tunnel matrix elements are real.

         Parameters
         ----------
         b : int
            the row (state)
         charge : int
            the charge of state b
         self.W2E_1 : ndarray
             (Modifies) The second order kernel used for evaluating the energy current. This Kernel
             has npauli * npauli entries.
         """
        (E, Tba, tleads, mulst, tlst, dlst) = (self.qd.Ea, self.leads.Tba, self.leads.tleads_array,
                                                   self.leads.mulst, self.leads.tlst, self.leads.dlst)
        si, kh = self.si, self.kernel_handler
        nleads, statesdm, nsingle = si.nleads, si.statesdm, si.nsingle

        acharge = bcharge-1
        ccharge = bcharge+1
        maxTemp = max(tlst)
        t_cutoff = 1e-15 * maxTemp ** 2

        bb = si.get_ind_dm0(b, b, bcharge)
        for a in statesdm[acharge]:
            aa = si.get_ind_dm0(a, a, acharge)
            for l in range(nleads):
                mu, Tr, gamma = mulst[l], tlst[l], 0.0
                dE = E[b] - E[a]
                for lp in range(nleads):
                    if lp == l:
                        continue
                    for n1 in range(nsingle):
                        gamma += Tba[l, a, b] * Tba[lp, a, b].conj() * tleads[l, n1] * tleads[lp, n1].conj()
                temp = gamma.real * phi((dE - mu) / Tr, dlst[l, 0] / Tr, dlst[l, 1] / Tr)
                temp += gamma.real * phi(-(dE - mu) / Tr, dlst[l, 0] / Tr, dlst[l, 1] / Tr)
                if abs(gamma.imag) > t_cutoff:
                    self.ImGamma = True
                    #temp += gamma.imag * fermi_func((dE - mu) / Tr)*np.pi
                    #temp += gamma.imag * fermi_func(-(dE - mu) / Tr) * np.pi

                temp *= np.pi
                kh.set_matrix_element_dd(l, temp, temp, bb, aa, RtdMatrix.WE1)

        for c in statesdm[ccharge]:
            cc = si.get_ind_dm0(c, c, ccharge)
            for l in range(nleads):
                mu, Tr, gamma = mulst[l], tlst[l], 0.0
                dE = E[c] - E[b]
                for lp in range(nleads):
                    if lp == l:
                        continue
                    for n1 in range(nsingle):
                        gamma += Tba[l, b, c] * Tba[lp, b, c].conj() * tleads[l, n1] * tleads[lp, n1].conj()
                temp = gamma.real * phi((dE - mu) / Tr, dlst[l, 0] / Tr, dlst[l, 1] / Tr)
                temp += gamma.real * phi(-(dE - mu) / Tr, dlst[l, 0] / Tr, dlst[l, 1] / Tr)
                if abs(gamma.imag) > t_cutoff:
                    self.ImGamma = True
                    #temp += gamma.imag * fermi_func((dE - mu) / Tr) * np.pi
                    #temp += gamma.imag * fermi_func(-(dE - mu) / Tr) * np.pi

                temp *= np.pi
                kh.set_matrix_element_dd(l, temp, temp, bb, cc, RtdMatrix.WE1)

    def generate_row_2nd_energy_kernel(self, b, bcharge):
        r""" Generates a row in the second kernel for the barrier part of the energy current :math:`W_{E,2}`.
        This kernel is obatined from a diagram of the form
        :math:`L_{T,r}^+  L_{T, r_2}^-  (z-L_{dot} - L_r )^{-1}  L_T  (z-L_{dot} - L_r )^{-1} L_T \phi_0`,
        when contracting the first and fourth, as well as second and third, reservoir superoperators. Assumes
        that the wide-band limit is valid and that all products of tunnel matrix elements are real.

        Parameters
        ----------
        b : int
            the row (state)
        charge : int
            the charge of state b
        self.W2E_2 : ndarray
            (Modifies) The second order kernel used for evaluating the energy current. This Kernel
            has npauli * npauli entries.
        """
        (E, si, Tba, tleads, mulst, tlst, dlst) = (self.qd.Ea, self.si, self.leads.Tba, self.leads.tleads_array,
                                                   self.leads.mulst, self.leads.tlst, self.leads.dlst)
        si, kh = self.si, self.kernel_handler
        nleads, statesdm, nsingle = si.nleads, si.statesdm, si.nsingle

        acharge = bcharge - 1
        ccharge = bcharge + 1
        maxTemp = max(tlst)
        t_cutoff = 1e-15*maxTemp**2

        bb = si.get_ind_dm0(b, b, bcharge)
        for a in statesdm[acharge]:
            aa = si.get_ind_dm0(a, a, acharge)
            for l in range(nleads):
                temp = 0.0
                for lp in range(nleads):
                    if lp != l:
                        mu, Tr, gamma = mulst[lp], tlst[lp], 0.0
                        for n1 in range(nsingle):
                            gamma += Tba[l, a, b] * Tba[lp, a, b].conj() * tleads[l, n1] * tleads[lp, n1].conj()
                        if abs(gamma.imag) > t_cutoff:
                            self.ImGamma = True
                        dE = E[b] - E[a]
                        temp += gamma.real * phi((dE - mu) / Tr, dlst[lp, 0] / Tr, dlst[lp, 1] / Tr)
                        temp += gamma.real * phi(-(dE - mu) / Tr, dlst[lp, 0] / Tr, dlst[lp, 1] / Tr)
                        if abs(gamma.imag) > t_cutoff:
                            self.ImGamma = True
                            #temp += gamma.imag * fermi_func((dE - mu) / Tr) * np.pi
                            #temp += gamma.imag * fermi_func(-(dE - mu) / Tr) * np.pi
                temp *= np.pi
                kh.set_matrix_element_dd(l, temp, temp, bb, aa, RtdMatrix.WE2)

        for c in statesdm[ccharge]:
            cc = si.get_ind_dm0(c, c, ccharge)
            for l in range(nleads):
                temp = 0.0
                for lp in range(nleads):
                    if lp != l:
                        mu, Tr, gamma = mulst[lp], tlst[lp], 0.0
                        for n1 in range(nsingle):
                            gamma += Tba[l, b, c] * Tba[lp, b, c].conj() * tleads[l, n1] * tleads[lp, n1].conj()
                        dE = E[c] - E[b]
                        temp += gamma.real * phi((dE - mu) / Tr, dlst[lp, 0] / Tr, dlst[lp, 1] / Tr)
                        temp += gamma.real * phi(-(dE - mu) / Tr, dlst[lp, 0] / Tr, dlst[lp, 1] / Tr)
                        if abs(gamma.imag) > t_cutoff:
                            self.ImGamma = True
                            #temp += gamma.imag * fermi_func((dE - mu) / Tr) * np.pi
                            #temp += gamma.imag * fermi_func(-(dE - mu) / Tr) * np.pi
                temp *= np.pi
                kh.set_matrix_element_dd(l, temp, temp, bb, cc, RtdMatrix.WE2)

    def generate_col_diag_kern_2nd_order(self, a0, charge):
        """Partly generates a column in the second order kernel for the diagonal density matrix :math:`W_{dd}^{(2)}`.
        Due to symmetries among the diagrammatic contributions for different matrix elements also contributions to
        other columns are generated. Assumes that the wide band limit is valid.

        Parameters
        ----------
        a0 : int
            initial state. Sets the column

        charge : int
            charge of state a0

        self.Wdd : ndarray
            (Modifies) diagonal lead-resolved kernel.

        """
        # Evaluating the matrix elements requires summing over three pairs of states |a_+><a_-|,
        # two leads, two electron-hole indices and four propagator signs. This is done as follows:
        # one starts with a loop over the initial states. Then, nested loops iterate over the
        # possible intermediate states of the diagram (which is a fairly small number due
        # to several delta functions arising when evaluating the diagrams). Eventually, the final state
        # is reached and a contribution to the matrix element, specified by the initial and the final
        # state, is found (this only provides a contribution since the intermediate states were not
        # fully looped over yet.) Calculating the full matrix elements requires calling this function
        # for every column in the kernel.
        #
        # Symmetries between diagram contributions are used to avoid looping over some Keldysh- and
        # electron-hole indices. Specifically p1 = 1, p4 = 1 and eta1 = 1 are fixed.
        #
        # For the tunnel matrix elements the following rules apply:
        # 1) t = Tba(i->f) if charge_i < charge_f else t = Tba(f->i).conj()
        # 2) t_n = t_n.conj() if p_n == -1
        #
        # Variable names follow Leijnse et al PRB 78, 235424 (2008). Specifically:
        # - aNp/aNm: states
        # - eta : electron-hole index
        # - r : lead index
        # - p : Keldysh sign
        # - z : energy difference

        statesdm, Tba, E = self.si.statesdm, self.leads.Tba, self.qd.Ea
        tlst, mulst, dlst = self.leads.tlst, self.leads.mulst, self.leads.dlst
        kh = self.kernel_handler
        nleads = self.si.nleads
        b_and_R = self.Ozaki_poles_and_residues

        t_cutoff1 = 0.0
        t_cutoff2 = 1e-10*max(tlst)
        t_cutoff3 = 1e-20*max(tlst)**2
        indx0 = self.si.get_ind_dm0(a0, a0, charge)
        for r0, r1 in product(range(nleads), range(nleads)):
            T1, T2 = tlst[r0], tlst[r1]
            mu1, mu2 = mulst[r0], mulst[r1]
            D = np.abs(dlst[r0, 1]) + np.abs(dlst[r0, 0])
            #N1 = (N0, N0 + 1), a1- = a0
            for a1p in statesdm[charge+1]:
                t = Tba[r0, a1p, a0]
                if abs(t) == t_cutoff1:
                    continue
                indx1 = self.si.get_ind_dm0(a1p, a1p, charge + 1)
                E1 = E[a1p] - E[a0]
                #eta1 = 1
                #p1 = 1
                #N2 = (N0, N0 + 2), a2m = a0
                for a2p in statesdm[charge+2]:
                    #p2 = 1
                    t1 = t * Tba[r1, a2p, a1p]
                    if abs(t1) < t_cutoff2:
                        continue
                    E2 = E[a2p] - E[a0]
                    #3 = (N0, N0 + 1 ), a3- = a2-
                    for a3p in statesdm[charge+1]:
                        #charge4 = charge + 0, a4 = a0
                        t2D = t1 * Tba[r1, a2p, a3p].conj() * Tba[r0, a3p, a0].conj()
                        t2X = t1 * Tba[r0, a2p, a3p].conj() * Tba[r1, a3p, a0].conj()
                        E3 = E[a3p] - E[a0]
                        if abs(t2D) > t_cutoff3:
                            tempD = t2D * integralD(1, 1, E1, E2, E3, T1, T2, mu1, mu2, D, b_and_R, _has_significant_imaginary_part(t2D))
                            kh.add_element_2nd_order(r0, tempD.real, indx0, indx1, a3p, charge + 1, a0, charge)
                        if abs(t2X) > t_cutoff3:
                            tempX = -t2X * integralX(1, 1, E1, E2, E3, T1, T2, mu1, mu2, D, b_and_R, _has_significant_imaginary_part(t2X))
                            kh.add_element_2nd_order(r1, tempX.real, indx0, indx1, a3p, charge + 1, a0, charge)
                    #p2 = -1
                    #N3 = ( N0 +1, N0 + 2)
                    for a3m in statesdm[charge+1]:
                        #charge4 = charge + 1, a4 = a3m
                        t2D = t1 * Tba[r1, a3m, a0].conj() * Tba[r0, a2p, a3m].conj()
                        t2X = t1 * Tba[r0, a3m, a0].conj() * Tba[r1, a2p, a3m].conj()
                        E3 = E[a2p] - E[a3m]
                        if abs(t2D) > t_cutoff3:
                            tempD = t2D * integralD(1, 1, E1, E2, E3, T1, T2, mu1, mu2, D, b_and_R, _has_significant_imaginary_part(t2D))
                            kh.add_element_2nd_order(r0, tempD.real, indx0, indx1, a2p, charge + 2, a3m, charge + 1)
                        if abs(t2X) > t_cutoff3:
                            tempX = -t2X * integralX(1, 1, E1, E2, E3, T1, T2, mu1, mu2, D, b_and_R, _has_significant_imaginary_part(t2X))
                            kh.add_element_2nd_order(r1, tempX.real, indx0, indx1, a2p, charge + 2, a3m, charge+1)
                #p1 = -1
                #N2 = ( N0 - 1, N0 + 1 ), a2+ = a1+
                for a2m in statesdm[charge-1]:
                    t1 = t * Tba[r1, a0, a2m]
                    if abs(t1) < t_cutoff2:
                        continue
                    E2 = E[a1p] - E[a2m]
                    #p2 = 1
                    #N3 = ( N0 - 1 , N0 ), a3- = a2-
                    for a3p in statesdm[charge]:
                        #charge4 = charge - 1, a0 = a2m
                        t2D = t1 * Tba[r1, a1p, a3p].conj() * Tba[r0, a3p, a2m].conj()
                        t2X = t1 * Tba[r0, a1p, a3p].conj() * Tba[r1, a3p, a2m].conj()
                        E3 = E[a3p] - E[a2m]
                        if abs(t2D) > t_cutoff3:
                            tempD = t2D * integralD(-1, 1, E1, E2, E3, T1, T2, mu1, mu2, D, b_and_R, _has_significant_imaginary_part(t2D))
                            kh.add_element_2nd_order(r0, tempD.real, indx0, indx1, a3p, charge, a2m, charge - 1)
                        if abs(t2X) > t_cutoff3:
                            tempX = -t2X * integralX(-1, 1, E1, E2, E3, T1, T2, mu1, mu2, D, b_and_R, _has_significant_imaginary_part(t2X))
                            kh.add_element_2nd_order(r1, tempX.real, indx0, indx1, a3p, charge, a2m, charge-1)
                    #p2 = -1
                    #N3 = ( N0 , N0 + 1), a3+ = a2+
                    for a3m in statesdm[charge]:
                        #charge4 = charge + 0, a4 = a3m
                        t2D = t1 * Tba[r1, a3m, a2m].conj() * Tba[r0, a1p, a3m].conj()
                        t2X = t1 * Tba[r0, a3m, a2m].conj() * Tba[r1, a1p, a3m].conj()
                        E3 = E[a1p] - E[a3m]
                        if abs(t2D) > t_cutoff3:
                            tempD = t2D * integralD(-1, 1, E1, E2, E3, T1, T2, mu1, mu2, D, b_and_R, _has_significant_imaginary_part(t2D))
                            kh.add_element_2nd_order(r0, tempD.real, indx0, indx1, a1p, charge + 1, a3m, charge)
                        if abs(t2X) > t_cutoff3:
                            tempX = -t2X * integralX(-1, 1, E1, E2, E3, T1, T2, mu1, mu2, D, b_and_R, _has_significant_imaginary_part(t2X))
                            kh.add_element_2nd_order(r1, tempX.real, indx0, indx1, a1p, charge + 1, a3m, charge)
                #eta1 = -1
                #p1 = 1
                #N2 = (N0, N0), a2- = a0
                for a2p in statesdm[charge]:
                    E2 = E[a2p] - E[a0]
                    t1 = t * Tba[r1, a1p, a2p].conj()
                    if abs(t1) < t_cutoff2:
                        continue
                    #p2 = 1
                    #N3 = ( N0, N0 +1), a3- = a0
                    for a3p in statesdm[charge+1]:
                        #charge4 = charge, a4 = a0
                        t2D = t1 * Tba[r1, a3p, a2p] * Tba[r0, a3p, a0].conj()
                        E3 = E[a3p] - E[a0]
                        if abs(t2D) > t_cutoff3:
                            tempD = t2D * integralD(1, -1, E1, E2, E3, T1, T2, mu1, mu2, D, b_and_R, _has_significant_imaginary_part(t2D))
                            kh.add_element_2nd_order(r0, tempD.real, indx0, indx1, a3p, charge + 1, a0, charge)
                    #N3 = (N0, N0-1), a3- = a0
                    for a3p in statesdm[charge-1]:
                        #charge4 = charge, a4 = a0
                        t2X = t1 * Tba[r0, a2p, a3p].conj() * Tba[r1, a0, a3p]
                        E3 = E[a3p] - E[a0]
                        if abs(t2X) > t_cutoff3:
                            tempX = -t2X * integralX(1, -1, E1, E2, E3, T1, T2, mu1, mu2, D, b_and_R, _has_significant_imaginary_part(t2X))
                            kh.add_element_2nd_order(r1, tempX.real, indx0, indx1, a3p, charge - 1, a0, charge)
                    #p2 = -1
                    #N3 = ( N0-1, N0 ), a3+ = a2+
                    for a3m in statesdm[charge-1]:
                        #charge4 = charge - 1, a4 = a3m
                        t2D = t1 * Tba[r1, a0, a3m] * Tba[r0, a2p, a3m].conj()
                        E3 = E[a2p] - E[a3m]
                        if abs(t2D) > t_cutoff3:
                            tempD = t2D * integralD(1, -1, E1, E2, E3, T1, T2, mu1, mu2, D, b_and_R, _has_significant_imaginary_part(t2D))
                            kh.add_element_2nd_order(r0, tempD.real, indx0, indx1, a2p, charge, a3m, charge - 1)
                    #N3 = (N0 + 1, N0)
                    for a3m in statesdm[charge+1]:
                        #charge4 = charge + 1, a4 = a3m
                        t2X = t1 * Tba[r0, a3m, a0].conj() * Tba[r1, a3m, a2p]
                        E3 = E[a2p] - E[a3m]
                        if abs(t2X) > t_cutoff3:
                            tempX = -t2X * integralX(1, -1, E1, E2, E3, T1, T2, mu1, mu2, D, b_and_R, _has_significant_imaginary_part(t2X))
                            kh.add_element_2nd_order(r1, tempX.real, indx0, indx1, a2p, charge, a3m, charge + 1)
                #p1 = -1
                #N2 = ( N0 + 1  , N0 + 1), a2+ = a1+
                for a2m in statesdm[charge+1]:
                    E2 = E[a1p] - E[a2m]
                    t1 = t * Tba[r1, a2m, a0].conj()
                    #p2 = 1
                    # N3 = (N0 + 1, N0 + 2), a3- = a2-
                    for a3p in statesdm[charge+2]:
                        #charge4 = charge + 1, a4 = a2m
                        t2D = t1 * Tba[r1, a3p, a1p] * Tba[r0, a3p, a2m].conj()
                        E3 = E[a3p] - E[a2m]
                        if abs(t2D) > t_cutoff3:
                            tempD = t2D * integralD(-1, -1, E1, E2, E3, T1, T2, mu1, mu2, D, b_and_R, _has_significant_imaginary_part(t2D))
                            kh.add_element_2nd_order(r0, tempD.real, indx0, indx1, a3p, charge + 2, a2m, charge + 1)
                    #N3 = ( N0 + 1, N0 )
                    for a3p in statesdm[charge]:
                        #charge4 = charge + 1, a4 = a2m
                        t2X = t1 * Tba[r0, a1p, a3p].conj() * Tba[r1, a2m, a3p]
                        E3 = E[a3p] - E[a2m]
                        if abs(t2X) > t_cutoff3:
                            tempX = -t2X * integralX(-1, -1, E1, E2, E3, T1, T2, mu1, mu2, D, b_and_R, _has_significant_imaginary_part(t2X))
                            kh.add_element_2nd_order(r1, tempX.real, indx0, indx1, a3p, charge, a2m, charge + 1)
                    #p2 = -1
                    #N3 = ( N0, N0+1), a3+ = a2+
                    for a3m in statesdm[charge]:
                        #charge4 = charge, a4 = a3m
                        t2D = t1 * Tba[r1, a2m, a3m] * Tba[r0, a1p, a3m].conj()
                        E3 = E[a1p] - E[a3m]
                        if abs(t2D) > t_cutoff3:
                            tempD = t2D * integralD(-1, -1, E1, E2, E3, T1, T2, mu1, mu2, D, b_and_R, _has_significant_imaginary_part(t2D))
                            kh.add_element_2nd_order(r0, tempD.real, indx0, indx1, a1p, charge + 1, a3m, charge)
                    #N3 = ( N0 + 2, N0 + 1 ), a3+ = a2+
                    for a3m in statesdm[charge+2]:
                        #charge4 = charge + 2, a4 = a3m
                        t2X = t1 * Tba[r0, a3m, a2m].conj() * Tba[r1, a3m, a1p]
                        E3 = E[a1p] - E[a3m]
                        if abs(t2X) > t_cutoff3:
                            tempX = -t2X * integralX(-1, -1, E1, E2, E3, T1, T2, mu1, mu2, D, b_and_R, _has_significant_imaginary_part(t2X))
                            kh.add_element_2nd_order(r1, tempX.real, indx0, indx1, a1p, charge + 1, a3m, charge + 2)

    def generate_col_nondiag_kern_1st_order_dn(self, a1, b1, charge):
        r""" Calculates a column in :math:`W_{dn}^{(1)}`, the part of the full first order off-diagonal kernel
        connecting diagonal :math:`|a2><a2|` and non-diagonal :math:`|a1><b1|` entries of the density matrix.
        Separates the real and imaginary parts into two matrices. Assumes that the wide band limit is valid.

        Parameters
        ----------
        a1 : int
            first state indexing the column

        b1 : int
            second state indexing the column

        charge: int
            charge of state a1 and b1

        self.ImWdn : ndarray
            (Modifies) :math:`\Im(W_{dn}^{(1)})`, the imaginary part of the first order kernel that connects
            diagonal and non-diagonal elements of the density matrix.
        self.ReWdn : ndarray
            (Modifies) :math:`\Re(W_{dn}^{(1)})`, the real part of the first order kernel that connects
            diagonal and non-diagonal elements of the density matrix.
        """
        # Variables naming follow Leijnse et al.PRB 78, 235424(2008).

        (E, si, Tba, mulst, tlst, dlst) = (self.qd.Ea, self.si, self.leads.Tba,
                                           self.leads.mulst, self.leads.tlst, self.leads.dlst)
        PI = np.pi
        si, kh = self.si, self.kernel_handler
        nleads, statesdm, ncharge = si.nleads, si.statesdm, si.ncharge

        # final state in higher charge state
        if charge != ncharge - 1:
            # Loop over diagonal final state
            for a2 in statesdm[charge + 1]:
                E1 = E[a2] - E[a1]
                E2 = E[a2] - E[b1]
                for l in range(nleads):
                    t2 = Tba[l, a2, a1] * Tba[l, a2, b1].conj()
                    f = fermi_func((E1 - mulst[l]) / tlst[l]) + fermi_func((E2 - mulst[l]) / tlst[l])
                    phi0 = delta_phi((E1 - mulst[l]) / tlst[l], (E2 - mulst[l]) / tlst[l],
                                     dlst[l, 0] / tlst[l], dlst[l, 1] / tlst[l])
                    temp1 = PI * t2.real * f - t2.imag * phi0
                    temp2 = t2.real * phi0 + PI * t2.imag * f
                    kh.add_matrix_element(temp1, l, a2, a2, charge + 1, a1, b1, charge, RtdMatrix.ReWdn)
                    kh.add_matrix_element(temp2, l, a2, a2, charge + 1, a1, b1, charge, RtdMatrix.ImWdn)
        # Final state in lower charge state
        if charge != 0:
            # Loop through diagonal final states
            for a2 in statesdm[charge - 1]:
                E1 = E[b1] - E[a2]
                E2 = E[a1] - E[a2]
                for l in range(nleads):
                    t2 = Tba[l, b1, a2] * Tba[l, a1, a2].conj()
                    f = fermi_func(-(E1 - mulst[l]) / tlst[l]) + fermi_func(-(E2 - mulst[l]) / tlst[l])
                    phi0 = delta_phi((E1 - mulst[l]) / tlst[l], (E2 - mulst[l]) / tlst[l],
                                     dlst[l, 0] / tlst[l], dlst[l, 1] / tlst[l], sign=-1)
                    temp1 = PI * t2.real * f - t2.imag * phi0
                    temp2 = t2.real * phi0 + PI * t2.imag * f
                    kh.add_matrix_element(temp1, l, a2, a2, charge - 1, a1, b1, charge, RtdMatrix.ReWdn)
                    kh.add_matrix_element(temp2, l, a2, a2, charge - 1, a1, b1, charge, RtdMatrix.ImWdn)
        # Loop over final state, conserving charge
        for a2 in statesdm[charge]:
            if charge != ncharge - 1:
                # Intermediate state in higher charge state
                for c in statesdm[charge + 1]:
                    if b1 == a2:  # vertices on upper prop -> state on lower prop cannot change
                        E1 = E[c] - E[a2]
                        for l in range(nleads):
                            t2 = Tba[l, c, a1] * Tba[l, c, a2].conj()
                            f = fermi_func((E1 - mulst[l]) / tlst[l])
                            phi0 = phi((E1 - mulst[l]) / tlst[l], dlst[l, 0] / tlst[l],
                                       dlst[l, 1] / tlst[l], sign=1)
                            temp1 = -PI * t2.real * f - t2.imag * phi0
                            temp2 = -PI * t2.imag * f + t2.real * phi0
                            kh.add_matrix_element(temp1, l, a2, a2, charge, a1, b1, charge, RtdMatrix.ReWdn)
                            kh.add_matrix_element(temp2, l, a2, a2, charge, a1, b1, charge, RtdMatrix.ImWdn)
                    if a1 == a2:  # vertices on lower prop -> state on upper prop cannot change
                        E1 = E[c] - E[a2]
                        for l in range(si.nleads):
                            t2 = Tba[l, c, a2] * Tba[l, c, b1].conj()
                            f = fermi_func((E1 - mulst[l]) / tlst[l])
                            phi0 = phi((E1 - mulst[l]) / tlst[l], dlst[l, 0] / tlst[l],
                                       dlst[l, 1] / tlst[l], sign=1)
                            temp1 = - PI * t2.real * f + t2.imag * phi0
                            temp2 = - PI * t2.imag * f - t2.real * phi0
                            kh.add_matrix_element(temp1, l, a2, a2, charge, a1, b1, charge, RtdMatrix.ReWdn)
                            kh.add_matrix_element(temp2, l, a2, a2, charge, a1, b1, charge, RtdMatrix.ImWdn)
            if charge != 0:
                # Intermediate state in lower charge state
                for c in statesdm[charge - 1]:
                    if b1 == a2:  # vertices on upper prop -> state on lower prop cannot change
                        E1 = E[a2] - E[c]
                        for l in range(nleads):
                            t2 = Tba[l, a2, c] * Tba[l, a1, c].conj()
                            f = fermi_func(-(E1 - mulst[l]) / tlst[l])
                            phi0 = phi((E1 - mulst[l]) / tlst[l], dlst[l, 0] / tlst[l],
                                       dlst[l, 1] / tlst[l], sign=-1)
                            temp1 = - PI * t2.real * f + t2.imag * phi0
                            temp2 = - PI * t2.imag * f - t2.real * phi0
                            kh.add_matrix_element(temp1, l, a2, a2, charge, a1, b1, charge, RtdMatrix.ReWdn)
                            kh.add_matrix_element(temp2, l, a2, a2, charge, a1, b1, charge, RtdMatrix.ImWdn)
                    if a1 == a2:  # vertices on lower prop -> state on upper prop cannot change
                        E1 = E[a2] - E[c]
                        for l in range(nleads):
                            t2 = Tba[l, b1, c] * Tba[l, a2, c].conj()
                            f = fermi_func(-(E1 - mulst[l]) / tlst[l])
                            phi0 = phi((E1 - mulst[l]) / tlst[l], dlst[l, 0] / tlst[l],
                                       dlst[l, 1] / tlst[l], sign=-1)
                            temp1 = - PI * t2.real * f - t2.imag * phi0
                            temp2 = - PI * t2.imag * f + t2.real * phi0
                            kh.add_matrix_element(temp1, l, a2, a2, charge, a1, b1, charge, RtdMatrix.ReWdn)
                            kh.add_matrix_element(temp2, l, a2, a2, charge, a1, b1, charge, RtdMatrix.ImWdn)

    def generate_col_nondiag_kern_1st_order_nd(self, a1, charge):
        r""" Calculates a column in :math:`W_{nd}^{(1)}`, the part of the full first order off-diagonal kernel
        connecting non-diagonal :math:`|a2><b2|` and diagonal :math:`|a1><a1|` entries of the density matrix.
        Separates the real and imaginary parts into two matrices. Assumes that the wide band limit is valid.

        Parameters
        ----------
        a1 : int
            state indexing the column

        charge: int
            charge of state a1

        self.ImWdn : ndarray
            (Modifies) :math:`\Im(W_{nd}^{(1)})`, the imaginary part of the first order kernel that connects
            non-diagonal and diagonal elements of the density matrix.
        self.ReWdn : ndarray
            (Modifies) :math:`\Re(W_{nd}^{(1)})`, the real part of the first order kernel that connects
            non-diagonal and diagonal elements of the density matrix.
        """
        # Variables naming follow Leijnse et al.PRB 78, 235424(2008).

        (E, si, Tba, mulst, tlst, dlst) = (self.qd.Ea, self.si, self.leads.Tba,
                                           self.leads.mulst, self.leads.tlst, self.leads.dlst)

        si, kh = self.si, self.kernel_handler
        nleads, statesdm, ncharge = si.nleads, si.statesdm, si.ncharge
        PI = np.pi

        if charge != ncharge - 1:
            # Loop over final state, adding electron to the QD
            for a2 in statesdm[charge + 1]:
                E2 = E[a2] - E[a1]
                for b2 in statesdm[charge + 1]:
                    if a2 == b2:  # Final state must be off-diagonal
                        continue
                    E1 = E[b2] - E[a1]
                    for l in range(nleads):
                        t2 = Tba[l, a2, a1] * Tba[l, b2, a1].conj()
                        f = fermi_func((E1 - mulst[l]) / tlst[l]) + fermi_func((E2 - mulst[l]) / tlst[l])
                        phi0 = delta_phi((E1 - mulst[l]) / tlst[l], (E2 - mulst[l]) / tlst[l],
                                         dlst[l, 0] / tlst[l], dlst[l, 1] / tlst[l])
                        temp1 = PI * t2.real * f - t2.imag * phi0
                        temp2 = t2.real * phi0 + PI * t2.imag * f
                        kh.add_matrix_element(temp1, l, a2, b2, charge + 1, a1, a1, charge, RtdMatrix.ReWnd)
                        kh.add_matrix_element(temp2, l, a2, b2, charge + 1, a1, a1, charge, RtdMatrix.ImWnd)
        if charge != 0:
            # Loop over final states, removing electron from the QD
            for a2 in statesdm[charge - 1]:
                E1 = E[a1] - E[a2]
                for b2 in statesdm[charge - 1]:
                    if b2 == a2:  # Final state must be off-diagonal
                        continue
                    E2 = E[a1] - E[b2]
                    for l in range(si.nleads):
                        t2 = Tba[l, a1, b2] * Tba[l, a1, a2].conj()
                        f = fermi_func(-(E1 - mulst[l]) / tlst[l]) + fermi_func(-(E2 - mulst[l]) / tlst[l])
                        phi0 = delta_phi((E1 - mulst[l]) / tlst[l], (E2 - mulst[l]) / tlst[l],
                                         dlst[l, 0] / tlst[l], dlst[l, 1] / tlst[l], sign=-1)
                        temp1 = PI * t2.real * f - t2.imag * phi0
                        temp2 = t2.real * phi0 + PI * t2.imag * f
                        kh.add_matrix_element(temp1, l, a2, b2, charge - 1, a1, a1, charge, RtdMatrix.ReWnd)
                        kh.add_matrix_element(temp2, l, a2, b2, charge - 1, a1, a1, charge, RtdMatrix.ImWnd)
        # Loop over final state conserving charge
        for a2 in statesdm[charge]:
            for b2 in statesdm[charge]:
                if a2 == b2:  # Final state must be off-diagonal
                    continue
                if charge != ncharge - 1:
                    # Intermediate state in higher charge state
                    for c in statesdm[charge + 1]:
                        if a1 == b2:
                            E1 = E[c] - E[b2]
                            for l in range(nleads):
                                t2 = Tba[l, c, a1] * Tba[l, c, a2].conj()
                                f = fermi_func((E1 - mulst[l]) / tlst[l])
                                phi0 = phi((E1 - mulst[l]) / tlst[l], dlst[l, 0] / tlst[l], dlst[l, 1] / tlst[l])
                                temp1 = - PI * t2.real * f - t2.imag * phi0
                                temp2 = - PI * t2.imag * f + t2.real * phi0
                                kh.add_matrix_element(temp1, l, a2, b2, charge, a1, a1, charge, RtdMatrix.ReWnd)
                                kh.add_matrix_element(temp2, l, a2, b2, charge, a1, a1, charge, RtdMatrix.ImWnd)
                        if a1 == a2:
                            E1 = E[c] - E[a2]
                            for l in range(nleads):
                                t2 = Tba[l, c, b2] * Tba[l, c, a1].conj()
                                f = fermi_func((E1 - mulst[l]) / tlst[l])
                                phi0 = phi((E1 - mulst[l]) / tlst[l], dlst[l, 0] / tlst[l], dlst[l, 1] / tlst[l])
                                temp1 = - PI * t2.real * f + t2.imag * phi0
                                temp2 = - PI * t2.imag * f - t2.real * phi0
                                kh.add_matrix_element(temp1, l, a2, b2, charge, a1, a1, charge, RtdMatrix.ReWnd)
                                kh.add_matrix_element(temp2, l, a2, b2, charge, a1, a1, charge, RtdMatrix.ImWnd)
                if charge != 0:
                    # Intermediate state in lower charge state
                    for c in statesdm[charge - 1]:
                        if a1 == b2:
                            E1 = E[b2] - E[c]
                            for l in range(nleads):
                                t2 = Tba[l, a2, c] * Tba[l, a1, c].conj()
                                f = fermi_func(-(E1 - mulst[l]) / tlst[l])
                                phi0 = phi((E1 - mulst[l])/tlst[l], dlst[l, 0] / tlst[l], dlst[l, 1] / tlst[l], sign=-1)
                                temp1 = - PI * t2.real * f + t2.imag * phi0
                                temp2 = - PI * t2.imag * f - t2.real * phi0
                                kh.add_matrix_element(temp1, l, a2, b2, charge, a1, a1, charge, RtdMatrix.ReWnd)
                                kh.add_matrix_element(temp2, l, a2, b2, charge, a1, a1, charge, RtdMatrix.ImWnd)
                        if a1 == a2:
                            E1 = E[a2] - E[c]
                            for l in range(nleads):
                                t2 = Tba[l, a1, c] * Tba[l, b2, c].conj()
                                f = fermi_func(-(E1 - mulst[l]) / tlst[l])
                                phi0 = phi((E1 - mulst[l])/tlst[l], dlst[l, 0] / tlst[l], dlst[l, 1] / tlst[l], sign=-1)
                                temp1 = - PI * t2.real * f - t2.imag * phi0
                                temp2 = - PI * t2.imag * f + t2.real * phi0
                                kh.add_matrix_element(temp1, l, a2, b2, charge, a1, a1, charge, RtdMatrix.ReWnd)
                                kh.add_matrix_element(temp2, l, a2, b2, charge, a1, a1, charge, RtdMatrix.ImWnd)

    def generate_row_inverse_Liouvillian(self, a1, b1, charge):
        """ Calculates a row of :math:`1/(L_{nn})` (in practice only the diagonal is needed) where :math:`L_{nn}` is the part
        of the Liouvillian connecting non-diagonal states :math:`|i><j|` with non-diagonal states :math:`|k><l|`.

        Parameters
        ----------
        a1 : int
            first state indexing the column

        b1 : int
            second state indexing the column

        charge: int
            charge of state a1 and b1

        self.Wnn : ndarray
            (Modifies) an ndarray representing :math:`1/L_{nn}`. This kernel has noffdiag * noffdiag entries.

        """
        # Variables names follow Leijnse et al PRB 78, 235424 (2008).
        E = self.qd.Ea

        minE = 1e-10
        E1 = E[a1] - E[b1]
        if minE > E1 >= 0:
            E1 = minE
        elif -minE < E1 <= 0:
            E1 = -minE

        self.kernel_handler.add_element_Lnn_inv(a1, b1, charge, 1.0/E1)

    def set_Ozaki_params(self):
        """
        Generates and stores the residues and poles for the Ozaki expansion
        of tanh(z).

        Parameters
        ----------
        self.BW_Ozaki_expansion : double
            (Modifies) The band width (over temperature) for which the Ozaki poles and residues are generated.

        self.Ozaki_poles_and_residues : ndarray
            a ndarray where the first column contains the reciprocal of the poles
            and the second columns contains the residues.
        """
        bandwidth = np.max(np.sum(np.abs(self.leads.dlst), axis=1))
        BW_T = bandwidth / 2.0 / min(self.leads.tlst)
        if self.BW_Ozaki_expansion < BW_T:
            self.Ozaki_poles_and_residues = BW_Ozaki(BW_T)
            self.BW_Ozaki_expansion = BW_T

    def generate_vec(self, phi0):
        """"""
        raise NotImplementedError('Matrix free methods are not supported by the RTD approach.')
