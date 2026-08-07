"""Module containing python functions, which generate first order Lindblad kernels."""

import numpy as np
import itertools

from ...wrappers.mytypes import complexnp
from ...wrappers.mytypes import doublenp

from ...specfunc.specfunc import func_pauli
from ...specfunc.specfunc import func_lambshift
from ..aprclass import Approach

from ..kernel_handler import KernelHandlerNoise

# ---------------------------------------------------------------------------------------------------
# Lamb shift Hamiltonian
# ---------------------------------------------------------------------------------------------------
def generate_lamb_shift(appr):
    r"""
    Makes the lead resolved Lamb shift Hamiltonian used in the Lindblad master equation.

    The Lamb shift is the renormalisation of the many-body energies of the quantum dot due
    to the coupling to the leads. It is the principal value counterpart of the Lindblad
    dissipator and enters the unitary part of the evolution,

    .. math::

        \dot{\rho} = -i[H_{QD}+H_{LS},\rho] + \mathcal{D}[\rho].

    Beyond the secular approximation, i.e., keeping all energy differences of the many-body
    spectrum and not only those of neighbouring levels, the Lamb shift Hamiltonian reads

    .. math::

        (H_{LS}^{l})_{bb'} = \frac{1}{2}\sum_{a}T^{l}_{ba}T^{l}_{ab'}
                             \left[\Lambda_{l}(E_b-E_a)+\Lambda_{l}(E_{b'}-E_a)\right]
                           + \frac{1}{2}\sum_{c}T^{l}_{bc}T^{l}_{cb'}
                             \left[\tilde{\Lambda}_{l}(E_b-E_c)
                                   +\tilde{\Lambda}_{l}(E_{b'}-E_c)\right],

    where :math:`H_{LS}=\sum_{l}H_{LS}^{l}`, the states :math:`a` (:math:`c`) have one
    electron less (more) than the states :math:`b`, :math:`b'`, and

    .. math::

        \Lambda_{l}(E) = \mathrm{Re}\,\psi\!\left(\frac{1}{2}
                         + i\frac{E-\mu_{l}}{2\pi T_{l}}\right), \qquad
        \tilde{\Lambda}_{l}(E) = \Lambda_{l}(E)\big|_{\mu_l\to-\mu_l},

    are the principal value factors returned by
    :func:`~qmeq.specfunc.specfunc.func_lambshift`. The particle (:math:`a`) and hole
    (:math:`c`) contributions correspond to the two terms of the anticommutator of the
    tunneling operators. :math:`H_{LS}` is Hermitian and block diagonal in the charge, as it
    has to be because the charge of the total system is conserved.

    The Lamb shift is controlled by the descriptive ``principal_part`` option. Setting
    it to ``'digamma'`` includes the shift and ``'omit'`` excludes it; numerical
    quadrature is not implemented for the Lindblad approach. The legacy ``itype`` option
    continues to control the dissipative transition rates but does not opt into the
    newly implemented Lamb shift.

    This is a module level function and not a method of
    :class:`~qmeq.approach.base.lindblad.ApproachLindblad`, because the electron-phonon
    Lindblad approach reuses the kernel of the electron-lead part with an instance which is
    not an :class:`~qmeq.approach.base.lindblad.ApproachLindblad`.

    Parameters
    ----------
    appr : Approach
        Approach object holding the arrays below.
    """
    Tba, E, si = appr.leads.Tba, appr.qd.Ea, appr.si
    mulst, tlst = appr.leads.mulst, appr.leads.tlst
    ncharge, nleads, statesdm = si.ncharge, si.nleads, si.statesdm

    if appr.funcp.principal_part != "digamma":
        return

    HLS = appr.HLS
    for bcharge in range(ncharge):
        acharge = bcharge-1
        ccharge = bcharge+1
        for b, bp in itertools.combinations_with_replacement(statesdm[bcharge], 2):
            for l in range(nleads):
                mu, T = mulst[l], tlst[l]
                fct = 0
                for a in statesdm[acharge]:
                    fct += 0.5*Tba[l, b, a]*Tba[l, a, bp]*(
                            func_lambshift(E[b]-E[a], mu, T)
                            + func_lambshift(E[bp]-E[a], mu, T))
                for c in statesdm[ccharge]:
                    fct += 0.5*Tba[l, b, c]*Tba[l, c, bp]*(
                            func_lambshift(E[b]-E[c], -mu, T)
                            + func_lambshift(E[bp]-E[c], -mu, T))
                HLS[l, b, bp] = fct
                HLS[l, bp, b] = fct.conjugate()


# ---------------------------------------------------------------------------------------------------
# Lindblad approach
# ---------------------------------------------------------------------------------------------------
class ApproachLindblad(Approach):

    kerntype = 'pyLindblad'

    def restart(self): # simon
        Approach.restart(self)
        self.Lpm = None
        self.current_noise = None
        self.energy_current_noise = None

    def prepare_kernel_handler(self):
        if self.funcp.mfreeq:
            Approach.prepare_kernel_handler(self)
        else:
            self.kernel_handler = KernelHandlerNoise(self.si)

    def prepare_arrays(self):
        Approach.prepare_arrays(self)
        Tba, mtype = self.leads.Tba, self.leads.mtype
        self.tLba = np.zeros(Tba.shape, dtype=mtype)
        self.HLS = np.zeros(Tba.shape, dtype=mtype)
        ndm0r = self.si.ndm0r #simon
        if not self.funcp.mfreeq:
            self.Lpm = np.zeros((2, ndm0r, ndm0r), dtype=doublenp)
            self.kernel_handler.set_lpm(self.Lpm)
            self.current_noise = np.zeros(2)
        # create additional vectors/matrices: 1d array with noise at all leads, matrix with derivative (liouvillian/parts)
        # make sure kernel handler knows where to find new kernel (point to it)

    def clean_arrays(self):
        Approach.clean_arrays(self)
        self.tLba.fill(0.0)
        self.HLS.fill(0.0)
        if not self.funcp.mfreeq:
            self.Lpm.fill(0.0)
            self.current_noise.fill(0.0)
        # create additional vectors/matrices: 1d array with noise at all leads, matrix with derivative (liouvillian/parts)

    def generate_fct(self):
        """
        Make factors used for generating Lindblad master equation kernel.

        Parameters
        ----------
        tLba : array
            (Modifies) Jump operator matrix in many-body basis.
        HLS : array
            (Modifies) Lead resolved Lamb shift Hamiltonian in many-body basis,
            see generate_lamb_shift.
        """
        Tba, E, si = self.leads.Tba, self.qd.Ea, self.si
        mulst, tlst, dlst = self.leads.mulst, self.leads.tlst, self.leads.dlst
        itype = self.funcp.itype
        ncharge, nleads, statesdm = si.ncharge, si.nleads, si.statesdm

        tLba = self.tLba
        for charge in range(ncharge-1):
            bcharge = charge+1
            acharge = charge
            for b, a in itertools.product(statesdm[bcharge], statesdm[acharge]):
                Eba = E[b]-E[a]
                for l in range(nleads):
                    fct1, fct2 = func_pauli(Eba, mulst[l], tlst[l], dlst[l, 0], dlst[l, 1], itype)
                    tLba[l, b, a] = np.sqrt(fct1)*Tba[l, b, a]
                    tLba[l, a, b] = np.sqrt(fct2)*Tba[l, a, b]

        generate_lamb_shift(self)

    def generate_coupling_terms(self, b, bp, bcharge):
        tLba, HLS = self.tLba, self.HLS
        si, kh = self.si, self.kernel_handler
        nleads, statesdm = si.nleads, si.statesdm
        Lpm = self.Lpm #simon
        countingleads = () if self.funcp.mfreeq else self.funcp.countingleads

        acharge = bcharge-1
        ccharge = bcharge+1

        for a, ap in itertools.product(statesdm[acharge], statesdm[acharge]):
            if kh.is_included(a, ap, acharge):
                fct_aap = 0
                for l in range(nleads):
                    fct_aap += tLba[l, b, a]*tLba[l, bp, ap].conjugate()
                    if l in countingleads:
                        kh.set_matrix_element_lpm(1j*tLba[l, b, a]*tLba[l, bp, ap].conjugate(), 0, b, bp, bcharge, a, ap, acharge)
                kh.set_matrix_element(1j*fct_aap, b, bp, bcharge, a, ap, acharge)
        # --------------------------------------------------
        for bpp in statesdm[bcharge]:
            if kh.is_included(bpp, bp, bcharge):
                fct_bppbp = 0
                for a in statesdm[acharge]:
                    for l in range(nleads):
                        fct_bppbp += -0.5*tLba[l, a, b].conjugate()*tLba[l, a, bpp]
                for c in statesdm[ccharge]:
                    for l in range(nleads):
                        fct_bppbp += -0.5*tLba[l, c, b].conjugate()*tLba[l, c, bpp]
                # Lamb shift, first term of -1j*(HLS*phi0 - phi0*HLS)
                for l in range(nleads):
                    fct_bppbp += -1j*HLS[l, b, bpp]
                kh.set_matrix_element(1j*fct_bppbp, b, bp, bcharge, bpp, bp, bcharge)
            # --------------------------------------------------
            if kh.is_included(b, bpp, bcharge):
                fct_bbpp = 0
                for a in statesdm[acharge]:
                    for l in range(nleads):
                        fct_bbpp += -0.5*tLba[l, a, bpp].conjugate()*tLba[l, a, bp]
                for c in statesdm[ccharge]:
                    for l in range(nleads):
                        fct_bbpp += -0.5*tLba[l, c, bpp].conjugate()*tLba[l, c, bp]
                # Lamb shift, second term of -1j*(HLS*phi0 - phi0*HLS)
                for l in range(nleads):
                    fct_bbpp += 1j*HLS[l, bpp, bp]
                kh.set_matrix_element(1j*fct_bbpp, b, bp, bcharge, b, bpp, bcharge)
        # --------------------------------------------------
        for c, cp in itertools.product(statesdm[ccharge], statesdm[ccharge]):
            if kh.is_included(c, cp, ccharge):
                fct_ccp = 0
                for l in range(nleads):
                    fct_ccp += tLba[l, b, c]*tLba[l, bp, cp].conjugate()
                    if l in countingleads:
                        kh.set_matrix_element_lpm(1j*tLba[l, b, c]*tLba[l, bp, cp].conjugate(), 1, b, bp, bcharge, c, cp, ccharge)
                kh.set_matrix_element(1j*fct_ccp, b, bp, bcharge, c, cp, ccharge)
        # --------------------------------------------------

    def generate_current(self):
        self.generate_current_std()
        if not self.funcp.mfreeq:
            self.generate_current_noise()

    def generate_current_std(self):
        """
        Calculates currents using Lindblad approach.

        Parameters
        ----------
        current : array
            (Modifies) Values of the current having nleads entries.
        energy_current : array
            (Modifies) Values of the energy current having nleads entries.
        heat_current : array
            (Modifies) Values of the heat current having nleads entries.
        """
        phi0p, E, tLba, si = self.phi0, self.qd.Ea, self.tLba, self.si
        ndm0, npauli = si.ndm0, si.npauli
        ncharge, nleads, statesdm = si.ncharge, si.nleads, si.statesdm

        current = self.current
        energy_current = self.energy_current

        kh = self.kernel_handler
        for charge in range(ncharge):
            ccharge = charge+1
            bcharge = charge
            acharge = charge-1

            for b, bp in itertools.product(statesdm[bcharge], statesdm[bcharge]):
                if not kh.is_included(b, bp, bcharge):
                    continue
                phi0bbp = kh.get_phi0_element(b, bp, bcharge)

                for l in range(nleads):
                    current_l, energy_current_l = 0, 0

                    for a in statesdm[acharge]:
                        fcta = tLba[l, a, b]*phi0bbp*tLba[l, a, bp].conjugate()
                        current_l -= fcta
                        energy_current_l += (E[a]-0.5*(E[b]+E[bp]))*fcta
                    for c in statesdm[ccharge]:
                        fctc = tLba[l, c, b]*phi0bbp*tLba[l, c, bp].conjugate()
                        current_l += fctc
                        energy_current_l += (E[c]-0.5*(E[b]+E[bp]))*fctc

                    current[l] += current_l.real
                    energy_current[l] += energy_current_l.real

        self.heat_current[:] = energy_current - current*self.leads.mulst

    def generate_current_noise(self): #simon
        """
        Calculates currents using Pauli master equation approach and noise via the C.Emary approach summed over countingleads passed

        Returns
        ----------
        current : float
            Value of the current attaching the counting field to countingleads.
        noise : array
            Value of the current noise attaching the counting field to countingleads.
        """
        phi0, E, si = self.phi0, self.qd.Ea, self.si
        nleads = si.nleads
        ndm0r, npauli = si.ndm0r, si.npauli
        kern, Lpm = self.kern, self.Lpm
        Lp, Lm = self.Lpm

        # auxilliary quantities
        # right eigenvector
        P = phi0[...,None]
        # left eigenvector
        O = np.zeros(ndm0r)[None,...]
        O[0,:npauli].fill(1.0)

        # projector
        Q = (np.eye(np.size(P)) - P @ O)
        # pseudoinverse
        R = Q @ np.linalg.pinv(kern[:P.size, :P.size]) @ Q

        # current and noise
        Jp  = 1j*Lp - 1j*Lm
        Jpp = -Lp - Lm
        c = -1j*(O @ Jp @ P)
        s = -O @ (Jpp - 2*(Jp @ R @ Jp)) @ P
        self.current_noise[0] = c.real.item()
        self.current_noise[1] = s.real.item()
# ---------------------------------------------------------------------------------------------------
