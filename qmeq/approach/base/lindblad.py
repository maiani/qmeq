"""Module containing python functions, which generate first order Lindblad kernels."""

import numpy as np
import itertools

from ...specfunc.specfunc import func_pauli
from ...specfunc.specfunc import func_lambshift
from ...specfunc.specfunc import func_ule_shift
from ...specfunc.specfunc import func_lambshift_quad
from ..aprclass import Approach

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

        (H_{LS}^{l})_{bb'} = \sum_{a}T^{l}_{ba}T^{l}_{ab'}\,
                             w_{l}\!\left(x^{a}_{b},x^{a}_{b'}\right)
                           + \sum_{c}T^{l}_{bc}T^{l}_{cb'}\,
                             \tilde{w}_{l}\!\left(y^{c}_{b},
                                                  y^{c}_{b'}\right),

    where :math:`H_{LS}=\sum_{l}H_{LS}^{l}`, the states :math:`a` (:math:`c`) have one
    electron less (more) than the states :math:`b`, :math:`b'`, the scaled transition
    energies are :math:`x^a_b=(E_b-E_a-\mu_l)/T_l` and
    :math:`y^c_b=(E_c-E_b-\mu_l)/T_l`. With
    :math:`S(x)=\operatorname{Re}\psi(1/2+ix/(2\pi))`, the weights are

    .. math::

        w_l(x_1,x_2) &= [S(x_1)+S(x_2)]/2+\delta_f(-x_1,-x_2),\\
        \tilde w_l(y_1,y_2) &= [S(y_1)+S(y_2)]/2+\delta_f(y_1,y_2).

    Here ``delta_f`` is :func:`~qmeq.specfunc.specfunc.func_ule_shift`.
    The N-1 intermediate describes emission into an empty lead state;
    the N+1 intermediate describes absorption from an occupied lead state.
    Their direct weights are respectively ``+P int sqrt(h*h)/v`` and
    ``-P int sqrt(f*f)/v``. Reversing v in the first gives the negative
    arguments above, INCLUDING the sign of mu. The evenness of S does not
    extend to delta_f. Both have the same negative bandwidth logarithm,
    which multiplies the fermionic anticommutator and hence the identity.

    Use [NathanRudner2020, Eq. (D7)] and the CORRECTED D8,
    [NathanRudner2021Erratum, Eq. (6)]: for outer b,b' and intermediate k,
    ``p=E_k-E_b, q=E_b'-E_k``. The original D8 and main-text Eq. (34)
    contain misprints. Hermiticity and equal-argument tests alone do not
    determine the off-diagonal weights; direct-integral tests cover both
    intermediate charge sectors separately. See ``theory/lambshift.md``.

    ``principal_part='digamma'`` evaluates the wide-band weights above.
    ``'quad'`` retains finite-band arithmetic principal values but still uses
    the wide-band geometric correction; it is not an exact finite-band ULE.
    ``'omit'`` drops the shift. ``itype`` controls the dissipative rates.

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
    mulst, tlst, dlst = appr.leads.mulst, appr.leads.tlst, appr.leads.dlst
    ncharge, nleads, statesdm = si.ncharge, si.nleads, si.statesdm
    mode, limit = appr.funcp.principal_part, appr.funcp.dqawc_limit

    if mode not in ("digamma", "quad"):
        return

    def principal(energy, chemical_potential, temperature, lead):
        """``Lambda`` in the selected evaluation."""
        if mode == "quad":
            return func_lambshift_quad(
                energy, chemical_potential, temperature,
                dlst[lead, 0], dlst[lead, 1], limit,
            )
        return func_lambshift(energy, chemical_potential, temperature)

    HLS = appr.HLS
    for bcharge in range(ncharge):
        acharge = bcharge-1
        ccharge = bcharge+1
        for b, bp in itertools.combinations_with_replacement(statesdm[bcharge], 2):
            for l in range(nleads):
                mu, T = mulst[l], tlst[l]
                fct = 0
                for a in statesdm[acharge]:
                    # Lower intermediate: emission into an empty lead state.
                    # [NathanRudner2021Erratum, Eq. (6)]: delta_f(-x1, -x2).
                    weight = 0.5*(principal(E[b]-E[a], mu, T, l)
                                  + principal(E[bp]-E[a], mu, T, l)) \
                             + func_ule_shift((E[a]-E[b]+mu)/T,
                                              (E[a]-E[bp]+mu)/T)
                    fct += Tba[l, b, a]*Tba[l, a, bp]*weight
                for c in statesdm[ccharge]:
                    # Upper intermediate: absorption from an occupied lead state.
                    # [NathanRudner2021Erratum, Eq. (6)]: delta_f(y1, y2).
                    weight = 0.5*(principal(E[b]-E[c], -mu, T, l)
                                  + principal(E[bp]-E[c], -mu, T, l)) \
                             + func_ule_shift((E[c]-E[b]-mu)/T,
                                              (E[c]-E[bp]-mu)/T)
                    fct += Tba[l, b, c]*Tba[l, c, bp]*weight
                HLS[l, b, bp] = fct
                HLS[l, bp, b] = fct.conjugate()


# ---------------------------------------------------------------------------------------------------
# Lindblad approach
# ---------------------------------------------------------------------------------------------------
class ApproachLindblad(Approach):

    kerntype = 'pyLindblad'

    def prepare_arrays(self):
        Approach.prepare_arrays(self)
        Tba, mtype = self.leads.Tba, self.leads.mtype
        self.tLba = np.zeros(Tba.shape, dtype=mtype)
        self.HLS = np.zeros(Tba.shape, dtype=mtype)

    def clean_arrays(self):
        Approach.clean_arrays(self)
        self.tLba.fill(0.0)
        self.HLS.fill(0.0)

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
        acharge = bcharge-1
        ccharge = bcharge+1

        for a, ap in itertools.product(statesdm[acharge], statesdm[acharge]):
            if kh.is_included(a, ap, acharge):
                fct_aap = 0
                for l in range(nleads):
                    fct_aap += tLba[l, b, a]*tLba[l, bp, ap].conjugate()
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
                kh.set_matrix_element(1j*fct_ccp, b, bp, bcharge, c, cp, ccharge)
        # --------------------------------------------------

    def generate_current(self):
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
        E, tLba, si = self.qd.Ea, self.tLba, self.si
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

# ---------------------------------------------------------------------------------------------------
