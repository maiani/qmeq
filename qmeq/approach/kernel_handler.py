"""Module containing python functions, which generate second order RTD kernels."""

from __future__ import annotations

from enum import IntEnum

import numpy as np

from ..indexing import StateIndexingDM
from ..indexing import StateIndexingDMc
from . import dm_layout
from .dm_layout import NO_INDEX


class KernelHandler(object):
    """Class responsible for inserting matrix elements into the various matrices used.

    The packed real layout these methods write into is specified in
    :mod:`qmeq.approach.dm_layout`; the rule names cited below refer to it.

    Parameters
    ----------
    si : StateIndexingDM
        State indexing for the system, from :mod:`qmeq.indexing`. Supplies the
        sizes ``ndm0``, ``ndm0r`` and ``npauli`` and the ``dm0`` index maps
        reached through ``get_ind_dm0``, ``get_ind_dm0_bool`` and
        ``get_ind_dm0_conj``. The 2vN approaches pass a ``StateIndexingDMc``
        instead; it provides the same sizes but carries no conjugation map, and
        the insertion methods below are not used on that path.
    """
    def __init__(self, si: StateIndexingDM | StateIndexingDMc) -> None:
        self.si = si
        self.ndm0 = si.ndm0
        self.ndm0r = si.ndm0r
        self.npauli = si.npauli
        # Distance from a reduced index to its imaginary partner (L4).
        self.imag_offset = si.ndm0 - si.npauli
        self.phi0 = None
        self.kern = None

    def set_kern(self, kern: np.ndarray) -> None:
        self.kern = kern

    def set_phi0(self, phi0: np.ndarray) -> None:
        self.phi0 = phi0

    def is_included(self, b: int, bp: int, bcharge: int) -> bool:
        """ Checks if the density matrix entry :math:`|b><bp|` is included in the calculations.

        Parameters
        ----------
        b : int
            first state
        bp : int
            second state
        bcharge : int
            charge of the states b and bp

        Returns
        -------
        bool
            true if it's included
        """
        bbp = self.si.get_ind_dm0(b, bp, bcharge)
        if bbp == NO_INDEX:
            return False

        return True

    def is_unique(self, b: int, bp: int, bcharge: int) -> bool:
        """ Check if the entry :math:`|b><bp|` is unique.

        Parameters
        ----------
        b  : int
            first state
        bp : int
            second state
        bcharge : int
            charge of the states b and bp

        Returns
        -------
        bool
            true if unique
        """
        bbp_bool = self.si.get_ind_dm0_bool(b, bp, bcharge)
        return bbp_bool

    def set_energy(self, energy: float, b: int, bp: int, bcharge: int) -> None:
        bbp = self.si.get_ind_dm0(b, bp, bcharge)
        bbpi = bbp + self.imag_offset
        bbpi_bool = bbp >= self.npauli

        if bbpi_bool:
            self.kern[bbp, bbpi] = self.kern[bbp, bbpi] + energy
            self.kern[bbpi, bbp] = self.kern[bbpi, bbp] - energy

    def set_matrix_element(self, fct: complex, b: int, bp: int, bcharge: int,
                           a: int, ap: int, acharge: int) -> None:
        """ Adds a complex value to the matrix element connecting :math:`|a><ap|` and :math:`|b><bp|` in the kernel.

        Parameters
        ----------
        fct : complex
            value to be added
        b : int
            first state of :math:`|b><bp|`
        bp : int
            second state of :math:`|b><bp|`
        bcharge : int
            charge of states b and bp
        a : int
            first state of :math:`|a><ap|`
        ap : int
            second state of :math:`|a><ap|`
        acharge : int
            charge of the states a and ap
        self.kern : ndarray
            (modifies) the kernel
        """
        bbp = self.si.get_ind_dm0(b, bp, bcharge)
        bbpi = bbp + self.imag_offset
        bbpi_bool = bbp >= self.npauli

        aap = self.si.get_ind_dm0(a, ap, acharge)
        if bbp == NO_INDEX or aap == NO_INDEX:
            # L2: an endpoint with no index has nowhere to contribute. Callers
            # all guard with is_included; without this the sentinel would index
            # the last row or column and silently corrupt an unrelated entry.
            if dm_layout.STRICT_INDEX:
                raise dm_layout.no_index_error(
                    'set_matrix_element', {'bbp': bbp, 'aap': aap})
            return
        aapi = aap + self.imag_offset
        aap_sgn = +1 if self.si.get_ind_dm0_conj(a, ap, acharge) else -1

        fct_imag = fct.imag
        fct_real = fct.real

        self.kern[bbp, aap] += fct_imag
        if aap >= self.npauli:
            self.kern[bbp, aapi] += fct_real*aap_sgn
            if bbpi_bool:
                self.kern[bbpi, aapi] += fct_imag*aap_sgn
        if bbpi_bool:
            self.kern[bbpi, aap] += -fct_real

    def set_matrix_element_pauli(self, fctm: float, fctp: float, bb: int, aa: int) -> None:
        """ Adds a real value (fctp) to the the matrix element connecting the states
        bb and aa in the Pauli kernel. In addition, adds another another real value (fctm)
        to the diagonal kern[bb, bb].

        Parameters
        ----------
        fctm : double
            value to be added to kern[bb, aa]
        fctp : double
            value to be added to kern[bb, bb]
        bb : int
            first state/index
        aa : int
            second state/index
        self.kern : ndarray
            (modifies) the kernel
        """
        self.kern[bb, bb] += fctm
        self.kern[bb, aa] += fctp

    def get_phi0_element(self, b: int, bp: int, bcharge: int) -> complex:
        r""" Gets the entry of the density matrix given by :math:`|b><bp|`.

        Parameters
        ----------
        b : int
            first state
        bp : int
            second state
        bcharge : int
            charge of the states b and bp

        Returns
        -------
        complex
            the value :math:`<b|\phi_0|bp>`
        """
        bbp = self.si.get_ind_dm0(b, bp, bcharge)
        if bbp == NO_INDEX:
            return 0.0

        bbpi = bbp + self.imag_offset
        bbpi_bool = bbp >= self.npauli

        phi0_real = self.phi0[bbp]
        phi0_imag = 0
        if bbpi_bool:
            bbp_conj = self.si.get_ind_dm0_conj(b, bp, bcharge)
            phi0_imag = self.phi0[bbpi] if bbp_conj else -self.phi0[bbpi]

        return phi0_real + 1j*phi0_imag

class KernelHandlerNoise(KernelHandler):
    """Class used for inserting matrix elements into the matrices used in the first order counting statistics approaches."""

    def __init__(self, si):
        KernelHandler.__init__(self, si)
        self.Lpm = None

    def set_lpm(self, Lpm: np.ndarray) -> None:
        self.Lpm = Lpm

    def set_matrix_element_lpm_pauli(self,pfct,pm,bb,aa):
        """ Adds a real value (fctp) to the the matrix element connecting the states
        bb and aa with counting index pm in the in the counting field dependend Pauli kernel.

        Parameters
        ----------
        pfct : double
            value to be added to kern[bb, aa]
        pm : int
            counting index
        bb : int
            first state/index
        aa : int
            second state/index
        self.Lpm : ndarray
            (modifies) the counting kernel
        """
        self.Lpm[pm,bb,aa] += pfct

    def set_matrix_element_lpm(self, fct, pm, b, bp, bcharge, a, ap, acharge):
        """ Adds a complex value to the matrix element connecting :math:`|a><ap|` and :math:`|b><bp|` with counting index pm in the counting field resolved kernel.

        Parameters
        ----------
        fct : complex
            value to be added
        pm : int
            counting index
        b : int
            first state of :math:`|b><bp|`
        bp : int
            second state of :math:`|b><bp|`
        bcharge : int
            charge of states b and bp
        a : int
            first state of :math:`|a><ap|`
        ap : int
            second state of :math:`|a><ap|`
        acharge : int
            charge of the states a and ap
        self.Lpm : ndarray
            (modifies) the counting kernel
        """
        bbp = self.si.get_ind_dm0(b, bp, bcharge)
        bbpi = bbp + self.imag_offset
        bbpi_bool = bbp >= self.npauli

        aap = self.si.get_ind_dm0(a, ap, acharge)
        if bbp == NO_INDEX or aap == NO_INDEX:
            # L2: an endpoint with no index has nowhere to contribute. Callers
            # all guard with is_included; without this the sentinel would index
            # the last row or column and silently corrupt an unrelated entry.
            if dm_layout.STRICT_INDEX:
                raise dm_layout.no_index_error(
                    'set_matrix_element', {'bbp': bbp, 'aap': aap})
            return
        aapi = aap + self.imag_offset
        aap_sgn = +1 if self.si.get_ind_dm0_conj(a, ap, acharge) else -1

        fct_imag = fct.imag
        fct_real = fct.real

        self.Lpm[pm,bbp, aap] += fct_imag
        if aap >= self.npauli:
            self.Lpm[pm,bbp, aapi] += fct_real*aap_sgn
            if bbpi_bool:
                self.Lpm[pm,bbpi, aapi] += fct_imag*aap_sgn
        if bbpi_bool:
            self.Lpm[pm,bbpi, aap] += -fct_real

class KernelHandlerMatrixFree(KernelHandler):
    """Class used for inserting matrix elements into vectors when using the matrix free
        solution method."""

    def __init__(self, si):
        KernelHandler.__init__(self, si)
        self.dphi0_dt = None

    def set_dphi0_dt(self, dphi0_dt: np.ndarray) -> None:
        self.dphi0_dt = dphi0_dt

    def set_energy(self, energy: float, b: int, bp: int, bcharge: int) -> None:
        if b == bp:
            return

        bbp = self.si.get_ind_dm0(b, bp, bcharge)
        if bbp == NO_INDEX:
            if dm_layout.STRICT_INDEX:
                raise dm_layout.no_index_error('set_energy', {'bbp': bbp})
            return
        bbpi = bbp + self.imag_offset

        phi0bbp = self.get_phi0_element(b, bp, bcharge)
        dphi0_dt_bbp = -1j*energy*phi0bbp

        self.dphi0_dt[bbp] += dphi0_dt_bbp.real
        self.dphi0_dt[bbpi] -= dphi0_dt_bbp.imag

    def set_matrix_element(self, fct: complex, b: int, bp: int, bcharge: int,
                           a: int, ap: int, acharge: int) -> None:
        r""" Adds a contribution to :math:`d\phi_o /dt` that stems from the matrix element
        connecting :math:`|b><bp|` and :math:`|a><ap|` in the full off-diagonal in the kernel.

        Parameters
        ----------
        fct : complex
            value to be added
        b : int
            first state of :math:`|b><bp|`
        bp : int
            second state of :math:`|b><bp|`
        bcharge : int
            charge for the states b and bp
        a : int
            first state of :math:`|a><ap|`
        ap : int
            second state of :math:`|a><ap|`
        acharge : int
            charge of the states a and ap
        self.dphi0_dt : ndarray
            (modifies) time derivative of the density matrix
        """
        bbp = self.si.get_ind_dm0(b, bp, bcharge)
        if bbp == NO_INDEX:
            if dm_layout.STRICT_INDEX:
                raise dm_layout.no_index_error(
                    'set_matrix_element', {'bbp': bbp})
            return
        bbpi = bbp + self.imag_offset
        bbpi_bool = bbp >= self.npauli
        phi0aap = self.get_phi0_element(a, ap, acharge)
        dphi0_dt_bbp = -1j*fct*phi0aap

        self.dphi0_dt[bbp] += dphi0_dt_bbp.real
        if bbpi_bool:
            self.dphi0_dt[bbpi] -= dphi0_dt_bbp.imag

    def set_matrix_element_pauli(self, fctm: float, fctp: float, bb: int, aa: int) -> None:
        r""" Adds a contribution to :math:`d\phi_o /dt` that stems from the matrix element
        connecting :math:`|b><b|` and :math:`|a><a|` in the Pauli kernel.

        Parameters
        ----------
        fctm : double
            value from the diagonal of the kernel kern[bb, bb]
        fctp : double
            value from the off-diagonal of the kernel kern[bb, aa]
        b : int
            first state
        a : int
            second state
        self.dphi0_dt : ndarray
            (modifies) time derivative of the density matrix
        """
        self.dphi0_dt[bb] += fctm*self.phi0[bb] + fctp*self.phi0[aa]

    def get_phi0_norm(self) -> float:
        ncharge, statesdm = self.si.ncharge, self.si.statesdm

        norm = 0.0
        for bcharge in range(ncharge):
            for b in statesdm[bcharge]:
                bb = self.si.get_ind_dm0(b, b, bcharge)
                norm += self.phi0[bb]

        return norm

class RtdMatrix(IntEnum):
    """Selects which RTD matrix an insertion writes into.

    Member names match the array attributes they select, so
    ``RtdMatrix.ReWdn`` writes into ``ReWdn``. The values are the historical
    integers and must not be reordered: they index
    :meth:`KernelHandlerRTD.set_matrix_list` and are mirrored by a ``cdef enum``
    in ``c_kernel_handler.pxd`` for the compiled path.

    ``Lnn_inv`` is accepted by ``add_matrix_element`` only in the compiled
    backend. The pure-Python approach routes it through the dedicated
    :meth:`KernelHandlerRTD.add_element_Lnn_inv`, because the pure-Python array
    is two-dimensional while the compiled one is a bare diagonal.
    """

    Wdd = 0
    WE1 = 1
    WE2 = 2
    ReWdn = 3
    ImWdn = 4
    ReWnd = 5
    ImWnd = 6
    Lnn_inv = 7


class KernelHandlerRTD(KernelHandler):
    """Class used for inserting matrix elements into the matrices used in the RTD approach."""

    def set_matrix_list(self) -> None:
        self.mats = [
            self.Wdd, self.WE1, self.WE2,
            getattr(self, "ReWdn", None), getattr(self, "ImWdn", None),
            getattr(self, "ReWnd", None), getattr(self, "ImWnd", None),
            getattr(self, "Lnn_inv", None),
        ]

    def add_matrix_element(self, fct, l, b, bp, bcharge, a, ap, acharge, mi):
        r"""
        Adds a value to the lead-resolved ndarray (kernel) given by index mi. The indices are set by the entries
        :math:`|b><bp|` and :math:`|a><ap|` in the density matrix. Which matrix to add the value to is
        determined by mi, a :class:`RtdMatrix` member.

        Parameters
        ----------
        fct : float
            the value to be added
        l : int
            lead index
        b : int
            first index for state 1
        bp : int
            second index for state 1
        bcharge : int
            charge of state 1
        a : int
            first index for state 2
        ap : int
            second index for state 2
        acharge : int
            charge of state 2
        mi : RtdMatrix
            selects which matrix to insert into
        self.mats[mi] : ndarray
            (Modifies) the matrix selected by mi
        """


        self.add_matrix_element_to(
            self.mats[mi], fct, l, b, bp, bcharge, a, ap, acharge
        )

    def add_matrix_element_to(
            self, matrix: np.ndarray, fct: float | complex,
            l: int, b: int, bp: int, bcharge: int,
            a: int, ap: int, acharge: int) -> None:
        """Insert into an explicit RTD block using the canonical packed layout.

        This is the composable counterpart of :meth:`add_matrix_element` for
        block families that are not part of the historical ``RtdMatrix`` enum,
        such as Laplace derivatives of the population-coherence blocks.
        """
        indx1 = self.si.get_ind_dm0(b, bp, bcharge)
        indx2 = self.si.get_ind_dm0(a, ap, acharge)
        if b != bp:
            indx1 -= self.npauli
            if b > bp:
                indx1 += self.imag_offset
        if a != ap:
            indx2 -= self.npauli
            if a > ap:
                indx2 += self.imag_offset

        matrix[l, indx1, indx2] += fct

    def set_matrix_element_dd(self, l, fctm, fctp, bb, aa, mi):
        """
        Adds a value to the lead-resolved kernel connecting :math:`|b><b|` to :math:`|a><a|`,
        and uses the conservation law to add a second value to the diagonal (connecting :math:`|b><b|`
        to itself).

        Parameters
        ----------
        l : int
            lead index
        fctm : float
            value to be added to the diagonal (tunneling out)
        fctp : float
            value to be added to the off-diagonal (tunneling in)
        bb : int
            index for the entry :math:`|b><b|`
        aa :  int
            index for the entry :math:`|a><a|`
        mi : RtdMatrix
            selects which matrix to insert into
        self.mats[mi] : ndarray
            (Modifies) the matrix selected by mi
        """
        mat = self.mats[mi]
        mat[l, bb, bb] += fctm
        mat[l, bb, aa] += fctp

    def add_element_2nd_order(self, r, fct, indx0, indx1, a3, charge3, a4, charge4):
        """
        Adds a value to the lead-resolved kernel for the diagonal density matrix. Uses symmetries
        between second order diagrams in the RTD approach to add the value to four places in the matrix.


        Parameters
        ----------
        r : int
            lead index
        fct : float
            value to be added
        indx0 : int
            index for inital state
        indx1 : int
            index for intermidiate state 1
        a3 : int
            intermediate state 3 is given by :math:`|a3><a3|`
        charge3 : int
            charge of intermediate state 3
        a4 : int
            final state is given by :math:`|a4><a4|`
        charge4 : int
            charge of the final state
        self.Wdd : ndarray
            (Modifies) the lead-resolved kernel for the diagonal density matrix.

        """
        si = self.si
        indx3 = si.get_ind_dm0(a3, a3, charge3)
        indx4 = si.get_ind_dm0(a4, a4, charge4)

        fct = 2 * fct
        self.Wdd[r, indx4, indx0] += fct
        # Flipping left-most vertex p3 = -p3
        self.Wdd[r, indx3, indx0] += -fct
        # Flipping right-most vertex p0 = -p0
        self.Wdd[r, indx4, indx1] += fct
        # Flipping left-most and right-most vertices p0 = -p0 and p3 = -p3
        self.Wdd[r, indx3, indx1] += -fct

    def add_element_Lnn_inv(self, a1, b1, charge, fct):
        """
        Adds a value to the part of :math:`L_{N,+}` connecting an off-diagonal component of the density matrix to
        itself.

        Parameters
        ----------
        a1 : int
            first part of the component :math:`|a1><b1|`
        b1 :  int
            second part of the component :math:`|a1><b1|`
        charge : int
            charge of the states a1 and b1
        fct : float
            the value to be added
        self.Lnn_inv : ndarray
            (Modifies) the anti-commutator Liouvillian connecting non-diagonal elements
        """
        indx = self.si.get_ind_dm0(a1, b1, charge) - self.npauli
        if a1 > b1:
            indx += self.imag_offset
        self.Lnn_inv[indx, indx] += fct

class KernelHandlerRTDnoise(KernelHandlerNoise, KernelHandlerRTD):
    """Class used for inserting matrix elements into the matrices used in the RTD noise approach."""

    def set_matrix_element_lpm_first(self,l,pfct,pfct_dz,pm,bb,aa):
        """ Adds a kernel value and its Laplace derivative to the matrix element connecting the states
        bb and aa with counting index pm in the in the counting field dependend first order kernels.

        Parameters
        ----------
        pfct : double
            value to be added to Lpm_first[l,pm,bb,aa]
        pfct_dz : double
            Laplace derivative to be added to Lpm_first_dz[l,pm,bb,aa]
        pm : int
            counting index
        bb : int
            first state/index
        aa : int
            second state/index
        self.Lpm_first : ndarray
            (modifies) the first order counting kernel
        self.Lpm_first_dz : ndarray
            (modifies) the Laplace derivatives of the first order counting kernel
        """

        self.Lpm_first[l,pm,bb,aa] += pfct
        self.Lpm_first_dz[l,pm,bb,aa] += pfct_dz

    def add_element_2nd_order(self, r0, r1, eta0, eta1, p1, p2, fct,
                              fct_dz, indx0, indx1, a3, charge3, a4,
                              charge4, dx):
        """Add one independent second-order population contribution.

        Only the ``eta0 = +1`` half of the traversal is independent.  The
        approach completes its ``eta0 = -1`` conjugate partners after all
        columns have been generated.  Keeping that completion at the assembled
        array boundary makes the complex-conjugation and Laplace-derivative
        signs explicit instead of reconstructing them vertex by vertex.

        Parameters
        ----------
        r0 : int
            lead index 0
        r1 : int
            lead index 1
        eta0 : int
            electron-hole index (note: different sign convention in emary, i.e. eta=-xi)
        eta1 : int
            electron-hole index (note: different sign convention in emary, i.e. eta=-xi)
        p1 : int
            keldysh index
        p2 : int
            keldysh index
        fct : float
            value to be added
        fct_dz : complex
            Laplace derivative, represented by a common negative shift of all
            three propagator energies
        indx0 : int
            index for inital state
        indx1 : int
            index for intermidiate state 1
        a3 : int
            intermediate state 3 is given by :math:`|a3><a3|`
        charge3 : int
            charge of intermediate state 3
        a4 : int
            final state is given by :math:`|a4><a4|`
        charge4 : int
            charge of the final state
        dx : string
            indicates if direct or exchange integral
        self.Wdd : ndarray
            (Modifies) the lead-resolved kernel for the diagonal density matrix.
        """
        si = self.si
        indx3 = si.get_ind_dm0(a3, a3, charge3)
        indx4 = si.get_ind_dm0(a4, a4, charge4)

        # calculate counting indices
        if dx == 'd': # eta0 * (p0 - p3)/2 , eta1 * (p1 - p2)/2
            cind0 = eta0 * (1 - 1)//2 , eta1 * (p1 - p2)//2 # p0=1,p3=1
            cind1 = eta0 * (1 + 1)//2 , eta1 * (p1 - p2)//2 # p0=1,p3=-1
            cind2 = eta0 * (-1 - 1)//2 , eta1 * (p1 - p2)//2 # p0=-1,p3=1
            cind3 = eta0 * (-1 + 1)//2 , eta1 * (p1 - p2)//2 # p0=-1,p3=-1
        elif dx == 'x': # eta1 * (p1 - p3)/2 + eta0 * (p0 - p2)/2
            cind0 = eta0 * (1 - p2)//2 , eta1 * (p1 - 1)//2 # p0=1,p3=1
            cind1 = eta0 * (1 - p2)//2 , eta1 * (p1 + 1)//2 # p0=1,p3=-1
            cind2 = eta0 * (-1 - p2)//2 , eta1 * (p1 - 1)//2 # p0=-1,p3=1
            cind3 = eta0 * (-1 - p2)//2 , eta1 * (p1 + 1)//2 # p0=-1,p3=-1

        # add kernel elements
        self.Lpm_second[r0,r1,cind0[0],cind0[1], indx4, indx0] += fct
        # Flipping left-most vertex p3 = -p3
        self.Lpm_second[r0,r1,cind1[0],cind1[1], indx3, indx0] += -fct
        # Flipping right-most vertex p0 = -p0
        self.Lpm_second[r0,r1,cind2[0],cind2[1], indx4, indx1] += fct
        # Flipping left-most and right-most vertices p0 = -p0 and p3 = -p3
        self.Lpm_second[r0,r1,cind3[0],cind3[1], indx3, indx1] += -fct

        # add derivatives
        self.Lpm_second_dz[r0,r1,cind0[0],cind0[1], indx4, indx0] += fct_dz
        # Flipping left-most vertex p3 = -p3
        self.Lpm_second_dz[r0,r1,cind1[0],cind1[1], indx3, indx0] += -fct_dz
        # Flipping right-most vertex p0 = -p0
        self.Lpm_second_dz[r0,r1,cind2[0],cind2[1], indx4, indx1] += fct_dz
        # Flipping left-most and right-most vertices p0 = -p0 and p3 = -p3
        self.Lpm_second_dz[r0,r1,cind3[0],cind3[1], indx3, indx1] += -fct_dz

        # for std currents
        if dx == 'd':
            self.Wdd[r0, indx4, indx0] += fct.real
            # Flipping left-most vertex p3 = -p3
            self.Wdd[r0, indx3, indx0] += -fct.real
            # Flipping right-most vertex p0 = -p0
            self.Wdd[r0, indx4, indx1] += fct.real
            # Flipping left-most and right-most vertices p0 = -p0 and p3 = -p3
            self.Wdd[r0, indx3, indx1] += -fct.real
        elif dx == 'x':
            self.Wdd[r1, indx4, indx0] += fct.real
            # Flipping left-most vertex p3 = -p3
            self.Wdd[r1, indx3, indx0] += -fct.real
            # Flipping right-most vertex p0 = -p0
            self.Wdd[r1, indx4, indx1] += fct.real
            # Flipping left-most and right-most vertices p0 = -p0 and p3 = -p3
            self.Wdd[r1, indx3, indx1] += -fct.real
