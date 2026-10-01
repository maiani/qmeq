"""RTD zero-frequency current and noise approaches."""

from itertools import product
import warnings

import numpy as np

from ..._warnings import QmeqRuntimeWarning
from ...wrappers.mytypes import doublenp
from ...wrappers.mytypes import complexnp

# The selected aliases resolve to compiled scalar wrappers under the Cython
# backend and to the canonical Python functions otherwise.  The explicit
# Python imports keep the documented ``pyRTDnoise`` backend all-Python.
from ...specfunc import c_integralD_lpm as selected_integralD_lpm
from ...specfunc import (
    c_integralD_lpm_derivative as selected_integralD_lpm_derivative,
)
from ...specfunc import c_integralX_lpm as selected_integralX_lpm
from ...specfunc import (
    c_integralX_lpm_derivative as selected_integralX_lpm_derivative,
)
from ...specfunc.specfunc import func_pauli
from ...specfunc.specfunc import diff_phi
from ...specfunc.specfunc import integralD_lpm as python_integralD_lpm
from ...specfunc.specfunc import (
    integralD_lpm_derivative as python_integralD_lpm_derivative,
)
from ...specfunc.specfunc import integralX_lpm as python_integralX_lpm
from ...specfunc.specfunc import (
    integralX_lpm_derivative as python_integralX_lpm_derivative,
)
from .RTD import ApproachPyRTD
from .RTD import _warn_if_rtd_coherence_is_not_resolved
from .RTD import _warn_if_unequal_temperature_cutoff_is_small
from ..counting import nonmarkovian_current_noise_matrix
from ..counting import stationary_kernel_pseudoinverse
from ..counting import stationary_projected_pseudoinverse
from ..diagnostics import check_stationary_solution
from ..kernel_handler import KernelHandlerRTDnoise
from ..rtd_diagrams import DIRECT
from ..rtd_diagrams import GAIN
from ..rtd_diagrams import first_order_diagrams
from ..rtd_diagrams import second_order_diagrams
from ..rtd_blocks import counting_resolved_coherence_correction
from ..rtd_blocks import generate_population_coherence_blocks


class RTDNoiseLaplaceProjectionWarning(QmeqRuntimeWarning):
    """Warning that a discarded Laplace-derivative real part was not roundoff."""


#: Largest ``max|Re| / max|Im|`` accepted as roundoff when projecting the
#: Laplace derivative onto its analytically allowed imaginary channel.
#: Calibrated against the test suite: with real tunnel amplitudes the observed
#: ratio never exceeds 1.2e-5.  Before conjugate-partner completion was fixed,
#: complex amplitudes produced ratios of 0.30 to 0.95; the completed traversal
#: is analytically imaginary for them too.  A ratio above this bound now flags
#: a partner or derivative regression rather than a supported physical channel.
_LAPLACE_REAL_PROJECTION_RTOL = 1e-3


def _project_laplace_derivative_onto_imaginary(approach, name, array):
    """Zero the analytically forbidden real channel, but never silently.

    In QmeQ's RTD convention the zero-field kernel is real and its Laplace
    derivative is purely imaginary, so a nonzero real part is numerical
    residue from centered differences of individually complex diagrams.  That
    holds only where the diagram sum is qualified: with complex tunnel
    amplitudes the discarded part reaches the same order as the physical
    imaginary channel, which means the counting construction -- not the
    projection -- is the thing that is wrong.  Warn there instead of
    presenting a projected result as if it were clean.
    """
    real_scale = float(np.abs(array.real).max()) if array.size else 0.0
    imaginary_scale = float(np.abs(array.imag).max()) if array.size else 0.0
    if real_scale > _LAPLACE_REAL_PROJECTION_RTOL*max(imaginary_scale, 1e-300):
        warnings.warn(
            f"RTDnoise discarded a real part of {name} that is not roundoff: "
            f"max|Re| = {real_scale:.3e} against max|Im| = {imaginary_scale:.3e} "
            f"(ratio {real_scale/max(imaginary_scale, 1e-300):.3e}, allowed "
            f"{_LAPLACE_REAL_PROJECTION_RTOL:.0e}). The Laplace derivative is "
            "analytically imaginary, so this indicates a conjugate-partner or "
            "Laplace-derivative inconsistency. The returned noise is not "
            "trustworthy here.",
            RTDNoiseLaplaceProjectionWarning,
        )
    array[:] = 1j*array.imag


class ApproachPyRTDnoise(ApproachPyRTD):
    """Counting-resolved RTD kernel and its non-Markovian noise.

    The diagram traversal, kernel assembly, and scalar direct/exchange
    special-function calls are all implemented in Python.  The unprefixed
    :class:`ApproachRTDnoise` subclass may select compiled scalar functions.

    Laplace-derivative convention.  ``phi`` and the Fermi function take the
    *scaled* argument ``(E - mu)/T``, so a derivative with respect to the
    Laplace energy ``z`` carries a factor ``1/T_lead``.  Omitting it leaves a
    quantity with the dimension of an energy where a dimensionless one is
    required, and the reported non-Markovian noise then stops being covariant
    under an overall rescaling of ``E``, ``mu``, ``T`` and ``Gamma`` -- an
    error that is invisible at ``T = 1`` and puts an ``O(Gamma**2)`` term back
    into the noise everywhere else.  Pinned by
    ``test_noise_is_covariant_under_an_overall_energy_rescaling`` and by the
    unequal-temperature non-interacting gate.
    """

    kerntype = 'pyRTDnoise'
    coherence_laplace_derivatives = True
    integralD_lpm = staticmethod(python_integralD_lpm)
    integralD_lpm_derivative = staticmethod(python_integralD_lpm_derivative)
    integralX_lpm = staticmethod(python_integralX_lpm)
    integralX_lpm_derivative = staticmethod(python_integralX_lpm_derivative)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def restart(self):
        ApproachPyRTD.restart(self)
        # to separate kernel
        self.Lpm = None # temporary
        self.Lpm_first = None
        self.Lpm_first_dz = None
        self.Lpm_second = None
        self.Lpm_second_dz = None
        # for first order results at the smae time
        self.phi0_first = None
        self.kern_first = None
        # for second order results
        self.phi0_second = None
        self.kern_second = None
        # results
        self.current_noise = None
        self.current_noise_first = None
        self.current_noise_o4trunc = None
        self.current_noise_matrix = None
        self.current_noise_matrix_first = None
        self.coherence_correction = None
        self.coherence_correction_dz = None

        self.lpm_imaginary_2nd = None # temporary

    def prepare_kernel_handler(self):
        self.kernel_handler = KernelHandlerRTDnoise(self.si)

    def prepare_arrays(self):
        ApproachPyRTD.prepare_arrays(self)

        nleads, ndm1 = self.si.nleads, self.si.ndm1
        kern_size = self.get_kern_size()

        self.paulifct_eps = np.zeros((nleads, ndm1, 2), dtype=doublenp)

        self.Lpm = np.zeros((3*5, kern_size, kern_size), dtype=complexnp) # temporary
        self.Lpm_first = np.zeros((nleads,3, kern_size, kern_size), dtype=complexnp)
        self.Lpm_first_dz = np.zeros((nleads,3, kern_size, kern_size), dtype=complexnp)
        self.Lpm_second = np.zeros((nleads,nleads,3,3, kern_size, kern_size), dtype=complexnp)
        self.Lpm_second_dz = np.zeros((nleads,nleads,3,3, kern_size, kern_size), dtype=complexnp)

        kern_size_rows = kern_size if self.funcp.symq else kern_size+1
        self.kern_first = np.zeros((kern_size_rows, kern_size), dtype=self.dtype, order='F')
        self.phi0_first = np.zeros(kern_size, dtype=self.dtype)
        self.kern_second = np.zeros((kern_size_rows, kern_size), dtype=self.dtype, order='F')
        self.phi0_second = np.zeros(kern_size, dtype=self.dtype)

        kh = self.kernel_handler
        kh.phi0_first=self.phi0_first
        kh.kern_first=self.kern_first
        kh.phi0_second=self.phi0_second
        kh.kern_second=self.kern_second
        kh.Lpm = self.Lpm
        kh.Lpm_first = self.Lpm_first
        kh.Lpm_first_dz = self.Lpm_first_dz
        kh.Lpm_second = self.Lpm_second
        kh.Lpm_second_dz = self.Lpm_second_dz

        self.current_noise = np.zeros(2, dtype = complexnp)
        self.current_noise_first = np.zeros(2, dtype = complexnp)
        self.current_noise_o4trunc = np.zeros(4, dtype = complexnp)

        self.lpm_imaginary_2nd = True

    def clean_arrays(self):
        ApproachPyRTD.clean_arrays(self)

        self.Lpm.fill(0.0) # temporary
        self.Lpm_first.fill(0.0)
        self.Lpm_first_dz.fill(0.0)
        self.Lpm_second.fill(0.0)
        self.Lpm_second_dz.fill(0.0)
        self.current_noise.fill(0.0)
        self.current_noise_first.fill(0.0)
        self.current_noise_o4trunc.fill(0.0)
        self.phi0_first.fill(0.0)
        self.kern_first.fill(0.0)
        self.phi0_second.fill(0.0)
        self.kern_second.fill(0.0)
        self.coherence_correction = None
        self.coherence_correction_dz = None

        self.lpm_imaginary_2nd = True

    def solve(self, qdq=True, rotateq=True, masterq=True, currentq=True, *args, **kwargs):
        """
        Solves the master equation.

        Parameters
        ----------
        qdq : bool
            Diagonalise many-body quantum dot Hamiltonian
            and express the lead matrix Tba in the eigenbasis.
        rotateq : bool
            Rotate the many-body tunneling matrix Tba.
        masterq : bool
            Solve the master equation.
        currentq : bool
            Calculate the current.
        """
        if self.funcp.countingleads is None:
            raise ValueError(
                "RTDnoise requires a nonempty countingleads iterable."
            )
        if self.funcp.mfreeq:
            raise NotImplementedError(
                "Matrix-free counting statistics are not implemented."
            )
        if qdq:
            self.qd.diagonalise()
            if rotateq:
                self.rotate()
        #
        if masterq:
            self.prepare_kern()
            self.generate_fct()
            self.generate_kern()
            self.solve_kern()
            self.solve_kern_first()
            self.solve_kern_second()
            if getattr(self, 'success', False):
                check_stationary_solution(self)
            if currentq:
                self.generate_currents()


    def solve_kern_first(self):
        """Finds the stationary state using least squares or using LU decomposition."""
        solmethod = self.funcp.solmethod
        symq = self.funcp.symq
        norm_row = self.funcp.norm_row
        replaced_eq = self.replaced_eq

        kern = self.kern_first
        bvec = self.bvec

        # Replace one equation by the normalisation condition
        if symq:
            replaced_eq[:] = kern[norm_row]
            kern[norm_row] = self.norm_vec
            bvec[norm_row] = 1
        else:
            kern[-1] = self.norm_vec
            bvec[-1] = 1

        # Try to solve the master equation
        try:
            if solmethod == 'solve':
                self.sol0 = [np.linalg.solve(kern, bvec)]
            elif solmethod == 'lsqr':
                self.sol0 = np.linalg.lstsq(kern, bvec, rcond=-1)

            self.phi0_first[:] = self.sol0[0]
            self.success = True
        except Exception as exept:
            self.funcp.print_error(exept)
            self.phi0_first.fill(0.0)
            self.success = False
        if symq:
            kern[norm_row] = replaced_eq

    def solve_kern_second(self):
        """Finds the stationary state using least squares or using LU decomposition."""
        solmethod = self.funcp.solmethod
        symq = self.funcp.symq
        norm_row = self.funcp.norm_row
        replaced_eq = self.replaced_eq

        kern = self.kern_first
        bvec = - self.kern_second @ self.phi0_first

        # Replace one equation by the normalisation condition
        if symq:
            replaced_eq[:] = kern[norm_row]
            kern[norm_row] = self.norm_vec
            bvec[norm_row] = 1
        else:
            kern[-1] = self.norm_vec
            bvec[-1] = 1

        # Try to solve the master equation
        try:
            if solmethod == 'solve':
                self.sol0 = [np.linalg.solve(kern, bvec)]
            elif solmethod == 'lsqr':
                self.sol0 = np.linalg.lstsq(kern, bvec, rcond=-1)

            self.phi0_second[:] = self.sol0[0]
            self.success = True
        except Exception as exept:
            self.funcp.print_error(exept)
            self.phi0_second.fill(0.0)
            self.success = False
        if symq:
            kern[norm_row] = replaced_eq

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
        rtd_order = self.rtd_order

        # Stage 1: validate the two approximations that are external to the
        # diagram traversal.  The integral formulas need a sufficiently large
        # bandwidth, while eliminating same-charge coherences needs their Bohr
        # frequencies to remain resolved on the dissipative scale.  The
        # bandwidth requirement belongs to the four-vertex integrals alone.
        if rtd_order >= 2:
            _warn_if_unequal_temperature_cutoff_is_small(self.qd, self.leads)
        _warn_if_rtd_coherence_is_not_resolved(self)

        # Stage 2: prepare the pole representation used by the scalar direct
        # and exchange integrals.  Equal-temperature wide-band calls take an
        # analytic shortcut, but keeping the pole data ready gives both paths
        # one traversal and supports unequal temperatures.  Only the
        # four-vertex integrals consume it.
        if rtd_order >= 2:
            self.set_Ozaki_params()

        # Stage 3: assemble the independent population-space diagrams.  Each
        # unique population supplies a first-order row and a second-order
        # column.  Energy-current blocks share these state traversals, but are
        # separate observables and retain their own complex-amplitude warning.
        for bcharge in range(ncharge):
            for b in statesdm[bcharge]:
                if not kh.is_unique(b, b, bcharge):
                    continue
                self.generate_row_1st_order_kernel_lpm(b, bcharge)
                if rtd_order >= 2:
                    self.generate_col_diag_kern_2nd_order_lpm(b, bcharge)
                    # WE1 and WE2 are two contractions of one O(Gamma^2)
                    # correction, not first and second order: the leading
                    # energy current is the LE contraction of Wdd.
                    self.generate_row_1st_energy_kernel(b, bcharge)
                    self.generate_row_2nd_energy_kernel(b, bcharge)

        # Stage 4: the traversal inserts only one member of each eta0 pair.
        # Complete its value and Laplace-derivative partners before adding any
        # other physical block, so the partner identity is independently
        # testable and cannot accidentally duplicate the Schur correction.
        # It doubles ``Wdd``, which holds only second-order content at this
        # point, so it must not run when no second-order block was assembled.
        if rtd_order >= 2:
            self._complete_second_order_conjugate_partners()

        # Stage 5: optionally eliminate the same-charge coherence sector.  Its
        # Schur product is already resolved by lead and transferred charge, so
        # it can be added directly to both counted arrays and the stationary
        # lead-resolved kernel.
        if self.funcp.off_diag_corrections:
            generate_population_coherence_blocks(self)
            correction = counting_resolved_coherence_correction(self)
            self.coherence_correction = correction.kernel
            self.coherence_correction_dz = correction.laplace_derivative
            self.Lpm_second += correction.kernel
            self.Lpm_second_dz += correction.laplace_derivative
            self.Wdd += np.sum(correction.kernel.real, axis=(1, 2, 3))

        # Stage 6: enforce the analytic z-derivative channel only after every
        # population contribution has been assembled.  The helper warns before
        # removing a real part that is too large to be roundoff. See
        # :func:`_project_laplace_derivative_onto_imaginary`.
        _project_laplace_derivative_onto_imaginary(
            self, "Lpm_first_dz", self.Lpm_first_dz,
        )
        _project_laplace_derivative_onto_imaginary(
            self, "Lpm_second_dz", self.Lpm_second_dz,
        )

        # Stage 7: collapse lead and transfer axes into the physical stationary
        # kernel.  Keep first- and second-order matrices separately as well;
        # current/noise routines use them to expose order-resolved results.
        kern_size = self.get_kern_size()
        first_order = np.sum(self.Lpm_first.real, axis=(0, 1))
        second_order = np.sum(self.Lpm_second.real, axis=(0, 1, 2, 3))
        self.kern[:kern_size, :kern_size] += first_order + second_order
        self.kern_first[:kern_size, :kern_size] += first_order
        self.kern_second[:kern_size, :kern_size] += second_order

        self.Wdd += np.sum(self.Lpm_first, axis=1).real

    def _complete_second_order_conjugate_partners(self):
        r"""Complete the inverted-Keldysh partners of fourth-order diagrams.

        The traversal evaluates the independent ``eta0 = +1`` direct and
        exchange diagrams.  Inverting every electron-hole and Keldysh index
        gives the complex-conjugate population-kernel contribution
        [LeijnseWegewijs2008, Eqs. (B1)-(B3), (D1)-(D3)].  Complex tunnel
        vertices obey :math:`t_- = t_+^*` [Emary2009, Eqs. (6), (21)].

        Emary's contraction factor
        :math:`\exp[i s_\alpha\xi(p_1-p_2)\chi_\alpha/2]` is invariant when
        ``xi`` and both Keldysh indices are inverted, so a partner remains in
        the same transfer-resolved array entry [Emary2009, Eq. (31)].  At zero
        Laplace energy its value is therefore the complex conjugate.  From
        :math:`W(z)=W(-z^*)^*`, its Laplace derivative is instead the *negative*
        complex conjugate.  Consequently completed value arrays are real and
        completed derivative arrays are purely imaginary.

        ``KernelHandlerRTDnoise`` also records the independent contribution in
        the lead-resolved stationary kernel, whose real part is doubled here.
        """
        self.Lpm_second[:] = 2.0*self.Lpm_second.real
        self.Lpm_second_dz[:] = 2.0j*self.Lpm_second_dz.imag
        self.Wdd *= 2.0

    def generate_currents(self):
        self.generate_current_noise_first()
        self.generate_current_noise()
        self.generate_current_noise_o4trunc()
        self.generate_current()

    def generate_current_noise(self):
        """
        Calculates currents and noise with the second-order counting kernel via
        [Emary2009, Eqs. (40)-(41)]. The selected counting
        leads are summed.

        Returns
        ----------
        current : float
            Value of the current attaching the counting field to countingleads.
        noise : array
            Value of the current noise attaching the counting field to countingleads.
        """
        countingleads = self.funcp.countingleads
        first, second, _ = self._lead_resolved_derivatives(
            self.Lpm_first, self.Lpm_second, countingleads
        )
        first_dz, _, kernel_dz = self._lead_resolved_derivatives(
            self.Lpm_first_dz, self.Lpm_second_dz, countingleads
        )
        currents, noise_matrix = nonmarkovian_current_noise_matrix(
            self.kern, self.phi0, self.norm_vec, first, second,
            kernel_dz, first_dz,
        )
        self.current_noise_matrix = noise_matrix
        self.current_noise[0] = np.sum(currents)
        self.current_noise[1] = np.sum(noise_matrix)

    def generate_current_noise_first(self):
        """
        Calculates currents and noise with the first-order counting kernel via
        [Emary2009, Eqs. (40)-(41)]. The selected counting
        leads are summed.

        Returns
        ----------
        current : float
            Value of the current attaching the counting field to countingleads.
        noise : array
            Value of the current noise attaching the counting field to countingleads.
        """

        countingleads = self.funcp.countingleads
        first, second, _ = self._lead_resolved_derivatives(
            self.Lpm_first, None, countingleads
        )
        first_dz, _, kernel_dz = self._lead_resolved_derivatives(
            self.Lpm_first_dz, None, countingleads
        )
        currents, noise_matrix = nonmarkovian_current_noise_matrix(
            self.kern_first, self.phi0_first, self.norm_vec, first, second,
            kernel_dz, first_dz,
        )
        self.current_noise_matrix_first = noise_matrix
        self.current_noise_first[0] = np.sum(currents)
        self.current_noise_first[1] = np.sum(noise_matrix)

    @staticmethod
    def _lead_resolved_derivatives(
            Lpm_first, Lpm_second, countingleads):
        """Return individual counting-field derivatives at zero field."""
        countingleads = tuple(countingleads)
        lead_positions = {
            lead: position for position, lead in enumerate(countingleads)
        }
        ncounted = len(countingleads)
        size = Lpm_first.shape[-1]
        first = np.zeros((ncounted, size, size), dtype=complexnp)
        second = np.zeros(
            (ncounted, ncounted, size, size), dtype=complexnp
        )
        kernel = np.sum(Lpm_first, axis=(0, 1))

        for lead, position in lead_positions.items():
            for transfer in (-1, 1):
                contribution = Lpm_first[lead, transfer]
                first[position] += 1j * transfer * contribution
                second[position, position] -= transfer**2 * contribution

        if Lpm_second is None:
            return first, second, kernel

        kernel = kernel + np.sum(Lpm_second, axis=(0, 1, 2, 3))
        nleads = Lpm_second.shape[0]
        for lead0, lead1 in product(range(nleads), repeat=2):
            position0 = lead_positions.get(lead0)
            position1 = lead_positions.get(lead1)
            if position0 is None and position1 is None:
                continue
            for transfer0, transfer1 in product((-1, 0, 1), repeat=2):
                transfers = np.zeros(ncounted, dtype=int)
                if position0 is not None:
                    transfers[position0] += transfer0
                if position1 is not None:
                    transfers[position1] += transfer1
                if not np.any(transfers):
                    continue
                contribution = Lpm_second[
                    lead0, lead1, transfer0, transfer1
                ]
                for i in np.flatnonzero(transfers):
                    first[i] += 1j * transfers[i] * contribution
                    for j in np.flatnonzero(transfers):
                        second[i, j] -= (
                            transfers[i] * transfers[j] * contribution
                        )

        return first, second, kernel

    def generate_current_noise_o4trunc(self):
        """
        Calculates currents and noise with the O4trunc approximation from
        [Emary2009].

        Returns
        ----------
        current : float
            Value of the current attaching the counting field to countingleads.
        noise : array
            Value of the current noise attaching the counting field to countingleads.
        """
        countingleads = self.funcp.countingleads

        phi0 = self.phi0
        phi0_first = self.phi0_first
        phi0_second = self.phi0_second

        L0_1, Lp1_1, Lp2_1, Lm2_1, Lm1_1 = self.build_counting_kernels(self.Lpm_first,np.zeros(self.Lpm_second.shape),countingleads)
        _, Lp1_2, Lp2_2, Lm2_2, Lm1_2 = self.build_counting_kernels(np.zeros(self.Lpm_first.shape),self.Lpm_second,countingleads)
        L0p_1,Lp1p_1, Lp2p_1, Lm2p_1, Lm1p_1 = self.build_counting_kernels(self.Lpm_first_dz,np.zeros(self.Lpm_second_dz.shape),countingleads)
        kern = self.kern
        kern_first = self.kern_first
        kern_second = self.kern_second

        # auxilliary quantities
        # right eigenvector
        P = phi0[...,None]
        P0 = phi0_first[...,None]
        P1 = phi0_second[...,None]-phi0_first[...,None]
        P01 = P0 + P1
        # left eigenvector
        O = np.asarray(self.norm_vec)[None, :]
        # projector
        Q0 = (np.eye(np.size(P0)) - P0 @ O)
        stationary_projected_pseudoinverse(
            kern, phi0, self.norm_vec
        )
        stationary_projected_pseudoinverse(
            kern_first, phi0_first, self.norm_vec
        )
        # pseudoinverse
        # eps = 1e-8
        size = P.size
        kern_first_square = kern_first[:size, :size]
        kern_second_square = kern_second[:size, :size]
        Rm1 = stationary_kernel_pseudoinverse(kern_first_square)
        R0 = -Rm1 @ kern_second_square @ Rm1
        # derivatives of noise kernel
        Jp_1 = 1j*(Lp1_1 - Lm1_1 + 2*Lp2_1 - 2*Lm2_1)
        Jp_2 = 1j*(Lp1_2 - Lm1_2 + 2*Lp2_2 - 2*Lm2_2)
        Jpp_1 = -Lp1_1 - Lm1_1 - 4*Lp2_1 - 4*Lm2_1
        Jpp_2 = -Lp1_2 - Lm1_2 - 4*Lp2_2 - 4*Lm2_2
        Jdz_1 = L0p_1 + Lp1p_1 + Lp2p_1 + Lm2p_1 + Lm1p_1
        Jdzp_1 = 1j*(Lp1p_1 - Lm1p_1 + 2*Lp2p_1 - 2*Lm2p_1)
        # current
        c1 = -1j*(O @ Jp_1 @ P0)
        c2 = -1j*(O @ Jp_2 @ P0) - 1j*(O @ Jp_1 @ P01)
        # noise
        s1 = -(O @ Jpp_1 @ P0) + 2 * (O @ Jp_1 @ Q0 @ Rm1 @ Q0 @ Jp_1 @ P0)
        s2 = -(O @ Jpp_1 @ P01) - (O @ Jpp_2 @ P0) +\
            2 * (O @ Jp_1 @ Q0 @ Rm1 @ Q0 @ Jp_1 @ P01) +\
            2 * (O @ Jp_1 @ Q0 @ R0 @ Q0 @ Jp_1 @ P0) +\
            2 * (O @ Jp_1 @ P1)*(O @ Rm1 @ Q0 @ Jp_1 @ P0) +\
            2 * (O @ Jp_1 @ Q0 @ Rm1 @ P1)*(O @ Jp_1 @ P0) +\
            2 * (O @ Jp_2 @ Q0 @ Rm1 @ Q0 @ Jp_1 @ P0) +\
            -2 * (O @ Jp_1 @ P1)*(O @ Rm1 @ Q0 @ Jp_1 @ P0) +\
            -2 * (O @ Jp_1 @ Q0 @ Rm1 @ P1)*(O @ Jp_1 @ P0) +\
            2 * (O @ Jp_1 @ Q0 @ Rm1 @ Q0 @ Jp_2@ P0) +\
            -2 * (O @ Jp_1 @ P1)*(O @ Rm1 @ Q0 @ Jp_1@ P0) +\
            -2 * (O @ Jp_1 @ Q0 @ Rm1 @ P1)*(O @ Jp_1@ P0) +\
            2 * c1 * (O @ Jdzp_1 @ P0) -\
            2 * c1 * (O @ Jp_1 @ Q0 @ Rm1 @ Q0 @ Jdz_1 @ P0)
        self.current_noise_o4trunc[0] = c1.item()
        self.current_noise_o4trunc[1] = c2.item()
        self.current_noise_o4trunc[2] = s1.item()
        self.current_noise_o4trunc[3] = s2.item()

    def generate_row_1st_order_kernel_lpm(self, b, bcharge):
        """Generates a row in the first order diagonal kernel :math:`W_{dd}^{(1)}`.

        Parameters
        ----------
        b : int
            the final state (row)

        bcharge : int
            charge of state b

        self.Wdd : ndarray
            (Modifies) The kernel connecting diagional density-matrix elements. This Kernel
            has npauli * npauli entries.
        """
        kh = self.kernel_handler
        countingleads = self.funcp.countingleads
        itype = self.funcp.itype
        mulst, tlst, dlst = self.leads.mulst, self.leads.tlst, self.leads.dlst

        # The four branches keep each diagram's own argument orientation; see
        # the eta = -xi note in rtd_diagrams. A gain diagram from the lower
        # state adds an electron (L+), one from the upper state removes it
        # (L-), and a loss diagram is the escape through its other state.
        for d in first_order_diagrams(self, b, bcharge):
            l, dE, gamma, bb = d.lead, d.energy, d.gamma, d.row
            mu, Tr = mulst[l], tlst[l]
            if d.kind == GAIN and d.lower:
                aa = d.column
                # p0=1,p1=-1,eta=1
                lamb_p = dE - mu
                fermi_p = func_pauli(lamb_p, 0, Tr, dlst[l, 0], dlst[l, 1], itype)[0]
                # p0=-1,p1=1,eta=-1
                lamb_m = -dE + mu
                fermi_m = func_pauli(-lamb_m, 0, Tr, dlst[l, 0], dlst[l, 1], itype)[0]
                # d(phi)/d(energy): see the 1/Tr note on the class.
                phi_eps = diff_phi(lamb_p/Tr)/Tr
                kh.set_matrix_element_lpm_first(l,gamma/2*(fermi_p+fermi_m), 2j*gamma*phi_eps, 1, bb, aa)
                if l in countingleads:
                    kh.set_matrix_element_lpm_pauli(gamma/2*(fermi_p+fermi_m), 2, bb, aa)
                    kh.set_matrix_element_lpm_pauli(2j*gamma*phi_eps, 7, bb, aa)
                else:
                    kh.set_matrix_element_lpm_pauli(gamma/2*(fermi_p+fermi_m), 0, bb, aa)
                    kh.set_matrix_element_lpm_pauli(2j*gamma*phi_eps, 5, bb, aa)
            elif d.kind == GAIN:
                cc = d.column
                # p0=-1,p1=1,eta=1
                lamb_m = dE - mu
                fermi_m = func_pauli(-lamb_m, 0, Tr, dlst[l, 0], dlst[l, 1], itype)[0]
                # p0=1,p1=-1,eta=-1
                lamb_p = -dE + mu
                fermi_p = func_pauli(lamb_p, 0, Tr, dlst[l, 0], dlst[l, 1], itype)[0]
                phi_eps = diff_phi(lamb_p/Tr)/Tr
                kh.set_matrix_element_lpm_first(l,gamma/2*(fermi_p+fermi_m), 2j*gamma*phi_eps, -1, bb, cc)
                if l in countingleads:
                    kh.set_matrix_element_lpm_pauli(gamma/2*(fermi_p+fermi_m),1,bb,cc)
                    kh.set_matrix_element_lpm_pauli(2j*gamma*phi_eps, 6, bb, cc)
                else:
                    kh.set_matrix_element_lpm_pauli(gamma/2*(fermi_p+fermi_m),0,bb,cc)
                    kh.set_matrix_element_lpm_pauli(2j*gamma*phi_eps, 5, bb, cc)
            elif d.lower:
                # p0=-1,p1=-1,eta=1
                lamb_m = dE - mu
                fermi_m = func_pauli(-lamb_m, 0, Tr, dlst[l, 0], dlst[l, 1], itype)[0]
                # p0=1,p1=1,eta=-1
                lamb_p = -dE + mu
                fermi_p = func_pauli(lamb_p, 0, Tr, dlst[l, 0], dlst[l, 1], itype)[0]
                phi_eps = diff_phi(lamb_p/Tr)/Tr
                kh.set_matrix_element_lpm_first(l,-gamma/2*(fermi_p+fermi_m), -2j*gamma*phi_eps, 0, bb, bb)
                kh.set_matrix_element_lpm_pauli(-gamma/2*(fermi_m+fermi_p), 0, bb, bb)
                kh.set_matrix_element_lpm_pauli(-2j*gamma*phi_eps, 5, bb, bb)
            else:
                # p0=1,p1=1,eta=1
                lamb_p = dE - mu
                fermi_p = func_pauli(lamb_p, 0, Tr, dlst[l, 0], dlst[l, 1], itype)[0]
                # p0=-1,p1=-1,eta=-1
                lamb_m = -dE + mu
                fermi_m = func_pauli(-lamb_m, 0, Tr, dlst[l, 0], dlst[l, 1], itype)[0]
                phi_eps = diff_phi(lamb_p/Tr)/Tr
                kh.set_matrix_element_lpm_first(l,-gamma/2*(fermi_p+fermi_m), -2j*gamma*phi_eps, 0, bb, bb)
                kh.set_matrix_element_lpm_pauli(-gamma/2*(fermi_p+fermi_m), 0, bb, bb)
                kh.set_matrix_element_lpm_pauli(-2j*gamma*phi_eps, 5, bb, bb)

    def generate_col_diag_kern_2nd_order_lpm(self, a0, charge):
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

        self.Lpm : ndarray
            (Modifies) noise kernels.

        """
        # Each record is one independent eta0 = +1 diagram. Its eta0 = -1
        # partner is not evaluated: _complete_second_order_conjugate_partners
        # constructs it from the whole-diagram symmetry once every column
        # exists, keeping the record's counting labels.
        integralD_lpm = self.integralD_lpm
        integralD_lpm_derivative = self.integralD_lpm_derivative
        integralX_lpm = self.integralX_lpm
        integralX_lpm_derivative = self.integralX_lpm_derivative
        kh = self.kernel_handler
        b_and_R = self.Ozaki_poles_and_residues
        lpm_imaginary_2nd = self.lpm_imaginary_2nd

        for d in second_order_diagrams(self, a0, charge):
            t = d.tunnel_product
            args = (lpm_imaginary_2nd, d.p1, 1, d.eta1, d.E1, d.E2, d.E3,
                    d.T1, d.T2, d.mu1, d.mu2, d.D, b_and_R, True)
            if d.topology == DIRECT:
                value = t * integralD_lpm(*args)
                value_dz = t * integralD_lpm_derivative(*args)
                dx = 'd'
            else:
                value = -t * integralX_lpm(*args)
                value_dz = -t * integralX_lpm_derivative(*args)
                dx = 'x'
            kh.add_element_2nd_order(d.r0, d.r1, 1, d.eta1, d.p1, d.p2, value, value_dz,
                                     d.initial, d.initial_flipped, d.flipped_state,
                                     d.flipped_charge, d.final_state, d.final_charge, dx)

    def build_counting_kernels(self, Lpm_first, Lpm_second, countingleads):
        """Aggregate lead-resolved blocks by net counted charge.

        ``Lpm_second[r0, r1, q0, q1]`` records the transfers attached to its
        two reservoir contractions.  This routine maps them to ``L[q]`` for
        the selected lead set.  If only one contraction belongs to that set,
        its transfer is the total; if both do, their transfers are added.

        The zero-transfer block is filled last from the zero-field kernel. This
        guarantees that summing every ``L[q]`` exactly recovers the physical
        kernel, including diagrams whose individual transfer is zero.
        """
        nleads = self.si.nleads
        kern_size = self.get_kern_size()
        L = np.zeros((5, kern_size, kern_size), dtype=complex)

        # A first-order diagram has one reservoir contraction, so its stored
        # transfer index is already the net transfer through the counted set.
        for r0 in countingleads:
            L[-1] += Lpm_first[r0, -1, :, :]
            L[1] += Lpm_first[r0, 1, :, :]

        # A second-order diagram has two contractions. Separate the three
        # membership cases to avoid assigning a transfer from an uncounted lead.
        for r0, r1 in product(range(nleads), range(nleads)):
            if (r0 in countingleads) and (r1 not in countingleads):
                L[-1] += np.sum(Lpm_second[r0,r1,-1,:,:,:],axis=0)
                L[1] += np.sum(Lpm_second[r0,r1,1,:,:,:],axis=0)
            if (r1 in countingleads) and (r0 not in countingleads):
                L[-1] += np.sum(Lpm_second[r0,r1,:,-1,:,:],axis=0)
                L[1] += np.sum(Lpm_second[r0,r1,:,1,:,:],axis=0)
            if (r0 in countingleads) and (r1 in countingleads):
                # q0 + q1 = +/-1 can arise as (+/-1, 0) or (0, +/-1).
                L[-1] += Lpm_second[r0,r1,-1,0,:,:]
                L[-1] += Lpm_second[r0,r1,0,-1,:,:]
                L[1] += Lpm_second[r0,r1,1,0,:,:]
                L[1] += Lpm_second[r0,r1,0,1,:,:]
                L[-2] += Lpm_second[r0,r1,-1,-1,:,:]
                L[2] += Lpm_second[r0,r1,1,1,:,:]

        # Define q=0 by exact zero-field closure rather than by another case
        # split. This includes both genuinely zero-transfer diagrams and the
        # diagonal escape terms required by probability conservation.
        zero_field = (
            np.sum(Lpm_first, axis=(0, 1))
            + np.sum(Lpm_second, axis=(0, 1, 2, 3))
        )
        L[0] = zero_field - np.sum(L[1:], axis=0)
        return L

    def build_counting_kernels_first(self,Lpm_first,countingleads):
        """Builds the counting kernels for the given counting leads.
        """
        kern_size = self.get_kern_size()
        L = np.zeros((3,kern_size,kern_size),dtype=complex)
        # first order
        for r0 in countingleads:
            L[-1] += Lpm_first[r0,-1,:,:]
            L[1] += Lpm_first[r0,1,:,:]
        # for completeness
        L[0]=np.sum(Lpm_first,axis=(0,1))-np.sum(L[1:],axis=0)
        return L


class ApproachRTDnoise(ApproachPyRTDnoise):
    """RTDnoise with backend-selected compiled scalar integral acceleration.

    Kernel traversal remains Python because profiling identified the four
    direct/exchange value and derivative functions as the useful bounded
    Cython target.  Under ``QMEQ_BACKEND=python`` the selected functions resolve
    to their Python implementations, so this class remains portable.
    """

    kerntype = 'RTDnoise'
    integralD_lpm = staticmethod(selected_integralD_lpm)
    integralD_lpm_derivative = staticmethod(selected_integralD_lpm_derivative)
    integralX_lpm = staticmethod(selected_integralX_lpm)
    integralX_lpm_derivative = staticmethod(selected_integralX_lpm_derivative)
