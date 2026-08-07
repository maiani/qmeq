"""Zero-frequency particle-current cumulants from counting fields.

The implementation follows Eqs. (40) and (41) of C. Emary,
Phys. Rev. B 80, 235306 (2009).  It constructs the lead-selected jump
superoperators for QmeQ's first-order approaches and evaluates the first two
cumulants with the projected pseudoinverse of the stationary kernel.
"""

from __future__ import annotations

import itertools

import numpy as np

from ..wrappers.mytypes import doublenp
from .kernel_handler import KernelHandlerNoise


_SUPPORTED_APPROACHES = {"Pauli", "Lindblad", "Redfield", "1vN"}


def _approach_name(approach):
    return approach.kerntype.removeprefix("py")


def _new_counting_handler(approach):
    size = approach.phi0.size
    lpm = np.zeros((2, size, size), dtype=doublenp)
    handler = KernelHandlerNoise(approach.si)
    handler.set_lpm(lpm)
    return handler, lpm


def _build_pauli_counting_kernel(approach, countingleads):
    paulifct, si = approach.paulifct, approach.si
    statesdm = si.statesdm
    handler, lpm = _new_counting_handler(approach)

    for bcharge in range(si.ncharge):
        for b in statesdm[bcharge]:
            if not handler.is_unique(b, b, bcharge):
                continue
            bb = si.get_ind_dm0(b, b, bcharge)
            for a in statesdm[bcharge - 1]:
                aa = si.get_ind_dm0(a, a, bcharge - 1)
                ba = si.get_ind_dm1(b, a, bcharge - 1)
                for lead in countingleads:
                    handler.set_matrix_element_lpm_pauli(
                        paulifct[lead, ba, 0], 1, bb, aa
                    )
            for c in statesdm[bcharge + 1]:
                cc = si.get_ind_dm0(c, c, bcharge + 1)
                cb = si.get_ind_dm1(c, b, bcharge)
                for lead in countingleads:
                    handler.set_matrix_element_lpm_pauli(
                        paulifct[lead, cb, 1], 0, bb, cc
                    )
    return lpm


def _build_lindblad_counting_kernel(approach, countingleads):
    tLba, si = approach.tLba, approach.si
    statesdm = si.statesdm
    handler, lpm = _new_counting_handler(approach)

    for bcharge in range(si.ncharge):
        for b, bp in itertools.combinations_with_replacement(
                statesdm[bcharge], 2):
            if not (
                handler.is_included(b, bp, bcharge)
                and handler.is_unique(b, bp, bcharge)
            ):
                continue
            for a, ap in itertools.product(
                    statesdm[bcharge - 1], repeat=2):
                if not handler.is_included(a, ap, bcharge - 1):
                    continue
                for lead in countingleads:
                    handler.set_matrix_element_lpm(
                        1j * tLba[lead, b, a]
                        * tLba[lead, bp, ap].conjugate(),
                        1, b, bp, bcharge, a, ap, bcharge - 1,
                    )
            for c, cp in itertools.product(
                    statesdm[bcharge + 1], repeat=2):
                if not handler.is_included(c, cp, bcharge + 1):
                    continue
                for lead in countingleads:
                    handler.set_matrix_element_lpm(
                        1j * tLba[lead, b, c]
                        * tLba[lead, bp, cp].conjugate(),
                        0, b, bp, bcharge, c, cp, bcharge + 1,
                    )
    return lpm


def _build_1vn_counting_kernel(approach, countingleads):
    Tba, phi1fct, si = (
        approach.leads.Tba, approach.phi1fct, approach.si
    )
    statesdm = si.statesdm
    handler, lpm = _new_counting_handler(approach)

    for bcharge in range(si.ncharge):
        for b, bp in itertools.combinations_with_replacement(
                statesdm[bcharge], 2):
            if not (
                handler.is_included(b, bp, bcharge)
                and handler.is_unique(b, bp, bcharge)
            ):
                continue
            for a, ap in itertools.product(
                    statesdm[bcharge - 1], repeat=2):
                if not handler.is_included(a, ap, bcharge - 1):
                    continue
                bpa = si.get_ind_dm1(bp, a, bcharge - 1)
                bap = si.get_ind_dm1(b, ap, bcharge - 1)
                for lead in countingleads:
                    fct = (
                        Tba[lead, b, a] * Tba[lead, ap, bp]
                        * phi1fct[lead, bpa, 0].conjugate()
                        - Tba[lead, b, a] * Tba[lead, ap, bp]
                        * phi1fct[lead, bap, 0]
                    )
                    handler.set_matrix_element_lpm(
                        fct, 1, b, bp, bcharge, a, ap, bcharge - 1
                    )
            for c, cp in itertools.product(
                    statesdm[bcharge + 1], repeat=2):
                if not handler.is_included(c, cp, bcharge + 1):
                    continue
                cbp = si.get_ind_dm1(c, bp, bcharge)
                cpb = si.get_ind_dm1(cp, b, bcharge)
                for lead in countingleads:
                    fct = (
                        Tba[lead, b, c] * Tba[lead, cp, bp]
                        * phi1fct[lead, cbp, 1]
                        - Tba[lead, b, c] * Tba[lead, cp, bp]
                        * phi1fct[lead, cpb, 1].conjugate()
                    )
                    handler.set_matrix_element_lpm(
                        fct, 0, b, bp, bcharge, c, cp, bcharge + 1
                    )
    return lpm


def _build_redfield_counting_kernel(approach, countingleads):
    Tba, phi1fct, si = (
        approach.leads.Tba, approach.phi1fct, approach.si
    )
    statesdm = si.statesdm
    handler, lpm = _new_counting_handler(approach)

    for bcharge in range(si.ncharge):
        for b, bp in itertools.combinations_with_replacement(
                statesdm[bcharge], 2):
            if not (
                handler.is_included(b, bp, bcharge)
                and handler.is_unique(b, bp, bcharge)
            ):
                continue
            for a, ap in itertools.product(
                    statesdm[bcharge - 1], repeat=2):
                if not handler.is_included(a, ap, bcharge - 1):
                    continue
                bpap = si.get_ind_dm1(bp, ap, bcharge - 1)
                ba = si.get_ind_dm1(b, a, bcharge - 1)
                for lead in countingleads:
                    fct = (
                        Tba[lead, b, a] * Tba[lead, ap, bp]
                        * phi1fct[lead, bpap, 0].conjugate()
                        - Tba[lead, b, a] * Tba[lead, ap, bp]
                        * phi1fct[lead, ba, 0]
                    )
                    handler.set_matrix_element_lpm(
                        fct, 1, b, bp, bcharge, a, ap, bcharge - 1
                    )
            for c, cp in itertools.product(
                    statesdm[bcharge + 1], repeat=2):
                if not handler.is_included(c, cp, bcharge + 1):
                    continue
                cpbp = si.get_ind_dm1(cp, bp, bcharge)
                cb = si.get_ind_dm1(c, b, bcharge)
                for lead in countingleads:
                    fct = (
                        Tba[lead, b, c] * Tba[lead, cp, bp]
                        * phi1fct[lead, cpbp, 1]
                        - Tba[lead, b, c] * Tba[lead, cp, bp]
                        * phi1fct[lead, cb, 1].conjugate()
                    )
                    handler.set_matrix_element_lpm(
                        fct, 0, b, bp, bcharge, c, cp, bcharge + 1
                    )
    return lpm


def build_first_order_counting_kernel(approach, countingleads):
    """Return ``[L_minus, L_plus]`` for one aggregated counting field."""
    name = _approach_name(approach)
    if name == "Pauli":
        return _build_pauli_counting_kernel(approach, countingleads)
    if name == "Lindblad":
        return _build_lindblad_counting_kernel(approach, countingleads)
    if name == "1vN":
        return _build_1vn_counting_kernel(approach, countingleads)
    if name == "Redfield":
        return _build_redfield_counting_kernel(approach, countingleads)
    raise NotImplementedError(
        f"Counting statistics are not implemented for {approach.kerntype}."
    )


def stationary_projected_pseudoinverse(
        kernel, stationary_state, trace_vector):
    """Return the stationary projector and projected kernel pseudoinverse."""
    stationary_state = np.asarray(stationary_state)
    size = stationary_state.size
    kernel = np.asarray(kernel)[:size, :size]
    trace_vector = np.asarray(trace_vector)[:size]

    singular_values = np.linalg.svd(kernel, compute_uv=False)
    scale = singular_values[0] if singular_values.size else 0.0
    tolerance = max(kernel.shape) * np.finfo(float).eps * scale
    nullity = np.count_nonzero(singular_values <= tolerance)
    if nullity != 1:
        raise np.linalg.LinAlgError(
            "Counting statistics require a unique stationary state; "
            f"the kernel has nullity {nullity}."
        )

    trace = trace_vector @ stationary_state
    if not np.isclose(trace, 1.0, rtol=1e-10, atol=1e-12):
        raise np.linalg.LinAlgError(
            "The stationary state is not normalized for counting statistics."
        )

    right = stationary_state[:, None]
    left = trace_vector[None, :]
    projector = np.eye(size) - right @ left
    pseudoinverse = (
        projector @ np.linalg.pinv(kernel, rcond=1e-15) @ projector
    )
    return right, left, projector, pseudoinverse


def markovian_current_noise(kernel, stationary_state, trace_vector, lpm):
    """Evaluate the first two cumulants of a Markovian counting kernel."""
    lminus, lplus = np.asarray(lpm)
    right, left, _, pseudoinverse = stationary_projected_pseudoinverse(
        kernel, stationary_state, trace_vector
    )

    first_derivative = 1j * (lplus - lminus)
    second_derivative = -(lplus + lminus)
    current = -1j * (left @ first_derivative @ right)
    noise = -left @ (
        second_derivative
        - 2 * first_derivative @ pseudoinverse @ first_derivative
    ) @ right
    return np.asarray([current.real.item(), noise.real.item()], dtype=doublenp)


def validate_counting_request(approach):
    """Reject requested counting modes that have no controlled implementation."""
    countingleads = approach.funcp.countingleads
    if countingleads is None:
        return
    if approach.funcp.mfreeq:
        raise NotImplementedError(
            "Matrix-free counting statistics are not implemented; set "
            "mfreeq=False."
        )
    if hasattr(approach, "baths"):
        raise NotImplementedError(
            "Counting statistics are not implemented for electron-phonon "
            "approaches."
        )
    name = _approach_name(approach)
    if name not in _SUPPORTED_APPROACHES:
        raise NotImplementedError(
            f"Counting statistics are not implemented for {approach.kerntype}."
        )


def generate_counting_statistics(approach):
    """Populate first-order counting results after an approach has solved."""
    countingleads = approach.funcp.countingleads
    if countingleads is None:
        approach._counting_kernel = None
        approach.Lpm = None
        approach.current_noise = None
        return
    validate_counting_request(approach)

    lpm = build_first_order_counting_kernel(approach, countingleads)
    approach.Lpm = lpm
    approach.current_noise = markovian_current_noise(
        approach._counting_kernel, approach.phi0, approach.norm_vec, lpm
    )
