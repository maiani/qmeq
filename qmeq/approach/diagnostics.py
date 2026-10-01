"""Diagnostics for stationary density-matrix solutions.

The approximate master equations implemented in QmeQ (Redfield, 1vN, 2vN, RTD)
can produce stationary reduced density matrices that violate positivity or
normalization while the computed currents look unremarkable. This module
checks every stationary solution and both

* emits a :class:`qmeq.QmeqRuntimeWarning` (once per approach instance), and
* stores a queryable :class:`StationarySolutionDiagnostics` on the approach
  as ``approach.stationary_diagnostics``, so scripted sweeps can filter on
  ``diagnostics.physical`` instead of parsing stderr.

The check runs in pure Python and is called from the approach ``solve``
methods, so its behaviour is identical for the pure-Python and Cython
backends and for all approaches.
"""

from dataclasses import dataclass
from typing import Optional

import warnings

import numpy as np

from .._warnings import QmeqRuntimeWarning
from .._warnings import QmeqWarning

POPULATION_TOL = 1e-8
"""Tolerance for the smallest population: populations below ``-POPULATION_TOL``
flag the stationary state as unphysical."""

TRACE_TOL = 1e-8
"""Tolerance for the deviation of the density-matrix trace from one."""


@dataclass(frozen=True)
class StationarySolutionDiagnostics(object):
    """Result of checking a stationary solution for physicality.

    Attributes
    ----------
    physical : bool
        True if the solution passed all checks within the tolerances.
    min_population : float
        Smallest population (diagonal element) of the reduced density matrix.
        A physical state has ``min_population >= -POPULATION_TOL``.
    trace : float
        Trace of the reduced density matrix (multiplicity-weighted under
        ``indexing='ssq'``). A physical state has ``trace ≈ 1``.
    trace_deviation : float
        Absolute deviation of the trace from one,
        ``abs(trace - 1) <= TRACE_TOL`` for a physical state.
    solver_rank : int or None
        Rank reported by the least-squares solver, when available. A value
        below the kernel size indicates a rank-deficient kernel.
    solver_residual : float or None
        Solver-reported residual at the solution, when available: the
        least-squares residual norm, or the max-abs Liouvillian residual for
        matrix-free root finding. None for direct linear solves.
    """

    physical: bool
    min_population: float
    trace: float
    trace_deviation: float
    solver_rank: Optional[int] = None
    solver_residual: Optional[float] = None


def _solver_conditioning(appr):
    """Extract solver conditioning information from the stored solution."""
    sol0 = getattr(appr, 'sol0', None)
    if sol0 is None:
        return None, None
    # numpy.linalg.lstsq result: (solution, residuals, rank, singular_values)
    if isinstance(sol0, tuple) and len(sol0) == 4:
        residuals, rank = sol0[1], int(sol0[2])
        residual = float(np.max(residuals)) if np.size(residuals) else 0.0
        return rank, residual
    # scipy.optimize.root result carries the residual in fun
    fun = getattr(sol0, 'fun', None)
    if fun is not None:
        return None, float(np.max(np.abs(fun)))
    return None, None


def _trace(appr, phi0_real):
    """Trace of the packed density matrix (rule L8 of dm_layout).

    Uses the precomputed normalization vector when available; otherwise sums
    the population entries over all many-body states, which applies the same
    multiplicity weighting under ``indexing='ssq'``.
    """
    norm_vec = getattr(appr, 'norm_vec', None)
    if norm_vec is not None:
        # Some complex-valued approaches allocate norm_vec with the approach
        # dtype even though normalization itself is purely real.
        return float(np.dot(np.real(norm_vec), phi0_real))
    si = appr.si
    trace = 0.0
    for charge in range(si.ncharge):
        for b in si.statesdm[charge]:
            trace += phi0_real[si.get_ind_dm0(b, b, charge)]
    return float(trace)


def check_stationary_solution(appr, warn=True):
    """Diagnose the stationary solution in ``appr.phi0`` for physicality.

    Checks negative populations, the trace deviation from one, and the
    solver-reported conditioning. Stores the result as
    ``appr.stationary_diagnostics`` and, unless the diagnosis is suppressed,
    warns once per approach instance when the solution is unphysical.

    Parameters
    ----------
    appr : Approach
        Approach object with a solved ``phi0``.
    warn : bool
        When False, only compute and store the diagnostics.

    Returns
    -------
    StationarySolutionDiagnostics
        The stored diagnostics.
    """
    si = appr.si
    phi0 = np.asarray(appr.phi0)

    populations = np.real(phi0[:si.npauli])
    min_population = float(np.min(populations))

    trace = _trace(appr, np.real(phi0))
    trace_deviation = abs(trace - 1.0)

    finite = bool(np.isfinite(min_population) and np.isfinite(trace_deviation))
    physical = (
        finite
        and min_population >= -POPULATION_TOL
        and trace_deviation <= TRACE_TOL
    )

    rank, residual = _solver_conditioning(appr)

    diag = StationarySolutionDiagnostics(
        physical=physical,
        min_population=min_population,
        trace=trace,
        trace_deviation=trace_deviation,
        solver_rank=rank,
        solver_residual=residual,
    )
    appr.stationary_diagnostics = diag

    funcp = appr.funcp
    if warn and not physical and not funcp.suppress_unphysical_wrn:
        kerntype = getattr(appr, 'kerntype', 'unknown').removeprefix('py')
        warnings.warn(
            "Unphysical stationary solution from the %s approach:\n"
            "  minimum population %+.3g (tolerance %g),\n"
            "  trace deviation %.3g (tolerance %g).\n"
            "The reduced density matrix violates positivity or "
            "normalization, and observables computed from it may be wrong.\n"
            "This warning is shown once per approach instance; filter "
            "scripted sweeps on `approach.stationary_diagnostics.physical`."
            % (kerntype, min_population, POPULATION_TOL,
               trace_deviation, TRACE_TOL),
            QmeqRuntimeWarning,
            stacklevel=2,
        )
        funcp.suppress_unphysical_wrn = True

    return diag


def _coupled_transitions_by_lead(appr):
    """Yield, per lead, the transitions it couples to and those inside its band.

    A transition ``c <- b`` adds one electron at energy ``E_c - E_b``; a
    finite band keeps it only strictly inside ``(dlst[l, 0], dlst[l, 1])``.
    Yields ``(lead, coupled, inside)`` with two boolean matrices over the
    carried states, rows ``c`` and columns ``b``, and skips leads that couple
    to no transition at all.
    """
    si = appr.si
    states = [b for sector in si.statesdm for b in sector]
    if not states:
        return
    states = np.asarray(states, dtype=int)
    charge = np.empty(len(states), dtype=int)
    offset = 0
    for n, sector in enumerate(si.statesdm):
        charge[offset:offset + len(sector)] = n
        offset += len(sector)
    energy = np.asarray(appr.qd.Ea)[states]
    adds_one = charge[:, None] == charge[None, :] + 1
    transition = energy[:, None] - energy[None, :]
    Tba, dlst = np.asarray(appr.leads.Tba), np.asarray(appr.leads.dlst)

    for lead in range(si.nleads):
        coupled = adds_one & (np.abs(Tba[lead][np.ix_(states, states)]) > 0.0)
        if not coupled.any():
            continue
        inside = (dlst[lead, 0] < transition) & (transition < dlst[lead, 1])
        yield lead, coupled, inside


def _listed_bands(appr, leads):
    dlst = np.asarray(appr.leads.dlst)
    return ", ".join(
        f"lead {lead} (band {dlst[lead, 0]:g} to {dlst[lead, 1]:g})"
        for lead in leads
    )


def check_band_coverage(appr):
    """Warn once when a finite band leaves a lead no transition to tunnel through.

    With a finite band (``itype`` 0 or 2) a rate is kept only for a
    transition energy ``E_c - E_b`` strictly inside the lead's band
    ``(dlst[l, 0], dlst[l, 1])``, so a lead whose every coupled transition
    lies outside it has all its rates zero and carries no current. Moving
    ``dband`` by a hair across the transition energy is then the difference
    between a finite current and an exact, silent zero. Leads with no coupled
    transition at all are not flagged: that is the model, not the band.

    The warning is shown once per system. The wide-band ``itype`` values 1
    and 3 are covered by :func:`check_wide_band_edges` instead.
    """
    funcp = appr.funcp
    if funcp.itype not in (0, 2) or funcp.suppress_band_wrn:
        return
    silenced = [
        lead for lead, coupled, inside in _coupled_transitions_by_lead(appr)
        if not (coupled & inside).any()
    ]

    if silenced:
        listed = _listed_bands(appr, silenced)
        warnings.warn(
            "With a finite band (itype=%d) no transition these leads couple to "
            "lies inside their band: %s. Their tunnelling rates are all zero, "
            "so they carry no current. Widen dband, or use bandwidth='infinite' "
            "if the wide-band limit is intended. This warning is shown once "
            "per system." % (funcp.itype, listed),
            QmeqWarning,
            stacklevel=3,
        )
        funcp.suppress_band_wrn = True


_BAND_REGULATED_KERNTYPES = frozenset({'RTD', 'pyRTD', 'RTDnoise', 'pyRTDnoise'})
"""Approaches whose wide-band integrals keep ``dband`` as a finite regulator.

Their bandwidth requirement is checked by the RTD diagnostics instead."""


def check_wide_band_edges(appr):
    """Warn once when a wide-band calculation is given a band its rates ignore.

    With ``bandwidth='infinite'`` (``itype`` 1 or 3) the rates keep every
    transition, wherever the band edges lie, and the wide-band forms of the
    principal parts assume that every edge lies far outside the transition
    energies. A band that excludes a transition the lead couples to therefore
    does not act as a band. For a symmetric band the current is unchanged
    when ``dband`` shrinks below the transition energies. For an asymmetric
    one, the ``log|D_-/D_+|`` term of the digamma principal part changes it,
    in a regime where that form does not hold.

    A lead whose band is exactly ``(0, 0)`` is not flagged: that is the
    stored value when no ``dband`` was given. RTD and RTDnoise are skipped,
    because there ``dband`` is the regulator of the second-order integrals.
    The warning is shown once per system.
    """
    funcp = appr.funcp
    if funcp.itype not in (1, 3) or funcp.suppress_wide_band_wrn:
        return
    if appr.kerntype in _BAND_REGULATED_KERNTYPES:
        return
    dlst = np.asarray(appr.leads.dlst)
    ignored = [
        lead for lead, coupled, inside in _coupled_transitions_by_lead(appr)
        if not (dlst[lead, 0] == 0.0 and dlst[lead, 1] == 0.0)
        and (coupled & ~inside).any()
    ]

    if ignored:
        warnings.warn(
            "With bandwidth='infinite' (itype=%d) the band edges do not remove "
            "transitions from the rates, and the wide-band principal parts "
            "assume every edge lies far outside the transition energies. "
            "These leads couple to transitions outside their band: %s. Use "
            "bandwidth='finite' if the band edges are meant to cut those "
            "transitions off, or widen dband. This warning is shown once per "
            "system." % (funcp.itype, _listed_bands(appr, ignored)),
            QmeqWarning,
            stacklevel=3,
        )
        funcp.suppress_wide_band_wrn = True


SYMMETRY_TOL = 1e-10
"""Relative size below which a symmetry-breaking coupling is taken as roundoff."""


def check_indexing_symmetry(appr):
    """Warn once when the couplings break the symmetry 'sz'/'ssq' indexing assumes.

    'sz' indexing keeps density-matrix coherences only between states of equal
    S_z, and 'ssq' additionally of equal total spin, reducing each multiplet to
    one representative. That is exact only if tunnelling and phonon couplings
    respect the symmetry; otherwise couplings or coherences the model has are
    dropped without notice, which moved the current of a spin-dependent double
    dot under 'ssq' by 8%, and by factors in other models. The checks are on
    the single-particle
    couplings, with the first ``nsingle//2`` orbitals spin up:

    * 'sz': no lead channel or phonon coupling connects orbitals of opposite
      spin;
    * 'ssq': in addition, the lead channels that share a chemical potential,
      temperature and band couple spin up and spin down identically, and so
      does every phonon bath.

    Pauli under 'sz' is skipped: it keeps no coherences, so the indexing does
    not change it. The dot Hamiltonian is not checked here; S_z-changing dot
    terms are refused when the dot is built.
    """
    si = appr.si
    funcp = appr.funcp
    if si.indexing not in ('sz', 'ssq') or funcp.suppress_symmetry_wrn:
        return
    if si.indexing == 'sz' and appr.kerntype.removeprefix('py') == 'Pauli':
        return
    half = si.nsingle//2
    problems = []

    amplitudes = np.asarray(appr.leads.tleads_array)
    for lead in range(amplitudes.shape[0]):
        row = np.abs(amplitudes[lead])
        if row[:half].any() and row[half:].any():
            problems.append(f"lead {lead} couples to both spins")
    if si.indexing == 'ssq':
        leads = appr.leads
        channels = {}
        for lead in range(amplitudes.shape[0]):
            key = (float(leads.mulst[lead]), float(leads.tlst[lead]),
                   *map(float, leads.dlst[lead]))
            channels.setdefault(key, []).append(lead)
        for members in channels.values():
            gamma = sum(np.outer(amplitudes[l].conj(), amplitudes[l])
                        for l in members)
            scale = np.abs(gamma).max()
            if scale > 0 and (np.abs(gamma[:half, :half] - gamma[half:, half:]).max()
                              > SYMMETRY_TOL*scale):
                problems.append(
                    f"leads {members} couple spin up and spin down differently")

    baths = getattr(appr, 'baths', None)
    if baths is not None and si.nbaths:
        coupling = np.zeros((si.nbaths, si.nsingle, si.nsingle), dtype=complex)
        for (bath, i, j), value in baths.velph.items():
            coupling[bath, i, j] += value
        for bath in range(si.nbaths):
            v = coupling[bath]
            scale = np.abs(v).max()
            if scale == 0:
                continue
            cross = max(np.abs(v[:half, half:]).max(), np.abs(v[half:, :half]).max())
            if cross > SYMMETRY_TOL*scale:
                problems.append(f"phonon bath {bath} couples opposite spins")
            elif (si.indexing == 'ssq'
                  and np.abs(v[:half, :half] - v[half:, half:]).max()
                  > SYMMETRY_TOL*scale):
                problems.append(
                    f"phonon bath {bath} couples spin up and spin down differently")

    if problems:
        conserved = "S_z" if si.indexing == 'sz' else "S_z and total spin"
        warnings.warn(
            "indexing=%r assumes the leads and phonon baths conserve %s, but "
            "%s. Couplings or coherences the model has are then dropped, which "
            "can change the results well beyond numerical error; use "
            "indexing='charge'. This warning is shown once per system."
            % (si.indexing, conserved, "; ".join(problems)),
            QmeqWarning,
            stacklevel=3,
        )
        funcp.suppress_symmetry_wrn = True
