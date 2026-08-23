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
