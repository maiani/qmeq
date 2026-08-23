"""Tests for the stationary solution diagnostics (TODO P0: flag unphysical
stationary solutions instead of returning them silently)."""

import warnings

from contextlib import contextmanager

from types import SimpleNamespace

import numpy as np
import pytest

import qmeq
from qmeq.approach.diagnostics import POPULATION_TOL
from qmeq.approach.diagnostics import TRACE_TOL
from qmeq.approach.diagnostics import _solver_conditioning
from qmeq.approach.diagnostics import check_stationary_solution


def build_pauli_system():
    return qmeq.Builder(1, {(0, 0): 0.0}, {}, 2,
                        {(0, 0): 1.0, (1, 0): 1.0},
                        {0: 0.0, 1: 0.0}, {0: 100.0, 1: 100.0},
                        {0: 1000.0, 1: 1000.0}, kerntype='Pauli')


@contextmanager
def catch_warnings():
    """Record all warnings, matching the style of test_warnings.py."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        yield caught


def test_physical_solution_passes_silently():
    system = build_pauli_system()
    with catch_warnings() as caught:
        system.solve()

    assert not caught
    diag = system.appr.stationary_diagnostics
    assert diag is not None
    assert diag.physical
    assert diag.min_population >= -POPULATION_TOL
    assert diag.trace_deviation <= TRACE_TOL
    assert diag.trace == pytest.approx(1.0)


def test_negative_population_flags_unphysical_and_warns():
    system = build_pauli_system()
    system.solve()
    appr = system.appr
    appr.phi0[0] = -0.81

    with catch_warnings() as caught:
        diag = check_stationary_solution(appr)

    assert not diag.physical
    assert diag.min_population == -0.81
    assert len(caught) == 1
    assert issubclass(caught[0].category, qmeq.QmeqRuntimeWarning)
    assert "Unphysical stationary solution" in str(caught[0].message)
    assert appr.stationary_diagnostics is diag


def test_warning_is_shown_once_per_instance():
    system = build_pauli_system()
    system.solve()
    appr = system.appr
    appr.phi0[0] = -0.5

    with catch_warnings() as first:
        check_stationary_solution(appr)
    assert len(first) == 1

    # The diagnostic keeps updating even though the warning is suppressed.
    appr.phi0[0] = -0.81
    with catch_warnings() as second:
        diag = check_stationary_solution(appr)
    assert not second
    assert not diag.physical
    assert diag.min_population == -0.81


def test_trace_deviation_flags_unphysical():
    system = build_pauli_system()
    system.solve()
    appr = system.appr
    appr.phi0[:] *= 0.5

    with catch_warnings() as caught:
        diag = check_stationary_solution(appr)

    assert not diag.physical
    assert diag.trace == pytest.approx(0.5)
    assert diag.trace_deviation == pytest.approx(0.5)
    assert len(caught) == 1


def test_nan_population_flags_unphysical():
    system = build_pauli_system()
    system.solve()
    appr = system.appr
    appr.phi0[0] = np.nan

    with catch_warnings() as caught:
        diag = check_stationary_solution(appr)

    assert not diag.physical
    assert len(caught) == 1


def test_warn_false_records_without_warning():
    system = build_pauli_system()
    system.solve()
    appr = system.appr
    appr.phi0[0] = -0.81

    with catch_warnings() as caught:
        diag = check_stationary_solution(appr, warn=False)

    assert not diag.physical
    assert not caught


def test_diagnostics_reset_on_restart():
    system = build_pauli_system()
    system.solve()
    assert system.appr.stationary_diagnostics is not None

    system.appr.restart()
    assert system.appr.stationary_diagnostics is None


def test_redfield_with_coherences_is_diagnosed():
    system = qmeq.Builder(1, {(0, 0): 0.0}, {}, 2,
                          {(0, 0): 1.0, (1, 0): 1.0},
                          {0: 0.0, 1: 0.0}, {0: 100.0, 1: 100.0},
                          {0: 1000.0, 1: 1000.0},
                          kerntype='Redfield', symq=False, solmethod='lsqr')
    with catch_warnings() as caught:
        system.solve()

    assert not caught
    diag = system.appr.stationary_diagnostics
    assert diag.physical
    # The least-squares provenance is recorded when the solver provides it.
    assert diag.solver_rank == system.si.ndm0r


def test_matrix_free_solution_is_diagnosed():
    system = qmeq.Builder(1, {(0, 0): 1.5}, {}, 2,
                          {(0, 0): 1.0, (1, 0): 1.0},
                          {0: 2.0, 1: -2.0}, {0: 0.4, 1: 0.4},
                          {0: 100.0, 1: 100.0}, kerntype='Lindblad', itype=1,
                          principal_part="digamma", mfreeq=True)
    # Provide phi0_init so the missing-initial-state warning does not fire.
    phi0_init = np.zeros(system.si.ndm0r)
    phi0_init[0] = 1.0
    system.phi0_init = phi0_init
    with catch_warnings() as caught:
        system.solve()

    assert not caught
    diag = system.appr.stationary_diagnostics
    assert diag.physical
    # Without norm_vec the trace comes from the kernel handler.
    assert diag.trace == pytest.approx(1.0, abs=1e-7)


def test_solver_conditioning_extraction():
    lstsq_like = (np.zeros(2), np.array([0.5, 1.0]), 3, np.zeros(4))
    assert _solver_conditioning(SimpleNamespace(sol0=lstsq_like)) == (3, 1.0)
    # Empty residuals (square full-rank lstsq call) mean zero residual.
    empty_residuals = (np.zeros(2), np.array([]), 2, np.zeros(2))
    assert _solver_conditioning(SimpleNamespace(sol0=empty_residuals)) == (2, 0.0)
    # scipy.optimize.root result carries the residual in fun.
    root_like = SimpleNamespace(sol0=SimpleNamespace(fun=np.array([0.1, -0.2])))
    rank, residual = _solver_conditioning(root_like)
    assert rank is None and residual == pytest.approx(0.2)
    # Direct linear solves carry no conditioning information.
    assert _solver_conditioning(SimpleNamespace(sol0=[np.zeros(2)])) == (None, None)
    assert _solver_conditioning(SimpleNamespace(sol0=None)) == (None, None)
