"""RTDnoise structural invariants.

These hold independently of the known real-part truncations in the current
second-order kernel. They are computed from a **live
solve** of the same scenario builders used to generate the pinned reference
bundle (``qmeq/tests/rtdnoise_reference_models.py``); nothing here compares
against the pinned ``counting_reference`` arrays, so these tests retain
their value even after that bundle's numbers move.

Covered:

* per-channel and per-order column-sum zero;
* current conservation, :math:`\\sum_\\alpha I_\\alpha = 0`;
* equilibrium zero current;
* covariance-matrix symmetry; and
* aggregate counting statistics equal the sum of their lead-resolved
  entries.
"""

import warnings

import numpy as np
import pytest

import qmeq
from qmeq.tests.rtdnoise_reference_models import (
    BASELINE_SCENARIOS,
    INVARIANT_SCENARIOS,
    build_equilibrium_scenario,
    rtdnoise_scenario_snapshot,
)


@pytest.mark.parametrize("scenario", INVARIANT_SCENARIOS)
def test_per_lead_kernel_column_sum_is_zero(scenario):
    """Each lead's own contribution to the kernel conserves probability.

    ``Wdd`` has shape ``(nleads, final_state, initial_state)``; for a valid
    master-equation generator, summing over the final-state (row) index must
    vanish for every column and every lead individually, not only after
    summing over leads.
    """
    snapshot, _ = rtdnoise_scenario_snapshot(scenario)
    column_sums = np.sum(snapshot["Wdd"], axis=1)
    np.testing.assert_allclose(
        column_sums, 0.0, atol=1e-10,
        err_msg=f"{scenario}: per-lead Wdd column sum is not zero",
    )


@pytest.mark.parametrize("scenario", INVARIANT_SCENARIOS)
def test_counting_blocks_conserve_probability_per_lead_and_order(scenario):
    """Each retained order conserves probability before leads are combined."""
    snapshot, _ = rtdnoise_scenario_snapshot(scenario)
    first_by_lead = np.sum(snapshot["Lpm_first"].real, axis=1)
    second_by_lead_pair = np.sum(snapshot["Lpm_second"].real, axis=(2, 3))
    np.testing.assert_allclose(
        np.sum(first_by_lead, axis=1), 0.0, atol=1e-10,
        err_msg=f"{scenario}: first-order block violates conservation",
    )
    np.testing.assert_allclose(
        np.sum(second_by_lead_pair, axis=2), 0.0, atol=1e-10,
        err_msg=f"{scenario}: second-order block violates conservation",
    )


@pytest.mark.parametrize("scenario", INVARIANT_SCENARIOS)
def test_current_is_conserved_across_leads(scenario):
    """Stationary particle current must sum to zero over all leads."""
    snapshot, _ = rtdnoise_scenario_snapshot(scenario)
    np.testing.assert_allclose(
        np.sum(snapshot["current"]), 0.0, atol=1e-10,
        err_msg=f"{scenario}: sum of lead currents is not zero",
    )


@pytest.mark.parametrize("scenario", BASELINE_SCENARIOS)
def test_counted_current_matches_ordinary_current(scenario):
    """Counting and ordinary-current paths use the same physical convention."""
    snapshot, system = rtdnoise_scenario_snapshot(scenario)
    expected = np.sum(system.current[np.asarray(system.countingleads, dtype=int)])
    np.testing.assert_allclose(
        snapshot["current_noise"][0].real, expected, rtol=1e-10, atol=1e-12,
        err_msg=f"{scenario}: counted and ordinary currents disagree",
    )
    np.testing.assert_allclose(
        snapshot["current_noise"][0].imag, 0.0, atol=1e-12,
        err_msg=f"{scenario}: physical counted current has an imaginary part",
    )


def test_equilibrium_gives_zero_current_and_zero_counting_current():
    """Zero bias, equal temperatures: both the ordinary and counted current vanish."""
    system = build_equilibrium_scenario()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=r".*energy_current and heat_current.*",
            category=qmeq.QmeqRuntimeWarning,
        )
        system.solve()
    np.testing.assert_allclose(
        system.current, 0.0, atol=1e-12,
        err_msg="equilibrium ordinary current is not zero",
    )
    np.testing.assert_allclose(
        system.current_noise[0], 0.0, atol=1e-12,
        err_msg="equilibrium counted current is not zero",
    )


@pytest.mark.parametrize("scenario", INVARIANT_SCENARIOS)
def test_covariance_matrices_are_symmetric(scenario):
    """The counting covariance matrix must be symmetric by construction.

    This holds for both the full and first-order-only covariance, and for
    both the real-amplitude and generic-flux scenarios: nothing
    about G2 licenses an asymmetric covariance matrix.
    """
    snapshot, _ = rtdnoise_scenario_snapshot(scenario)
    for field in ("current_noise_matrix", "current_noise_matrix_first"):
        matrix = snapshot[field]
        np.testing.assert_allclose(
            matrix, matrix.T, atol=1e-10,
            err_msg=f"{scenario}: {field} is not symmetric",
        )


@pytest.mark.parametrize("scenario", INVARIANT_SCENARIOS)
def test_aggregate_counting_equals_sum_of_lead_resolved_entries(scenario):
    """The reported aggregate cumulants equal the sum of their own detail matrix.

    ``current_noise[1]`` is the noise obtained by counting every lead in
    ``countingleads`` with one combined field; ``current_noise_matrix`` is
    the same solve's per-counted-lead covariance detail (diagonal:
    self-noise; off-diagonal: cross-lead covariance). Aggregating a joint
    counting field is exactly summing every entry of that matrix, so this
    identity must hold independently of the number of counted leads, and
    independently of whether the amplitudes are real or generic-flux complex.
    """
    snapshot, _ = rtdnoise_scenario_snapshot(scenario)
    np.testing.assert_allclose(
        snapshot["current_noise"][1],
        np.sum(snapshot["current_noise_matrix"]),
        atol=1e-10,
        err_msg=f"{scenario}: aggregate noise != sum of lead-resolved matrix",
    )
    np.testing.assert_allclose(
        snapshot["current_noise_first"][1],
        np.sum(snapshot["current_noise_matrix_first"]),
        atol=1e-10,
        err_msg=(
            f"{scenario}: aggregate first-order noise != sum of "
            "lead-resolved first-order matrix"
        ),
    )


def test_multi_lead_covariance_contains_cross_correlations():
    """The multi-lead check exercises real off-diagonal covariance entries."""
    snapshot, _ = rtdnoise_scenario_snapshot("multi_counted_leads")
    matrix = snapshot["current_noise_matrix"]
    assert matrix.shape == (2, 2)
    assert abs(matrix[0, 1]) > 1e-12
