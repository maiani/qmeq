"""RTDnoise regression against the final pinned source commit.

The immutable arrays are generated from a pristine checkout of the final
commit. Current QmeQ solves the same public-input models and must reproduce the
stored counting-resolved kernels, stationary states, currents, and noise.

Structural invariants that hold independently of the bundle
live in ``test_rtdnoise_structural_invariants.py``, computed from a live
solve rather than compared against stored arrays.
"""

import numpy as np
import pytest

from qmeq.tests.reference_data import load_reference_bundle
from qmeq.tests.rtdnoise_reference_models import (
    RTDNOISE_CHARACTERIZATION_SCENARIOS,
    rtdnoise_scenario_snapshot,
)


BUNDLE = load_reference_bundle("counting_reference")
MANIFEST = BUNDLE.manifest
SCENARIOS = BUNDLE.resolve(MANIFEST["snapshots"]["rtdnoise_scenarios"])

_SNAPSHOT_FIELDS = (
    "Lpm_first", "Lpm_second", "Lpm_first_dot", "Lpm_second_dot",
    "phi0", "phi0_first", "phi0_second", "kern_first", "kern_second", "Wdd",
    "current", "current_noise", "current_noise_first",
    "current_noise_o4trunc",
)
_DERIVATIVE_FIELDS = frozenset(("Lpm_first_dot", "Lpm_second_dot"))

# The pinned source differentiated the first-order blocks with respect to the
# *scaled* Laplace energy z/T_lead rather than z, so its `Lpm_first_dot` is
# T_lead times too large.  The bundle is a provenance artifact and stays
# byte-identical; the comparison carries the correction instead.  Expressing it
# as an exact per-lead rescaling is a sharper pin than the original equality:
# it reproduces the historical array *and* the precise form of the defect,
# including in the two unequal-temperature scenarios, where a single global
# factor cannot match.
_LEAD_SCALED_DERIVATIVE_FIELDS = frozenset(("Lpm_first_dot",))

# Observables downstream of that derivative therefore differ from the pinned
# ones at the 1e-3 level.  Those historical values are known-wrong and must not
# become a compatibility contract, so they are not asserted as equalities.
# They remain in the immutable bundle schema, but are no longer equality
# references. Their corrected physics is gated independently at unequal
# temperature and by energy-rescaling covariance.
_SUPERSEDED_NONMARKOVIAN_FIELDS = frozenset((
    "current_noise", "current_noise_first", "current_noise_o4trunc",
))


def _assert_characterization_field(
        actual: np.ndarray, expected: np.ndarray, field: str,
        tolerance: dict[str, float], message: str,
        temperatures: np.ndarray | None = None) -> None:
    """Compare a field in its physical channel at the measured numeric floor.

    For real-amplitude fixtures the Laplace derivatives are imaginary. Their
    tiny real components are subtraction roundoff from the historical fixed-
    step finite difference and move across SciPy patch releases. Comparing
    those non-physical crumbs bitwise makes the characterization environment-
    pinned without protecting a scientific result. The ``*_dot`` field names
    are likewise part of the immutable historical bundle schema; live runtime
    arrays use ``*_dz``.
    """
    if field in _SUPERSEDED_NONMARKOVIAN_FIELDS:
        assert np.all(np.isfinite(actual)), f"{message}: non-finite result"
        return

    if field in _LEAD_SCALED_DERIVATIVE_FIELDS:
        # The lead axis is axis 0 of Lpm_first*.
        if temperatures is None:
            raise ValueError("Lead temperatures are required for this field.")
        broadcast = np.asarray(temperatures, dtype=float).reshape(
            (-1,) + (1,)*(np.ndim(expected) - 1)
        )
        expected = expected/broadcast

    if field not in _DERIVATIVE_FIELDS:
        np.testing.assert_allclose(
            actual, expected, rtol=tolerance["rtol"], atol=tolerance["atol"],
            equal_nan=True, err_msg=message,
        )
        return

    scale = max(float(np.max(np.abs(expected.imag))), tolerance["atol"])
    roundoff_floor = 4.0 * np.finfo(float).eps / 1e-8 * scale
    np.testing.assert_allclose(
        actual.imag, expected.imag,
        rtol=tolerance["rtol"],
        atol=max(tolerance["atol"], roundoff_floor),
        equal_nan=True, err_msg=f"{message} (physical imaginary channel)",
    )
    assert np.max(np.abs(actual.real)) <= roundoff_floor, message
    assert np.max(np.abs(expected.real)) <= roundoff_floor, message


def test_bundle_provenance_is_pinned_to_final_source_commit():
    assert MANIFEST["reference_bundle_schema"] == 1
    assert MANIFEST["bundle_id"] == "counting_reference"
    provenance = MANIFEST["provenance"]
    assert provenance["classification"] == "pinned historical source"
    assert provenance["source_author"]
    assert provenance["source_commit"] == (
        "aa1af46dd687c271505d28dbfb7ccce03a8a1739"
    )
    assert provenance["source_tag"] == "reference/rtdnoise-source"
    assert provenance["source_version"] == "1.1"
    assert MANIFEST["generator"] == (
        "scripts/reference_data/generate_counting_reference.py"
    )
    assert MANIFEST["rtdnoise_model_definitions"] == (
        "qmeq/tests/rtdnoise_reference_models.py"
    )


def test_bundle_covers_every_characterization_scenario():
    assert set(SCENARIOS) == set(RTDNOISE_CHARACTERIZATION_SCENARIOS)
    assert set(MANIFEST["rtdnoise_scenario_metadata"]) == set(SCENARIOS)
    scenario_map = MANIFEST["snapshots"]["rtdnoise_scenarios"]
    for scenario, fields in scenario_map.items():
        assert set(fields) == set(_SNAPSHOT_FIELDS)
        for field, array_name in fields.items():
            metadata = MANIFEST["arrays"][array_name]
            array = SCENARIOS[scenario][field]
            assert metadata["shape"] == list(array.shape)
            assert metadata["dtype"] == str(array.dtype)
@pytest.mark.parametrize("scenario", RTDNOISE_CHARACTERIZATION_SCENARIOS)
def test_scenario_forces_off_diag_corrections_false(scenario):
    # The pinned source predates the counted coherence correction.  Preserve
    # that historical compatibility mode rather than changing the fixture.
    resolved = MANIFEST["rtdnoise_scenario_metadata"][scenario]["resolved_model"]
    assert resolved["off_diag_corrections"] is False


@pytest.mark.parametrize("scenario", RTDNOISE_CHARACTERIZATION_SCENARIOS)
def test_current_scenario_reproduces_bundle(scenario):
    """Current QmeQ must reproduce the pinned block-level snapshot."""
    actual, system = rtdnoise_scenario_snapshot(scenario)
    expected = SCENARIOS[scenario]
    tolerance = MANIFEST["rtdnoise_scenario_metadata"][scenario]["tolerance"]
    assert set(expected) == set(_SNAPSHOT_FIELDS)
    assert set(expected) <= set(actual)
    for field in expected:
        _assert_characterization_field(
            actual[field], expected[field], field, tolerance,
            f"RTDnoise pinned-source regression in {scenario}/{field}",
            temperatures=system.appr.leads.tlst,
        )


@pytest.mark.parametrize("scenario", RTDNOISE_CHARACTERIZATION_SCENARIOS)
def test_selected_backend_alias_reproduces_bundle(scenario):
    """The generic 'RTDnoise' kerntype alias must resolve identically.

    RTDnoise has no compiled counterpart: ``ApproachRTDnoise`` is
    ``ApproachPyRTDnoise`` regardless of ``QMEQ_BACKEND``
    (``qmeq/builder/builder_base.py``), so this does not exercise a second
    numerical implementation -- only that the alias resolves to the same
    class under whichever backend this process was forced into.
    """
    actual, system = rtdnoise_scenario_snapshot(
        scenario, use_selected_backend=True
    )
    expected = SCENARIOS[scenario]
    tolerance = MANIFEST["rtdnoise_scenario_metadata"][scenario]["tolerance"]
    for field in expected:
        _assert_characterization_field(
            actual[field], expected[field], field, tolerance,
            f"RTDnoise selected-backend alias drift in {scenario}/{field}",
            temperatures=system.appr.leads.tlst,
        )
