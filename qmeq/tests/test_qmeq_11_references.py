"""Regression tests against the last pre-modernization QmeQ 1.1 source."""

import numpy as np
import pytest

import qmeq
from qmeq.tests.qmeq_11_reference_models import _rtd_reference_snapshot
from qmeq.tests.qmeq_11_reference_models import build_reference_system
from qmeq.tests.qmeq_11_reference_models import build_rtd_reference_system
from qmeq.tests.qmeq_11_reference_models import RTD_REFERENCE_SCENARIOS
from qmeq.tests.qmeq_11_reference_models import solve_rtd_reference_system
from qmeq.tests.reference_data import load_reference_bundle


EXPECTED_SOURCE_COMMIT = "96cc51076458b11f7db81a5d7d8df04c30bf8384"
REFERENCE_BUNDLE = load_reference_bundle("qmeq_11")
REFERENCE_DOCUMENT = REFERENCE_BUNDLE.manifest
SOURCE_COMMIT = REFERENCE_DOCUMENT["source_commit"]
SOURCE_VERSION = REFERENCE_DOCUMENT["source_version"]


def _load_reference_arrays():
    return (
        REFERENCE_BUNDLE.resolve(REFERENCE_DOCUMENT["references"]),
        REFERENCE_BUNDLE.resolve(REFERENCE_DOCUMENT["rtd_scenarios"]),
    )


REFERENCES, RTD_REFERENCES = _load_reference_arrays()

BASE_CASES = (
    ("Pauli", 2),
    ("Lindblad", 2),
    ("Redfield", 0),
    ("Redfield", 1),
    ("Redfield", 2),
    ("1vN", 0),
    ("1vN", 1),
    ("1vN", 2),
    ("2vN", 2),
    ("RTD", 1),
)

ELPH_CASES = ("Pauli", "Lindblad", "Redfield", "1vN")
SELECTED_ELPH_CASES = (
    "Pauli",
    pytest.param(
        "Lindblad",
        marks=pytest.mark.xfail(
            condition=qmeq.get_backend_status()["active"] == "cython",
            reason=("Compiled electron-phonon Lindblad stationary-state parity "
                    "is an existing P1 roadmap gap."),
            strict=True,
        ),
    ),
    "Redfield",
    "1vN",
)
SNAPSHOT_FIELDS = ("current", "energy_current", "heat_current", "phi0", "kern")
STATIONARY_FIELDS = ("current", "energy_current", "heat_current", "phi0")


def _current_snapshot(system):
    return {
        "current": np.asarray(system.current),
        "energy_current": np.asarray(system.energy_current),
        "heat_current": np.asarray(system.heat_current),
        "phi0": np.asarray(system.phi0),
        "kern": np.asarray(system.appr.kern),
    }


def _assert_snapshot_matches(
        reference, actual, *, rtol, atol, fields=SNAPSHOT_FIELDS,
        comparison_overrides=None):
    if comparison_overrides is None:
        comparison_overrides = {}
    for field in fields:
        tolerance = comparison_overrides.get(field, {})
        np.testing.assert_allclose(
            actual[field], reference[field],
            rtol=tolerance.get("rtol", rtol),
            atol=tolerance.get("atol", atol),
            equal_nan=True, err_msg=f"QmeQ 1.1 regression in {field}",
        )


def _scenario_tolerance(scenario, field):
    metadata = REFERENCE_DOCUMENT["rtd_scenario_metadata"][scenario]
    return metadata.get("comparison_overrides", {}).get(
        field, metadata["tolerance"]
    )


def test_qmeq_11_reference_provenance_and_coverage():
    assert REFERENCE_DOCUMENT["reference_bundle_schema"] == 1
    assert REFERENCE_DOCUMENT["bundle_id"] == "qmeq_11"
    assert REFERENCE_DOCUMENT["schema_version"] == 2
    assert SOURCE_COMMIT == EXPECTED_SOURCE_COMMIT
    assert SOURCE_VERSION == "1.1"
    environment = REFERENCE_DOCUMENT["generation_environment"]
    assert environment["source_commit"] == EXPECTED_SOURCE_COMMIT
    assert environment["qmeq_version"] == "1.1"
    assert environment["implementation"] == "pure-python historical source"
    assert set(REFERENCES) == {
        *(f"base/{approach}/itype={itype}" for approach, itype in BASE_CASES),
        *(f"elph/{approach}/itype=2/itype_ph=2" for approach in ELPH_CASES),
    }
    assert set(REFERENCE_DOCUMENT["reference_metadata"]) == set(REFERENCES)
    assert set(RTD_REFERENCES) == set(RTD_REFERENCE_SCENARIOS)

    for case, fields in REFERENCE_DOCUMENT["references"].items():
        for field, array_name in fields.items():
            metadata = REFERENCE_DOCUMENT["arrays"][array_name]
            array = REFERENCES[case][field]
            assert metadata["shape"] == list(array.shape)
            assert metadata["dtype"] == str(array.dtype)

    for scenario, fields in REFERENCE_DOCUMENT[
            "rtd_scenarios"].items():
        fixture_metadata = REFERENCE_DOCUMENT[
            "rtd_scenario_metadata"][scenario]
        assert fixture_metadata["classification"].endswith("QmeQ 1.1")
        assert fixture_metadata["units"]
        assert fixture_metadata["array_ordering"]
        assert fixture_metadata["tolerance"]["rationale"]
        assert fixture_metadata["resolved_model"]["many_body_energies"]
        for field, array_name in fields.items():
            metadata = REFERENCE_DOCUMENT["arrays"][array_name]
            array = RTD_REFERENCES[scenario][field]
            assert metadata["shape"] == list(array.shape)
            assert metadata["dtype"] == str(array.dtype)


@pytest.mark.parametrize(("approach", "itype"), BASE_CASES)
def test_electronic_approaches_match_qmeq_11(approach, itype):
    key = f"base/{approach}/itype={itype}"
    system = build_reference_system(
        "base", approach, itype, use_selected_backend=True
    )
    rtol, atol = (2e-9, 2e-11) if approach == "2vN" else (2e-11, 2e-12)
    _assert_snapshot_matches(
        REFERENCES[key], _current_snapshot(system), rtol=rtol, atol=atol,
        fields=STATIONARY_FIELDS,
    )


@pytest.mark.parametrize("approach", SELECTED_ELPH_CASES)
def test_electron_phonon_approaches_match_qmeq_11(approach):
    key = f"elph/{approach}/itype=2/itype_ph=2"
    system = build_reference_system(
        "elph", approach, use_selected_backend=True
    )
    _assert_snapshot_matches(
        REFERENCES[key], _current_snapshot(system), rtol=2e-10, atol=1e-10,
        fields=STATIONARY_FIELDS,
    )


@pytest.mark.parametrize(("approach", "itype"), BASE_CASES)
def test_pure_python_electronic_kernels_match_qmeq_11(approach, itype):
    key = f"base/{approach}/itype={itype}"
    system = build_reference_system("base", approach, itype)
    np.testing.assert_allclose(
        system.appr.kern, REFERENCES[key]["kern"],
        rtol=2e-9 if approach == "2vN" else 2e-11,
        atol=2e-11 if approach == "2vN" else 2e-12,
        err_msg=f"QmeQ 1.1 regression in pure-Python {approach} kernel",
    )


@pytest.mark.parametrize("approach", ELPH_CASES)
def test_pure_python_electron_phonon_kernels_match_qmeq_11(approach):
    key = f"elph/{approach}/itype=2/itype_ph=2"
    system = build_reference_system("elph", approach)
    np.testing.assert_allclose(
        system.appr.kern, REFERENCES[key]["kern"], rtol=2e-10, atol=2e-12,
        err_msg=f"QmeQ 1.1 regression in pure-Python elph {approach} kernel",
    )


@pytest.mark.parametrize("scenario", RTD_REFERENCE_SCENARIOS)
def test_python_rtd_reference_scenarios_match_qmeq_11(scenario):
    actual = _rtd_reference_snapshot(scenario)
    reference = RTD_REFERENCES[scenario]
    assert actual.keys() == reference.keys()
    for field in reference:
        tolerance = _scenario_tolerance(scenario, field)
        np.testing.assert_allclose(
            actual[field], reference[field],
            rtol=tolerance["rtol"], atol=tolerance["atol"], equal_nan=True,
            err_msg=f"QmeQ 1.1 regression in {scenario}/{field}",
        )


@pytest.mark.parametrize("scenario", RTD_REFERENCE_SCENARIOS)
def test_selected_rtd_reference_results_match_qmeq_11(scenario):
    system = solve_rtd_reference_system(
        build_rtd_reference_system(scenario, use_selected_backend=True), scenario
    )
    reference = RTD_REFERENCES[scenario]
    tolerance = REFERENCE_DOCUMENT["rtd_scenario_metadata"][scenario][
        "tolerance"
    ]
    overrides = REFERENCE_DOCUMENT["rtd_scenario_metadata"][scenario].get(
        "comparison_overrides"
    )
    _assert_snapshot_matches(
        reference, _current_snapshot(system),
        rtol=tolerance["rtol"], atol=tolerance["atol"],
        fields=STATIONARY_FIELDS, comparison_overrides=overrides,
    )


@pytest.mark.parametrize("scenario", RTD_REFERENCE_SCENARIOS)
def test_selected_rtd_backend_matches_qmeq_11_full_blocks(scenario):
    system = solve_rtd_reference_system(
        build_rtd_reference_system(scenario, use_selected_backend=True), scenario
    )
    appr = system.appr
    compiled_layout = appr.Wdd is None
    reference_snapshot = RTD_REFERENCES[scenario]

    def optional_block(name, value):
        if value is None:
            return np.zeros_like(reference_snapshot[name])
        return value

    inverse_lnn = appr.Lnn
    if compiled_layout and inverse_lnn is not None:
        inverse_lnn = np.diag(inverse_lnn)
    selected_blocks = {
        "Wdd_total": appr.Wdd2[0] if compiled_layout else appr.Wdd,
        "ReWdn": optional_block("ReWdn", appr.ReWdn),
        "ImWdn": optional_block("ImWdn", appr.ImWdn),
        "ReWnd": optional_block("ReWnd", appr.ReWnd),
        "ImWnd": optional_block("ImWnd", appr.ImWnd),
        "inverse_Lnn": optional_block("inverse_Lnn", inverse_lnn),
        "WE1": appr.WE1,
        "WE2": appr.WE2,
    }
    for name, actual in selected_blocks.items():
        reference = reference_snapshot[name]
        if (compiled_layout and name in {"ReWnd", "ImWnd"}
                and np.asarray(actual).ndim == 2):
            inverse_lnn = np.asarray(
                RTD_REFERENCES[scenario]["inverse_Lnn"]
            )
            reference = inverse_lnn @ np.sum(reference, axis=0)
        tolerance = _scenario_tolerance(scenario, name)
        np.testing.assert_allclose(
            actual, reference, rtol=tolerance["rtol"],
            atol=tolerance["atol"], equal_nan=True,
            err_msg=(f"QmeQ 1.1 regression in selected RTD block "
                     f"{scenario}/{name}"),
        )


@pytest.mark.parametrize("scenario", RTD_REFERENCE_SCENARIOS)
def test_qmeq_11_rtd_block_decomposition_is_complete(scenario):
    reference = RTD_REFERENCES[scenario]
    np.testing.assert_allclose(
        reference["Wdd_first"] + reference["Wdd_second"]
        + reference["Wdd_elimination"],
        reference["Wdd_total"], rtol=2e-14, atol=2e-14,
    )


@pytest.mark.parametrize("scenario", RTD_REFERENCE_SCENARIOS)
def test_qmeq_11_rtd_reference_invariants(scenario):
    reference = RTD_REFERENCES[scenario]
    metadata = REFERENCE_DOCUMENT["rtd_scenario_metadata"][scenario]
    atol = max(metadata["tolerance"]["atol"], 2e-13)

    np.testing.assert_allclose(np.sum(reference["phi0"]), 1.0, atol=atol)
    np.testing.assert_allclose(
        reference["kern"] @ reference["phi0"], 0.0, atol=atol
    )
    np.testing.assert_allclose(np.sum(reference["current"]), 0.0, atol=atol)

    for field in (
            "Wdd_first", "Wdd_second", "Wdd_elimination", "Wdd_total"):
        np.testing.assert_allclose(
            np.sum(reference[field], axis=1), 0.0, atol=atol,
            err_msg=f"trace preservation failed for {scenario}/{field}",
        )

    if np.all(np.isfinite(reference["energy_current"])):
        np.testing.assert_allclose(
            np.sum(reference["energy_current"]), 0.0, atol=atol
        )
        chemical_potentials = np.asarray(
            metadata["resolved_model"]["chemical_potentials"]
        )
        np.testing.assert_allclose(
            reference["heat_current"],
            reference["energy_current"]
            - chemical_potentials * reference["current"],
            atol=atol,
        )


def test_qmeq_11_rtd_equilibrium_and_structural_zeros():
    reference = RTD_REFERENCES["single_level_equilibrium"]
    np.testing.assert_allclose(reference["current"], 0.0, atol=2e-15)
    np.testing.assert_allclose(reference["energy_current"], 0.0, atol=2e-15)
    assert reference["ReWdn"].shape[-1] == 0
    assert reference["ImWdn"].shape[-1] == 0
    assert reference["ReWnd"].shape[-2] == 0
    assert reference["ImWnd"].shape[-2] == 0
    assert reference["inverse_Lnn"].shape == (0, 0)
    np.testing.assert_array_equal(reference["Wdd_elimination"], 0.0)


def test_qmeq_11_rtd_off_diag_switch_is_isolated():
    enabled = RTD_REFERENCES["coherent_real_offdiag_on"]
    disabled = RTD_REFERENCES["coherent_real_offdiag_off"]
    np.testing.assert_array_equal(enabled["Wdd_first"], disabled["Wdd_first"])
    np.testing.assert_array_equal(
        enabled["Wdd_second"], disabled["Wdd_second"]
    )
    np.testing.assert_array_equal(disabled["Wdd_elimination"], 0.0)
    assert np.max(np.abs(enabled["Wdd_elimination"])) > 0.0


def test_qmeq_11_rtd_spin_symmetry_request_falls_back_to_charge():
    metadata = REFERENCE_DOCUMENT["rtd_scenario_metadata"][
        "spin_symmetry_fallback"
    ]
    assert metadata["resolved_model"]["indexing"] == "charge"
    assert "does not support spin symmetry" in metadata["notes"]
