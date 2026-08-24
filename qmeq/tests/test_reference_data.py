"""Checks for the shared external numerical-reference infrastructure."""

import importlib.util
import os
import subprocess
import sys

import pytest

from qmeq.tests.reference_data import REFERENCE_BUNDLE_SCHEMA
from qmeq.tests.reference_data import load_reference_bundle


BUNDLE_IDS = ("qmeq_11", "legacy", "counting_reference")


@pytest.mark.parametrize("bundle_id", BUNDLE_IDS)
def test_reference_bundle_uses_common_schema(bundle_id):
    bundle = load_reference_bundle(bundle_id)

    assert bundle.manifest["reference_bundle_schema"] == REFERENCE_BUNDLE_SCHEMA
    assert bundle.manifest["bundle_id"] == bundle_id
    assert bundle.arrays


def test_legacy_builder_snapshots_have_unknown_provenance():
    bundle = load_reference_bundle("legacy")
    snapshots = bundle.manifest["snapshots"]

    assert bundle.manifest["provenance"]["classification"] == "legacy snapshots"
    assert bundle.manifest["provenance"]["status"] == "unknown"
    assert len(snapshots["builder"]) == 44
    assert len(snapshots["builder_elph"]) == 56
    assert len(bundle.arrays) == 100


def test_counting_snapshot_retains_recorded_source_provenance():
    bundle = load_reference_bundle("counting_reference")
    snapshots = bundle.manifest["snapshots"]

    assert bundle.manifest["provenance"]["source_commit"] == (
        "aa1af46dd687c271505d28dbfb7ccce03a8a1739"
    )
    assert bundle.manifest["provenance"]["status"] == "recorded"
    assert len(snapshots["first_order"]) == 4
    assert len(snapshots["rtd"]) == 3
    assert len(snapshots["rtdnoise_scenarios"]) == 5
    assert len(bundle.arrays) == 85


@pytest.mark.parametrize("backend", ("python", "cython"))
def test_every_reference_bundle_loads_in_a_fresh_forced_backend(backend):
    """Backend selection and bundle loading must occur in the same fresh process."""
    if backend == "cython" and importlib.util.find_spec(
        "qmeq.approach.base.c_pauli"
    ) is None:
        pytest.skip("compiled backend is not installed")

    code = f"""
import qmeq
from qmeq.tests.reference_data import load_reference_bundle

assert qmeq.get_backend_status()["active"] == {backend!r}
for bundle_id in {BUNDLE_IDS!r}:
    bundle = load_reference_bundle(bundle_id)
    assert bundle.arrays
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=dict(os.environ, QMEQ_BACKEND=backend),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
