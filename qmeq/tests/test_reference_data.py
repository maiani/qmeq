"""Checks for the shared external numerical-reference infrastructure."""

import pytest

from qmeq.tests.reference_data import REFERENCE_BUNDLE_SCHEMA
from qmeq.tests.reference_data import load_reference_bundle


@pytest.mark.parametrize("bundle_id", ["qmeq_11", "legacy", "counting_simon"])
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


def test_counting_snapshot_retains_recorded_simon_provenance():
    bundle = load_reference_bundle("counting_simon")
    snapshots = bundle.manifest["snapshots"]

    assert bundle.manifest["provenance"]["source_commit"] == (
        "aa1af46dd687c271505d28dbfb7ccce03a8a1739"
    )
    assert bundle.manifest["provenance"]["status"] == "recorded"
    assert len(snapshots["first_order"]) == 4
    assert len(snapshots["rtd"]) == 3
    assert len(bundle.arrays) == 15
