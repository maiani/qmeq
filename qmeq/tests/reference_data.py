"""Load external numerical-reference bundles used by the test suite."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


REFERENCE_ROOT = Path(__file__).with_name("data")
REFERENCE_BUNDLE_SCHEMA = 1


@dataclass(frozen=True)
class ReferenceBundle:
    """Validated manifest and arrays from one external reference bundle."""

    path: Path
    manifest: dict
    arrays: dict[str, np.ndarray]

    def resolve(self, mapping):
        """Resolve a nested manifest mapping from array names to arrays."""
        if isinstance(mapping, str):
            return self.arrays[mapping]
        if isinstance(mapping, dict):
            return {key: self.resolve(value) for key, value in mapping.items()}
        if isinstance(mapping, list):
            return [self.resolve(value) for value in mapping]
        raise TypeError(f"Unsupported reference mapping: {type(mapping)!r}")


def load_reference_bundle(bundle_id):
    """Load and validate one ``manifest.json``/``references.npz`` bundle."""
    bundle_path = REFERENCE_ROOT / bundle_id
    manifest_path = bundle_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("reference_bundle_schema") != REFERENCE_BUNDLE_SCHEMA:
        raise ValueError(f"Unsupported reference bundle schema in {manifest_path}")
    if manifest.get("bundle_id") != bundle_id:
        raise ValueError(f"Reference bundle id mismatch in {manifest_path}")

    archive_path = bundle_path / manifest["array_archive"]
    with np.load(archive_path, allow_pickle=False) as archive:
        expected_names = set(manifest["arrays"])
        if set(archive.files) != expected_names:
            raise ValueError(f"Reference array inventory mismatch in {archive_path}")
        arrays = {name: archive[name].copy() for name in archive.files}

    for name, array in arrays.items():
        metadata = manifest["arrays"][name]
        if metadata["shape"] != list(array.shape):
            raise ValueError(f"Reference shape mismatch for {bundle_id}/{name}")
        if metadata["dtype"] != str(array.dtype):
            raise ValueError(f"Reference dtype mismatch for {bundle_id}/{name}")

    return ReferenceBundle(bundle_path, manifest, arrays)
