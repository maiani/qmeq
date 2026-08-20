"""Generate counting references from Simon Wozny's pristine source checkout.

This is a provenance tool, not part of the test suite.  Run it with the
working directory set to an untouched checkout of
``si8881wo/qmeq@aa1af46dd687c271505d28dbfb7ccce03a8a1739``. The output is a
manifest plus a compressed NumPy archive. Review and copy the
directory to ``qmeq/tests/data/counting_simon``; tests never regenerate it.
"""

import argparse
import json
from pathlib import Path
import platform
import subprocess

import numpy as np
import scipy

import qmeq


SOURCE_COMMIT = "aa1af46dd687c271505d28dbfb7ccce03a8a1739"
REFERENCE_BUNDLE_SCHEMA = 1


def _require_source_commit():
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != SOURCE_COMMIT:
        raise RuntimeError(
            f"Run from the pristine Simon checkout at {SOURCE_COMMIT}; "
            f"found {commit}."
        )


def _first_order(kerntype, gate, bias):
    return qmeq.Builder(
        nsingle=2,
        hsingle={(0, 0): -10 + gate, (1, 1): -12 + gate, (0, 1): 20},
        coulomb={(0, 1, 1, 0): 30},
        nleads=2,
        tleads={(0, 0): 2.0, (1, 1): 1.0, (0, 1): 0.6, (1, 0): 0.1},
        mulst={0: bias / 2, 1: -bias / 2},
        tlst={0: 25.0, 1: 25.0},
        dband={0: 1000.0, 1: 1000.0},
        kerntype=f"py{kerntype}",
        itype=2,
        countingleads=[0],
    )


def _rtd(gate, bias):
    system = qmeq.Builder(
        nsingle=1,
        hsingle={(0, 0): gate},
        nleads=2,
        tleads={
            (0, 0): np.sqrt(0.08 / (2 * np.pi)),
            (1, 0): np.sqrt(0.12 / (2 * np.pi)),
        },
        mulst={0: bias / 2, 1: -bias / 2},
        tlst={0: 1.0, 1: 1.0},
        dband={0: 100.0, 1: 100.0},
        kerntype="pyRTDnoise",
        countingleads=[0],
    )
    system.off_diag_corrections = False
    return system


def _array_metadata(arrays):
    return {
        name: {"shape": list(value.shape), "dtype": str(value.dtype)}
        for name, value in sorted(arrays.items())
    }


def generate_references():
    _require_source_commit()
    first_order_points = np.asarray(
        [(-15.0, 4.0), (0.0, 5.0), (18.0, 12.0)]
    )
    rtd_points = np.asarray([(-1.5, 4.0), (0.0, 6.0), (1.25, 8.0)])
    arrays = {
        "first_order_points": first_order_points,
        "rtd_points": rtd_points,
    }
    first_order = {}
    for kerntype in ("Pauli", "Lindblad", "Redfield", "1vN"):
        name = f"first_order_{kerntype}"
        values = []
        for gate, bias in first_order_points:
            system = _first_order(kerntype, gate, bias)
            system.solve()
            values.append(np.asarray(system.current_noise))
        arrays[name] = np.asarray(values)
        first_order[kerntype] = name

    rtd = []
    for index, (gate, bias) in enumerate(rtd_points):
        system = _rtd(gate, bias)
        system.solve()
        mapping = {
            "full": f"rtd_{index}_full",
            "first": f"rtd_{index}_first",
            "o4trunc": f"rtd_{index}_o4trunc",
        }
        arrays[mapping["full"]] = np.asarray(system.current_noise)
        arrays[mapping["first"]] = np.asarray(system.appr.current_noise_first)
        arrays[mapping["o4trunc"]] = np.asarray(
            system.appr.current_noise_o4trunc
        )
        rtd.append(mapping)

    manifest = {
        "reference_bundle_schema": REFERENCE_BUNDLE_SCHEMA,
        "bundle_id": "counting_simon",
        "description": (
            "Counting-statistics snapshots from Simon Wozny's final source "
            "commit."
        ),
        "array_archive": "references.npz",
        "arrays": _array_metadata(arrays),
        "provenance": {
            "classification": "pinned historical source",
            "status": "recorded",
            "source": "si8881wo/qmeq",
            "source_commit": SOURCE_COMMIT,
            "generation_environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
            },
        },
        "snapshots": {
            "first_order_points": "first_order_points",
            "first_order": first_order,
            "rtd_points": "rtd_points",
            "rtd": rtd,
        },
    }
    return manifest, arrays


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest, arrays = generate_references()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir / "references.npz", **arrays)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
