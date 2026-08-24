"""Generate counting references from the pristine reference-source checkout.

This is a provenance tool, not part of the test suite.  Run it with the
working directory set to an untouched checkout of
``si8881wo/qmeq@aa1af46dd687c271505d28dbfb7ccce03a8a1739``. The output is a
manifest plus a compressed NumPy archive. Review and copy the directory to
``qmeq/tests/data/counting_reference``; tests never regenerate it.
"""

import argparse
import importlib.util
import json
from pathlib import Path
import platform
import subprocess

import numpy as np
import scipy

import qmeq


SOURCE_COMMIT = "aa1af46dd687c271505d28dbfb7ccce03a8a1739"
REFERENCE_BUNDLE_SCHEMA = 1
MODEL_DEFINITIONS = (
    Path(__file__).resolve().parents[2]
    / "qmeq" / "tests" / "rtdnoise_reference_models.py"
)
RTDNOISE_FIELDS = (
    "Lpm_first",
    "Lpm_second",
    "Lpm_first_dot",
    "Lpm_second_dot",
    "phi0",
    "phi0_first",
    "phi0_second",
    "kern_first",
    "kern_second",
    "Wdd",
    "current",
    "current_noise",
    "current_noise_first",
    "current_noise_o4trunc",
)


def _require_source_commit():
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != SOURCE_COMMIT:
        raise RuntimeError(
            f"Run from the pristine reference checkout at {SOURCE_COMMIT}; "
            f"found {commit}."
        )

    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], text=True
    ).strip()
    if dirty:
        raise RuntimeError(
            "Run from a pristine reference checkout; the source tree is dirty."
        )


def _load_rtdnoise_model_definitions():
    """Load current scenario definitions against the pinned source package."""
    spec = importlib.util.spec_from_file_location(
        "qmeq_rtdnoise_reference_models", MODEL_DEFINITIONS
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load model definitions from {MODEL_DEFINITIONS}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _json_array(value):
    array = np.asarray(value)
    if np.iscomplexobj(array):
        return {"real": array.real.tolist(), "imag": array.imag.tolist()}
    return array.tolist()


def _resolved_model(system):
    return {
        "nsingle": int(system.nsingle),
        "nleads": int(system.nleads),
        "indexing": system.si.indexing,
        "countingleads": list(system.funcp.countingleads),
        "off_diag_corrections": bool(system.off_diag_corrections),
        "many_body_energies": _json_array(system.qd.Ea),
        "many_body_tunnelling": _json_array(system.leads.Tba),
        "chemical_potentials": _json_array(system.leads.mulst),
        "temperatures": _json_array(system.leads.tlst),
        "band_edges": _json_array(system.leads.dlst),
    }


def _rtdnoise_snapshot(system):
    """Return physical source arrays, excluding the solver's replaced row.

    The pinned source's ``solve_kern_first`` leaves the normalization condition in
    ``kern_first[norm_row]``. The physical first-order kernel is reconstructed
    directly from the transfer-resolved blocks, exactly as ``generate_kern``
    assembles it before the stationary solve. Current QmeQ restores that row.
    """
    snapshot = {
        field: np.asarray(getattr(system.appr, field)).copy()
        for field in RTDNOISE_FIELDS
    }
    snapshot["kern_first"] = np.sum(
        system.appr.Lpm_first.real, axis=(0, 1)
    )
    snapshot["kern_second"] = np.sum(
        system.appr.Lpm_second.real, axis=(0, 1, 2, 3)
    )
    return snapshot


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
    rtdnoise_models = _load_rtdnoise_model_definitions()
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

    rtdnoise_scenarios = {}
    rtdnoise_scenario_metadata = {}
    for index, scenario in enumerate(rtdnoise_models.STORED_SCENARIOS):
        system = rtdnoise_models.build_rtdnoise_scenario(scenario)
        # The pinned Builder accepted this constructor argument but did not
        # forward it into FunctionProperties. Set the public property directly,
        # matching the reference generator above.
        system.off_diag_corrections = False
        system.solve()
        mapping = {}
        snapshot = _rtdnoise_snapshot(system)
        for field in RTDNOISE_FIELDS:
            name = f"rtdnoise_{index:02d}_{field}"
            arrays[name] = snapshot[field]
            mapping[field] = name
        rtdnoise_scenarios[scenario] = mapping
        rtdnoise_scenario_metadata[scenario] = {
            "classification": "pinned historical source",
            "resolved_model": _resolved_model(system),
            "array_ordering": {
                "Lpm_first": (
                    "lead, transfer (Python-indexed -1/0/1), final state, "
                    "initial state"
                ),
                "Lpm_second": (
                    "lead0, lead1, transfer0 (Python-indexed -1/0/1), "
                    "transfer1 (Python-indexed -1/0/1), final state, "
                    "initial state"
                ),
                "Lpm_first_dot": "same layout as Lpm_first; Laplace derivative",
                "Lpm_second_dot": "same layout as Lpm_second; finite-difference Laplace derivative",
                "Wdd": "lead, final state, initial state",
                "kern_first/kern_second": "final coordinate, initial coordinate",
                "phi0/phi0_first/phi0_second": (
                    "RTDnoise-native diagonal density-matrix ordering"
                ),
                "current": "lead index",
                "current_noise/current_noise_first": (
                    "aggregate current, aggregate zero-frequency noise"
                ),
                "current_noise_o4trunc": (
                    "first-order current, second-order current correction, "
                    "first-order noise, second-order noise correction"
                ),
            },
            "source_representation": {
                "kern_first": (
                    "Reconstructed from sum(Lpm_first.real) because the pinned "
                    "stationary solve leaves its normalization row in the "
                    "workspace array."
                ),
                "kern_second": "Reconstructed from sum(Lpm_second.real).",
            },
            "tolerance": {
                "rtol": 2e-9,
                "atol": 2e-12,
                "rationale": (
                    "Covers floating-point summation-order variation between "
                    "the pinned source generation environment and supported "
                    "current environments; it is not an estimate of physical "
                    "accuracy."
                ),
            },
        }

    manifest = {
        "reference_bundle_schema": REFERENCE_BUNDLE_SCHEMA,
        "bundle_id": "counting_reference",
        "description": (
            "Counting-statistics outputs and RTDnoise block-level snapshots "
            "generated from the final reference-source commit."
        ),
        "generator": "scripts/reference_data/generate_counting_reference.py",
        "rtdnoise_model_definitions": (
            "qmeq/tests/rtdnoise_reference_models.py"
        ),
        "array_archive": "references.npz",
        "arrays": _array_metadata(arrays),
        "provenance": {
            "classification": "pinned historical source",
            "status": "recorded",
            "source": "si8881wo/qmeq",
            "source_commit": SOURCE_COMMIT,
            "source_tag": "reference/rtdnoise-source",
            "source_version": qmeq.__version__,
            "source_author": "Simon Wozny",
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
            "rtdnoise_scenarios": rtdnoise_scenarios,
        },
        "rtdnoise_scenario_metadata": rtdnoise_scenario_metadata,
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
