"""Generate regression references from the last pre-modernization QmeQ 1.1.

This is a provenance tool, not part of the test suite. Run it from an
untouched checkout of ``qmeq@96cc51076458b11f7db81a5d7d8df04c30bf8384``.
The script deliberately selects the pure-Python approaches so that generating
the scientific references does not depend on rebuilding historical extension
modules with a modern compiler and Cython version.

A reproducible invocation from the current repository is::

    git clone --no-checkout . /tmp/qmeq-1.1-reference
    git -C /tmp/qmeq-1.1-reference checkout --detach \
        96cc51076458b11f7db81a5d7d8df04c30bf8384
    cd /tmp/qmeq-1.1-reference
    python -c "import runpy; runpy.run_path( \
        '/path/to/current/scripts/reference_data/generate_qmeq_11_references.py', \
        run_name='__main__')" --output-dir /tmp/qmeq_11_references

The output is a JSON manifest plus a compressed NumPy array archive. Review
and copy the directory to ``qmeq/tests/data/qmeq_11``; tests never regenerate
it automatically. The manifest carries provenance and array metadata, while
the archive losslessly stores real and complex multidimensional arrays.
"""

from __future__ import annotations

import argparse
import importlib.util
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess

import numpy as np
import scipy

import qmeq


SOURCE_COMMIT = "96cc51076458b11f7db81a5d7d8df04c30bf8384"
SOURCE_VERSION = "1.1"
REFERENCE_BUNDLE_SCHEMA = 1


_MODELS_PATH = (
    Path(__file__).resolve().parents[2]
    / "qmeq/tests/qmeq_11_reference_models.py"
)
_MODELS_SPEC = importlib.util.spec_from_file_location(
    "qmeq_11_reference_models", _MODELS_PATH
)
_MODELS = importlib.util.module_from_spec(_MODELS_SPEC)
_MODELS_SPEC.loader.exec_module(_MODELS)
RTD_REFERENCE_SCENARIOS = _MODELS.RTD_REFERENCE_SCENARIOS
_snapshot = _MODELS._snapshot
_rtd_reference_snapshot = _MODELS._rtd_reference_snapshot
build_reference_system = _MODELS.build_reference_system
def _require_pristine_source():
    root = Path(subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], text=True
    ).strip()).resolve()
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != SOURCE_COMMIT:
        raise RuntimeError(
            f"Run from pristine QmeQ 1.1 commit {SOURCE_COMMIT}; found {commit}."
        )
    imported_source = Path(qmeq.__file__).resolve()
    if not imported_source.is_relative_to(root):
        raise RuntimeError(
            f"Imported qmeq from {imported_source}, outside source tree {root}."
        )
    if qmeq.__version__ != SOURCE_VERSION:
        raise RuntimeError(
            f"Expected qmeq {SOURCE_VERSION}, found {qmeq.__version__}."
        )


def _environment():
    try:
        cython_version = importlib.metadata.version("cython")
    except importlib.metadata.PackageNotFoundError:
        cython_version = None
    return {
        "source_commit": SOURCE_COMMIT,
        "qmeq_version": qmeq.__version__,
        "implementation": "pure-python historical source",
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "cython_installed_but_not_used": cython_version,
        "platform": platform.platform(),
    }


def _json_array(value):
    array = np.asarray(value)
    if np.iscomplexobj(array):
        return {
            "real": array.real.tolist(),
            "imag": array.imag.tolist(),
        }
    return array.tolist()


def _state_charges(system):
    charges = np.empty(system.si.nmany, dtype=int)
    for charge, states in enumerate(system.si.statesdm):
        charges[np.asarray(states, dtype=int)] = charge
    return charges


def _fixture_metadata(system, *, model, tolerance, notes=None):
    metadata = {
        "model": model,
        "builder": type(system).__name__,
        "units": "QmeQ natural units with hbar = kB = |e| = 1",
        "classification": "legacy characterization value from QmeQ 1.1",
        "tolerance": {
            "rtol": tolerance[0],
            "atol": tolerance[1],
            "rationale": (
                "Covers floating-point and LAPACK/OpenMP summation-order "
                "variation; it is not an estimate of physical accuracy."
            ),
        },
        "array_ordering": {
            "observables": "lead index",
            "phi0": "QmeQ 1.1 approach-native density-matrix ordering",
            "kern": "final density-matrix coordinate, initial coordinate",
            "rtd_blocks": "lead, final state, initial state",
        },
        "resolved_model": {
            "nsingle": int(system.nsingle),
            "nleads": int(system.nleads),
            "indexing": system.si.indexing,
            "many_body_energies": _json_array(system.qd.Ea),
            "particle_numbers": _json_array(_state_charges(system)),
            "many_body_tunnelling": _json_array(system.leads.Tba),
            "chemical_potentials": _json_array(system.leads.mulst),
            "temperatures": _json_array(system.leads.tlst),
            "band_edges": _json_array(system.leads.dlst),
        },
    }
    if notes is not None:
        metadata["notes"] = notes
    return metadata


def generate_references():
    references = {}
    reference_metadata = {}
    base_cases = (
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
    for approach, itype in base_cases:
        key = f"base/{approach}/itype={itype}"
        system = build_reference_system("base", approach, itype)
        references[key] = _snapshot(system)
        tolerance = (2e-9, 2e-11) if approach == "2vN" else (2e-11, 2e-12)
        reference_metadata[key] = _fixture_metadata(
            system, model="coherent_real_charge", tolerance=tolerance
        )
    for approach in ("Pauli", "Lindblad", "Redfield", "1vN"):
        key = f"elph/{approach}/itype=2/itype_ph=2"
        system = build_reference_system("elph", approach)
        references[key] = _snapshot(system)
        reference_metadata[key] = _fixture_metadata(
            system, model="electron_phonon_spin_symmetry",
            tolerance=(2e-10, 1e-10),
        )
    rtd_scenarios = {}
    rtd_scenario_metadata = {}
    for scenario in RTD_REFERENCE_SCENARIOS:
        snapshot, system = _rtd_reference_snapshot(
            scenario, return_system=True
        )
        rtd_scenarios[scenario] = snapshot
        notes = None
        if scenario == "complex_amplitudes":
            notes = (
                "QmeQ 1.1 warns that RTD energy-current corrections do not "
                "support complex matrix elements; values characterize that "
                "historical behavior rather than validating the observable."
            )
        elif scenario == "spin_symmetry_fallback":
            notes = (
                "QmeQ 1.1 explicitly does not support spin symmetry for RTD "
                "and resolves this request to charge indexing."
            )
        rtd_scenario_metadata[scenario] = _fixture_metadata(
            system, model=scenario, tolerance=(2e-10, 2e-12), notes=notes
        )
        rtd_scenario_metadata[scenario]["off_diag_corrections"] = bool(
            system.off_diag_corrections
        )
        if scenario == "complex_amplitudes":
            reason = (
                "Current QmeQ intentionally differs from QmeQ 1.1 after the "
                "RTD integralD/integralX complex-branch correction. The "
                "stored value remains the QmeQ 1.1 fixture; this fixed "
                "comparison envelope covers only the audited post-1.1 delta."
            )
            rtd_scenario_metadata[scenario]["comparison_overrides"] = {
                "Wdd_second": {"rtol": 2e-10, "atol": 5.5e-6,
                               "reason": reason},
                "Wdd_total": {"rtol": 2e-10, "atol": 5.5e-6,
                              "reason": reason},
                "kern": {"rtol": 2e-10, "atol": 9.5e-6,
                         "reason": reason},
                "current": {"rtol": 2e-10, "atol": 6.1e-7,
                            "reason": reason},
                "phi0": {"rtol": 2e-10, "atol": 1.05e-4,
                         "reason": reason},
            }
    return (references, reference_metadata, rtd_scenarios,
            rtd_scenario_metadata)


def _pack_arrays(references, rtd_scenarios):
    arrays = {}
    reference_map = {}
    rtd_scenario_map = {}
    array_metadata = {}

    def add_array(name, value):
        array = np.asarray(value)
        arrays[name] = array
        array_metadata[name] = {
            "dtype": str(array.dtype),
            "shape": list(array.shape),
        }
        return name

    for case_index, (case, snapshot) in enumerate(sorted(references.items())):
        reference_map[case] = {}
        for field, value in sorted(snapshot.items()):
            name = f"reference_{case_index:02d}_{field}"
            reference_map[case][field] = add_array(name, value)

    for scenario_index, (scenario, snapshot) in enumerate(
            sorted(rtd_scenarios.items())):
        rtd_scenario_map[scenario] = {}
        for field, value in sorted(snapshot.items()):
            name = f"rtd_{scenario_index:02d}_{field}"
            rtd_scenario_map[scenario][field] = add_array(name, value)

    return arrays, reference_map, rtd_scenario_map, array_metadata


def _reference_document(
        environment, reference_map, reference_metadata, rtd_scenario_map,
        rtd_scenario_metadata, array_metadata):
    return {
        "reference_bundle_schema": REFERENCE_BUNDLE_SCHEMA,
        "bundle_id": "qmeq_11",
        "schema_version": 2,
        "description": (
            "QmeQ 1.1 characterization references. These preserve the last "
            "pre-modernization behavior but are not independent proof of "
            "physical correctness."
        ),
        "generator": "scripts/reference_data/generate_qmeq_11_references.py",
        "source_commit": SOURCE_COMMIT,
        "source_version": SOURCE_VERSION,
        "provenance": {
            "classification": "pinned historical source",
            "source": "QmeQ 1.1",
            "source_commit": SOURCE_COMMIT,
            "source_version": SOURCE_VERSION,
        },
        "generation_environment": environment,
        "array_archive": "references.npz",
        "arrays": array_metadata,
        "references": reference_map,
        "reference_metadata": reference_metadata,
        "rtd_scenarios": rtd_scenario_map,
        "rtd_scenario_metadata": rtd_scenario_metadata,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Directory for manifest.json and references.npz.",
    )
    args = parser.parse_args(argv)
    _require_pristine_source()
    (references, reference_metadata, rtd_scenarios,
     rtd_scenario_metadata) = generate_references()
    arrays, reference_map, rtd_scenario_map, array_metadata = _pack_arrays(
        references, rtd_scenarios
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir / "references.npz", **arrays)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            _reference_document(
                _environment(), reference_map, reference_metadata,
                rtd_scenario_map, rtd_scenario_metadata, array_metadata
            ),
            allow_nan=False,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
