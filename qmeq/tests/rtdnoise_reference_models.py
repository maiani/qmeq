"""Models shared by pinned RTDnoise references and live validation tests.

These public-input definitions feed both the maintainer-only generator
(``scripts/reference_data/generate_counting_reference.py``) and the tests that
compare a live solve against the stored bundle
(``qmeq/tests/test_rtdnoise_references.py``), so the two can never
silently drift apart the way two independently retyped model definitions
could.

The stored values are generated with these definitions against the pinned
reference-source commit, not against the current working tree. Current QmeQ solves
the same models during testing and must reproduce the pinned block-level and
cumulant arrays within the manifest tolerances.

Every scenario forces ``off_diag_corrections=False``. ``RTDnoise`` raises
``NotImplementedError`` for ``off_diag_corrections=True``
(``qmeq/approach/base/RTDnoise.py:135``); until that limitation is removed,
``False`` is the only value RTDnoise accepts.

Two scenario families are provided, and they are used differently:

* ``BASELINE_SCENARIOS`` -- real tunnel amplitudes. These are the block-level
  fixtures stored in the pinned counting reference bundle.
* ``COMPLEX_AMPLITUDE_SCENARIOS`` -- a double dot at a generic plaquette flux
  (not a multiple of pi, not roundoff-scale). **No values are stored for
  these.** RTDnoise's second-order kernel is known to be wrong for complex
  tunnel amplitudes, so a stored value would be a recorded defect rather than
  a baseline. They exist to feed the live structural invariants, which hold
  regardless of that defect and would catch a regression there.

``STORED_SCENARIOS`` is what the bundle covers; ``INVARIANT_SCENARIOS`` is what
the structural-invariant suite runs over.
"""

from __future__ import annotations

import numpy as np

import qmeq


# A generic plaquette flux: the phase of t(0,0)*t(1,1)*conj(t(0,1))*conj(t(1,0))
# for the complex-amplitude models below. Deliberately not a multiple of pi
# (and nowhere near it) and far above roundoff scale -- a *physical* phase, not
# the 2e-18 one already pinned by test_RTD_ignores_roundoff_scale_tunnel_phase
# in test_rtd_regressions.py.
_COMPLEX_TLEADS = {
    (0, 0): 0.09,
    (0, 1): 0.05 * np.exp(1j * 0.5),
    (1, 0): 0.04 * np.exp(-1j * 0.3),
    (1, 1): 0.06 * np.exp(1j * 0.65),
}
_COMPLEX_PLAQUETTE_FLUX = float(np.angle(
    _COMPLEX_TLEADS[(0, 0)] * _COMPLEX_TLEADS[(1, 1)]
    * _COMPLEX_TLEADS[(0, 1)].conjugate() * _COMPLEX_TLEADS[(1, 0)].conjugate()
))


def _single_level_dot(*, tlst, mulst, countingleads, kerntype):
    return qmeq.Builder(
        nsingle=1,
        hsingle={(0, 0): 0.3},
        nleads=len(tlst),
        tleads={(lead, 0): amplitude for lead, amplitude in
                zip(sorted(tlst), (0.05, 0.04, 0.035)[:len(tlst)])},
        mulst=mulst,
        tlst=tlst,
        dband=1000.0,
        kerntype=kerntype,
        itype=1,
        indexing="charge",
        countingleads=countingleads,
        off_diag_corrections=False,
    )


def _double_dot(*, tlst, mulst, countingleads, kerntype):
    return qmeq.Builder(
        nsingle=2,
        hsingle={(0, 0): -0.3, (1, 1): 0.4, (0, 1): 0.12},
        coulomb={(0, 1, 1, 0): 2.0},
        nleads=2,
        tleads={(0, 0): 0.08, (0, 1): 0.05, (1, 0): 0.03, (1, 1): 0.07},
        mulst=mulst,
        tlst=tlst,
        dband=2000.0,
        kerntype=kerntype,
        itype=1,
        indexing="charge",
        countingleads=countingleads,
        off_diag_corrections=False,
    )


def _complex_flux_double_dot(*, tlst, mulst, countingleads, kerntype):
    return qmeq.Builder(
        nsingle=2,
        hsingle={(0, 0): -0.25, (1, 1): 0.35, (0, 1): 0.1},
        coulomb={(0, 1, 1, 0): 1.8},
        nleads=2,
        tleads=dict(_COMPLEX_TLEADS),
        mulst=mulst,
        tlst=tlst,
        dband=2000.0,
        kerntype=kerntype,
        itype=1,
        indexing="charge",
        countingleads=countingleads,
        off_diag_corrections=False,
    )


BASELINE_SCENARIOS = (
    "single_level_equal_temperature",
    "single_level_unequal_temperature",
    "double_dot_equal_temperature",
    "double_dot_unequal_temperature",
    "multi_counted_leads",
)

COMPLEX_AMPLITUDE_SCENARIOS = (
    "complex_generic_flux_equal_temperature",
    "complex_generic_flux_unequal_temperature",
    "complex_generic_flux_multi_counted_leads",
)

# Values are stored only for the real-amplitude family.
STORED_SCENARIOS = BASELINE_SCENARIOS

# The live invariants run over everything, complex amplitudes included.
INVARIANT_SCENARIOS = BASELINE_SCENARIOS + COMPLEX_AMPLITUDE_SCENARIOS

# Back-compatible alias for the set the bundle covers.
RTDNOISE_CHARACTERIZATION_SCENARIOS = STORED_SCENARIOS


def build_equilibrium_scenario(*, kerntype="pyRTDnoise"):
    """A single-level dot at zero bias and equal lead temperatures.

    Not part of ``RTDNOISE_CHARACTERIZATION_SCENARIOS``: it is used only by
    the structural "equilibrium zero current" invariant test, which is
    computed from a live solve and never compared against the stored bundle.
    """
    return _single_level_dot(
        tlst={0: 0.2, 1: 0.2}, mulst={0: 0.0, 1: 0.0},
        countingleads=[0], kerntype=kerntype,
    )


def build_rtdnoise_scenario(scenario, *, use_selected_backend=False):
    """Build (but do not solve) one RTDnoise reference scenario.

    ``kerntype`` follows the ``qmeq_11_reference_models.py`` convention:
    ``use_selected_backend=False`` (the default) always requests
    ``pyRTDnoise`` explicitly, since ``RTDnoise``/``pyRTDnoise`` has no
    compiled counterpart and both backends route to the same
    pure-Python class regardless of ``QMEQ_BACKEND``; ``True`` requests the
    generic ``RTDnoise`` alias instead, to exercise that alias resolution
    itself under both forced backends.
    """
    kerntype = "RTDnoise" if use_selected_backend else "pyRTDnoise"
    if scenario == "single_level_equal_temperature":
        return _single_level_dot(
            tlst={0: 0.2, 1: 0.2}, mulst={0: 0.1, 1: -0.1},
            countingleads=[0], kerntype=kerntype,
        )
    if scenario == "single_level_unequal_temperature":
        return _single_level_dot(
            tlst={0: 0.15, 1: 0.25}, mulst={0: 0.1, 1: -0.1},
            countingleads=[0], kerntype=kerntype,
        )
    if scenario == "double_dot_equal_temperature":
        return _double_dot(
            tlst={0: 0.25, 1: 0.25}, mulst={0: 0.15, 1: -0.15},
            countingleads=[0], kerntype=kerntype,
        )
    if scenario == "double_dot_unequal_temperature":
        return _double_dot(
            tlst={0: 0.2, 1: 0.3}, mulst={0: 0.15, 1: -0.15},
            countingleads=[0], kerntype=kerntype,
        )
    if scenario == "multi_counted_leads":
        return _single_level_dot(
            tlst={0: 0.2, 1: 0.2, 2: 0.2},
            mulst={0: 0.15, 1: -0.05, 2: -0.2},
            countingleads=[0, 1], kerntype=kerntype,
        )
    if scenario == "complex_generic_flux_equal_temperature":
        return _complex_flux_double_dot(
            tlst={0: 0.2, 1: 0.2}, mulst={0: 0.12, 1: -0.12},
            countingleads=[0], kerntype=kerntype,
        )
    if scenario == "complex_generic_flux_unequal_temperature":
        return _complex_flux_double_dot(
            tlst={0: 0.15, 1: 0.25}, mulst={0: 0.12, 1: -0.12},
            countingleads=[0], kerntype=kerntype,
        )
    if scenario == "complex_generic_flux_multi_counted_leads":
        return _complex_flux_double_dot(
            tlst={0: 0.2, 1: 0.2}, mulst={0: 0.12, 1: -0.12},
            countingleads=[0, 1], kerntype=kerntype,
        )
    raise ValueError(f"Unknown RTDnoise reference scenario: {scenario!r}")


_SNAPSHOT_FIELDS = (
    "Lpm_first", "Lpm_second", "Lpm_first_dot", "Lpm_second_dot",
    "phi0", "phi0_first", "phi0_second", "kern_first", "kern_second", "Wdd",
    "current", "current_noise", "current_noise_first",
    "current_noise_o4trunc", "current_noise_matrix",
    "current_noise_matrix_first",
)


def solve_rtdnoise_scenario(system, scenario):
    """Solve a scenario, silencing one expected and irrelevant warning.

    The complex-amplitude scenarios trip RTD's "complex matrix
    elements are not supported for the RTD energy current" warning
    (``qmeq/approach/base/RTD.py:409-420``), because ``RTDnoise.generate_kern``
    still populates the energy-current blocks ``WE1``/``WE2`` even though
    none of the fields this bundle captures depend on them (RTDnoise reports
    only the counting cumulants, never ``energy_current``/``heat_current``).
    The warning is genuine and not silenced anywhere else in the suite; it is
    filtered here, deliberately and only for this module, because emitting it
    on every characterization solve would be noise about a field this bundle
    does not use, not a signal about anything this bundle characterizes.
    """
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=r".*energy_current and heat_current.*",
            category=qmeq.QmeqRuntimeWarning,
        )
        system.solve()
    return system


def rtdnoise_scenario_snapshot(scenario, *, use_selected_backend=False):
    """Build, solve, and snapshot one scenario's block-level and cumulant arrays."""
    system = solve_rtdnoise_scenario(
        build_rtdnoise_scenario(scenario, use_selected_backend=use_selected_backend),
        scenario,
    )
    appr = system.appr
    snapshot = {field: np.asarray(getattr(appr, field)) for field in _SNAPSHOT_FIELDS}
    return snapshot, system
