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

Every historical scenario forces ``off_diag_corrections=False`` because the
pinned source predates the counting-resolved coherence correction.  That
compatibility mode remains supported; the immutable fixtures must not be
regenerated with the newer default mode.

Three scenario families are provided, and they are used differently:

* ``PINNED_REAL_AMPLITUDE_SCENARIOS`` -- real tunnel amplitudes. These are the
  block-level fixtures stored in the pinned counting reference bundle.
* ``LIVE_COMPLEX_AMPLITUDE_SCENARIOS`` -- a double dot at a generic plaquette flux
  (not a multiple of pi, not roundoff-scale). **No values are stored for
  these.** The pinned source predates the complex-amplitude conjugate-partner
  repair, so it cannot provide trusted expected values for that feature. These
  models instead feed live structural, stationary-RTD, gauge-covariance, and
  independent non-interacting checks.
* ``LIVE_ARBITRARY_SYSTEM_SCENARIOS`` -- larger live-only stress models for
  multi-orbital, multi-lead, rank-deficient, and non-collinear cases.

``STORED_SCENARIOS`` is what the bundle covers;
``RTDNOISE_LIVE_INVARIANT_SCENARIOS`` is the compact live matrix used by the
repeated structural-invariant tests. The larger
``LIVE_ARBITRARY_SYSTEM_SCENARIOS`` matrix is solved once per model.
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


PINNED_REAL_AMPLITUDE_SCENARIOS = (
    "single_level_equal_temperature",
    "single_level_unequal_temperature",
    "double_dot_equal_temperature",
    "double_dot_unequal_temperature",
    "multi_counted_leads",
)

LIVE_COMPLEX_AMPLITUDE_SCENARIOS = (
    "complex_generic_flux_equal_temperature",
    "complex_generic_flux_unequal_temperature",
    "complex_generic_flux_multi_counted_leads",
)

# Live-only stress cases.  They deliberately combine features instead of
# multiplying the test matrix into one model per feature.  The structural test
# solves each case once and checks every invariant on that solve.
LIVE_ARBITRARY_SYSTEM_SCENARIOS = (
    "three_orbital_dense_complex",
    "three_orbital_rank_deficient",
    "two_site_noncollinear",
)

# Values are stored only for the real-amplitude family.
STORED_SCENARIOS = PINNED_REAL_AMPLITUDE_SCENARIOS

# The live invariants run over everything, complex amplitudes included.
RTDNOISE_LIVE_INVARIANT_SCENARIOS = (
    PINNED_REAL_AMPLITUDE_SCENARIOS + LIVE_COMPLEX_AMPLITUDE_SCENARIOS
)

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
    ``pyRTDnoise`` explicitly, which keeps the traversal and scalar integrals
    all-Python regardless of ``QMEQ_BACKEND``. ``True`` requests the generic
    ``RTDnoise`` name: it uses the same traversal but selects compiled scalar
    integrals when the Cython backend is active.
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


def build_rtdnoise_arbitrary_system_scenario(
        scenario, *, use_selected_backend=False):
    """Build a live-only RTDnoise stress scenario.

    These systems are not historical fixtures and must never be added to the
    pinned reference bundle.  They exercise structural identities beyond the
    two-orbital, two-terminal models: dense complex coupling matrices, a
    rank-deficient lead coupling matrix, three unequal lead temperatures,
    interactions, and non-collinear local spin fields.
    """
    kerntype = "RTDnoise" if use_selected_backend else "pyRTDnoise"

    if scenario in {
            "three_orbital_dense_complex",
            "three_orbital_rank_deficient",
    }:
        hsingle = {
            (0, 0): -0.55,
            (1, 1): 0.05,
            (2, 2): 0.65,
            (0, 1): 0.16*np.exp(0.30j),
            (1, 2): 0.11*np.exp(-0.45j),
            (0, 2): 0.07*np.exp(0.80j),
        }
        coulomb = {
            (0, 1, 1, 0): 1.4,
            (0, 2, 2, 0): 1.1,
            (1, 2, 2, 1): 1.3,
        }
        if scenario == "three_orbital_dense_complex":
            amplitudes = np.array([
                [0.080, 0.045*np.exp(0.25j), 0.035*np.exp(-0.40j)],
                [0.050*np.exp(-0.30j), 0.070, 0.040*np.exp(0.55j)],
                [0.030*np.exp(0.60j), 0.055*np.exp(-0.20j), 0.065],
            ])
        else:
            # Every lead couples to the same orbital combination.  The matrix
            # has rank one, while hsingle mixes that combination with the two
            # orthogonal dot modes so the stationary problem remains connected.
            lead_vector = np.array([0.080, 0.055, 0.040])
            orbital_vector = np.array([
                1.0, 0.75*np.exp(0.35j), 0.60*np.exp(-0.20j),
            ])
            amplitudes = np.outer(lead_vector, orbital_vector)

        return qmeq.Builder(
            nsingle=3,
            hsingle=hsingle,
            coulomb=coulomb,
            nleads=3,
            tleads={
                (lead, orbital): amplitudes[lead, orbital]
                for lead in range(3) for orbital in range(3)
            },
            mulst={0: 0.25, 1: -0.05, 2: -0.22},
            tlst={0: 0.18, 1: 0.23, 2: 0.31},
            dband=3000.0,
            kerntype=kerntype,
            itype=1,
            indexing="charge",
            countingleads=[0, 2],
            off_diag_corrections=False,
        )

    if scenario == "two_site_noncollinear":
        # Modes (0, 1) and (2, 3) are the two spin states on the left and
        # right sites.  Complex on-site spin flips point the local transverse
        # fields in different directions; the diagonal splittings add distinct
        # longitudinal components.
        return qmeq.Builder(
            nsingle=4,
            hsingle={
                (0, 0): -0.48,
                (1, 1): -0.32,
                (2, 2): 0.28,
                (3, 3): 0.43,
                (0, 1): 0.09*np.exp(0.20j),
                (2, 3): 0.08*np.exp(-0.65j),
                (0, 2): 0.12,
                (1, 3): 0.10,
                (0, 3): 0.025j,
            },
            coulomb={
                (0, 1, 1, 0): 1.8,
                (2, 3, 3, 2): 1.7,
                (0, 2, 2, 0): 0.9,
                (0, 3, 3, 0): 0.9,
                (1, 2, 2, 1): 0.9,
                (1, 3, 3, 1): 0.9,
            },
            nleads=3,
            tleads={
                (0, 0): 0.070,
                (0, 1): 0.055*np.exp(0.15j),
                (0, 2): 0.018*np.exp(-0.35j),
                (0, 3): 0.014,
                (1, 0): 0.016,
                (1, 1): 0.020*np.exp(0.40j),
                (1, 2): 0.065,
                (1, 3): 0.060*np.exp(-0.25j),
                (2, 0): 0.028*np.exp(0.50j),
                (2, 1): 0.024,
                (2, 2): 0.030*np.exp(-0.45j),
                (2, 3): 0.026*np.exp(0.30j),
            },
            mulst={0: 0.20, 1: -0.18, 2: -0.02},
            tlst={0: 0.17, 1: 0.24, 2: 0.29},
            dband=3000.0,
            kerntype=kerntype,
            itype=1,
            indexing="charge",
            countingleads=[0, 2],
            off_diag_corrections=False,
        )

    raise ValueError(f"Unknown arbitrary-system RTDnoise scenario: {scenario!r}")


_SNAPSHOT_FIELDS = (
    "Lpm_first", "Lpm_second", "Lpm_first_dot", "Lpm_second_dot",
    "phi0", "phi0_first", "phi0_second", "kern_first", "kern_second", "Wdd",
    "current", "current_noise", "current_noise_first",
    "current_noise_o4trunc", "current_noise_matrix",
    "current_noise_matrix_first",
)

# Keep the pinned bundle's original field names immutable while exposing the
# differentiation variable explicitly in the live implementation.
_LIVE_FIELD_NAMES = {
    "Lpm_first_dot": "Lpm_first_dz",
    "Lpm_second_dot": "Lpm_second_dz",
}


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
    snapshot = {
        field: np.asarray(getattr(appr, _LIVE_FIELD_NAMES.get(field, field)))
        for field in _SNAPSHOT_FIELDS
    }
    return snapshot, system
