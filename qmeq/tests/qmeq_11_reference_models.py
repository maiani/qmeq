"""Model builders shared by QmeQ 1.1 regression tests and their generator."""

import numpy as np

import qmeq
from qmeq.specfunc import Func


class _OhmicBath(Func):
    def eval(self, energy):
        return 3.8804e-4 * energy


def _core_model(kerntype, itype):
    """A small coherent double dot used by all electronic approaches."""
    return qmeq.Builder(
        nsingle=2,
        hsingle={(0, 0): -10.0, (1, 1): -12.0, (0, 1): 20.0},
        coulomb={(0, 1, 1, 0): 30.0},
        nleads=2,
        tleads={(0, 0): 2.0, (1, 1): 1.0,
                (0, 1): 0.6, (1, 0): 0.1},
        mulst={0: 2.5, 1: -2.5},
        tlst={0: 25.0, 1: 25.0},
        dband={0: 1000.0, 1: 1000.0},
        kerntype=kerntype,
        itype=itype,
        indexing="charge",
        kpnt=64,
    )


def _elph_model(kerntype):
    """The compact electron-phonon model shared by all four elph methods."""
    phase = np.pi / 3 * 1j
    overlap = np.exp(phase / 2) * np.exp(-(120.0**2) / (4 * 5.8**2))
    return qmeq.BuilderElPh(
        nsingle=4,
        hsingle={(0, 0): 0.05, (1, 1): -0.05, (0, 1): 0.05},
        coulomb={(0, 0, 0, 0): 12.0,
                 (1, 1, 1, 1): 12.0,
                 (0, 1, 1, 0): 2.5},
        nleads=4,
        tleads={(0, 0): np.sqrt(9.0e-5 / (2 * np.pi)),
                (1, 1): np.sqrt(9.0e-5 / (2 * np.pi))},
        mulst={0: 0.0, 1: 0.0},
        tlst={0: 0.005, 1: 0.005},
        dband=50.0,
        nbaths=1,
        velph={(0, 0, 0): 1.0,
               (0, 1, 1): np.exp(phase),
               (0, 0, 1): overlap,
               (0, 1, 0): overlap},
        tlst_ph={0: 0.025},
        dband_ph={0: [1.0e-8, 100.0]},
        bath_func=[_OhmicBath()],
        kerntype=kerntype,
        itype=2,
        itype_ph=2,
        indexing="ssq",
        symmetry="spin",
    )


def _solve(system, *, niter=None, **kwargs):
    if niter is None:
        system.solve(**kwargs)
    else:
        system.solve(niter=niter, **kwargs)
    return system


def build_reference_system(
        family, approach, itype=2, *, use_selected_backend=False):
    """Build and solve one scenario under either QmeQ 1.1 or current QmeQ."""
    kerntype = approach if use_selected_backend else f"py{approach}"
    if family == "base":
        system = _core_model(kerntype, itype)
        return _solve(system, niter=3 if approach == "2vN" else None)
    if family == "elph":
        return _solve(_elph_model(kerntype))
    raise ValueError(f"Unknown reference family: {family!r}")


def _snapshot(system):
    appr = system.appr
    result = {
        "current": np.asarray(system.current),
        "energy_current": np.asarray(system.energy_current),
        "heat_current": np.asarray(system.heat_current),
        "phi0": np.asarray(system.phi0),
        "kern": np.asarray(appr.kern),
    }
    return result


def _single_level_rtd_model(kerntype):
    return qmeq.Builder(
        nsingle=1,
        hsingle={(0, 0): 0.3},
        nleads=2,
        tleads={(0, 0): 0.05, (1, 0): 0.04},
        mulst={0: 0.0, 1: 0.0},
        tlst={0: 0.2, 1: 0.2},
        dband={0: 1000.0, 1: 1000.0},
        kerntype=kerntype,
        itype=1,
        indexing="charge",
    )


def _complex_rtd_model(kerntype):
    return qmeq.Builder(
        nsingle=2,
        hsingle={(0, 0): -0.3, (1, 1): 0.4, (0, 1): 0.12},
        coulomb={(0, 1, 1, 0): 2.0},
        nleads=2,
        tleads={(0, 0): 0.08,
                (0, 1): 0.05 * np.exp(0.4j),
                (1, 0): 0.03 * np.exp(-0.2j),
                (1, 1): 0.07 * np.exp(0.7j)},
        mulst={0: 0.15, 1: -0.15},
        tlst={0: 0.25, 1: 0.25},
        dband={0: 100.0, 1: 100.0},
        kerntype=kerntype,
        itype=1,
        indexing="charge",
    )


def _unequal_temperature_rtd_model(kerntype):
    system = qmeq.Builder(
        nsingle=1,
        hsingle={(0, 0): 0.3},
        nleads=2,
        tleads={(0, 0): 0.05, (1, 0): 0.04},
        mulst={0: 0.0, 1: 0.0},
        tlst={0: 0.1, 1: 0.2},
        dband={0: 1000.0, 1: 1000.0},
        kerntype=kerntype,
        itype=1,
        indexing="charge",
    )
    system.off_diag_corrections = False
    return system


def _spin_symmetry_fallback_rtd_model(kerntype):
    amplitude = np.sqrt(0.2 / (2 * np.pi))
    return qmeq.Builder(
        nsingle=2,
        hsingle={(0, 0): 0.1, (1, 1): 0.1},
        coulomb={(0, 1, 1, 0): 2.5},
        nleads=4,
        tleads={(0, 0): amplitude, (1, 0): 0.8 * amplitude,
                (2, 1): amplitude, (3, 1): 0.8 * amplitude},
        mulst={0: 0.2, 1: -0.2, 2: 0.2, 3: -0.2},
        tlst={0: 0.3, 1: 0.3, 2: 0.3, 3: 0.3},
        dband=100.0,
        kerntype=kerntype,
        itype=1,
        indexing=None,
        symmetry="spin",
    )


def _many_body_rtd_model(kerntype):
    hybridization = 0.2
    amplitude = 0.08
    tleads = {
        (0, 0): amplitude, (0, 1): amplitude / 4,
        (1, 1): 0.9 * amplitude, (1, 0): amplitude / 5,
    }
    source = qmeq.Builder(
        nsingle=2,
        hsingle={(0, 0): -0.3, (1, 1): 0.1,
                 (0, 1): hybridization},
        coulomb={(0, 1, 1, 0): 3.0},
        nleads=2,
        tleads=tleads,
        mulst={0: 0.2, 1: -0.2},
        tlst={0: 0.25, 1: 0.25},
        dband=100.0,
        kerntype="pyPauli",
        indexing="charge",
    )
    _solve(source, masterq=False)
    system = qmeq.BuilderManyBody(
        Ea=np.asarray(source.qd.Ea).copy(),
        Na=[0, 1, 1, 2],
        Tba=np.asarray(source.Tba).copy(),
        mulst={0: 0.2, 1: -0.2},
        tlst={0: 0.25, 1: 0.25},
        dband=100.0,
        kerntype=kerntype,
        itype=1,
    )
    system.nsingle = 2
    system.tleads_array = np.array([
        [tleads[(0, 0)], tleads[(0, 1)]],
        [tleads[(1, 0)], tleads[(1, 1)]],
    ], dtype=complex)
    return system


RTD_REFERENCE_SCENARIOS = (
    "single_level_equilibrium",
    "coherent_real_offdiag_on",
    "coherent_real_offdiag_off",
    "complex_amplitudes",
    "unequal_temperatures",
    "many_body",
    "spin_symmetry_fallback",
)


def build_rtd_reference_system(scenario, *, use_selected_backend=False):
    """Build one RTD reference scenario under historical or current QmeQ."""
    kerntype = "RTD" if use_selected_backend else "pyRTD"
    if scenario == "single_level_equilibrium":
        return _single_level_rtd_model(kerntype)
    if scenario in {
            "coherent_real_offdiag_on", "coherent_real_offdiag_off"}:
        system = _core_model(kerntype, 1)
        system.off_diag_corrections = scenario.endswith("_on")
        return system
    if scenario == "complex_amplitudes":
        return _complex_rtd_model(kerntype)
    if scenario == "unequal_temperatures":
        return _unequal_temperature_rtd_model(kerntype)
    if scenario == "many_body":
        return _many_body_rtd_model(kerntype)
    if scenario == "spin_symmetry_fallback":
        return _spin_symmetry_fallback_rtd_model(kerntype)
    raise ValueError(f"Unknown RTD reference scenario: {scenario!r}")


def solve_rtd_reference_system(system, scenario):
    if scenario == "many_body":
        return _solve(system, qdq=False, rotateq=False)
    return _solve(system)


def _rtd_reference_snapshot(scenario, *, return_system=False):
    full = solve_rtd_reference_system(
        build_rtd_reference_system(scenario), scenario
    )
    no_elimination = build_rtd_reference_system(scenario)
    no_elimination.off_diag_corrections = False
    solve_rtd_reference_system(no_elimination, scenario)

    sequential = build_rtd_reference_system(scenario)
    sequential.off_diag_corrections = False
    original_second_order = sequential.appr.generate_col_diag_kern_2nd_order
    sequential.appr.generate_col_diag_kern_2nd_order = lambda *args: None
    try:
        solve_rtd_reference_system(sequential, scenario)
    finally:
        sequential.appr.generate_col_diag_kern_2nd_order = original_second_order

    appr = full.appr
    nleads = full.nleads
    npauli = full.si.npauli
    ncoherences = full.si.ndm0r - npauli

    def optional_block(value, shape):
        if value is None:
            return np.zeros(shape, dtype=float)
        return np.asarray(value)

    result = _snapshot(full)
    result.update({
        "Wdd_total": np.asarray(appr.Wdd),
        "Wdd_first": np.asarray(sequential.appr.Wdd),
        "Wdd_second": np.asarray(
            no_elimination.appr.Wdd - sequential.appr.Wdd
        ),
        "Wdd_elimination": np.asarray(
            appr.Wdd - no_elimination.appr.Wdd
        ),
        "ReWdn": optional_block(
            appr.ReWdn, (nleads, npauli, ncoherences)
        ),
        "ImWdn": optional_block(
            appr.ImWdn, (nleads, npauli, ncoherences)
        ),
        "ReWnd": optional_block(
            appr.ReWnd, (nleads, ncoherences, npauli)
        ),
        "ImWnd": optional_block(
            appr.ImWnd, (nleads, ncoherences, npauli)
        ),
        "inverse_Lnn": optional_block(
            appr.Lnn, (ncoherences, ncoherences)
        ),
        "WE1": np.asarray(appr.WE1),
        "WE2": np.asarray(appr.WE2),
    })
    if return_system:
        return result, full
    return result


def _rtd_block_snapshot():
    """Compatibility helper for the original coherent-core block fixture."""
    snapshot = _rtd_reference_snapshot("coherent_real_offdiag_on")
    return {key: value for key, value in snapshot.items()
            if key not in {"current", "energy_current", "heat_current",
                           "phi0", "kern"}}
