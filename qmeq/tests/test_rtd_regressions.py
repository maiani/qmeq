import numpy as np
import pytest
import warnings

import qmeq
from qmeq.approach.base.RTD import RTDBandwidthWarning


def _roundoff_phase_current(imaginary_part, kerntype="RTD"):
    temperature = 0.02
    gamma = 0.04
    hsingle = {
        (0, 0): 0.0084,
        (1, 1): 0.0044,
        (2, 2): -0.0048,
        (3, 3): -0.0080,
    }
    coulomb = {
        (0, 1, 1, 0): 5.0,
        (2, 3, 3, 2): 5.0,
    }
    for left in (0, 1):
        for right in (2, 3):
            coulomb[(left, right, right, left)] = 5.0

    amplitude = gamma/np.sqrt(2*np.pi)
    tleads = {
        (0, 0): amplitude,
        (0, 2): amplitude,
        (1, 0): amplitude,
        (1, 2): complex(-amplitude, imaginary_part),
        (2, 1): amplitude,
        (2, 3): amplitude,
        (3, 1): amplitude,
        (3, 3): complex(-amplitude, imaginary_part),
    }
    system = qmeq.Builder(
        4,
        hsingle,
        coulomb,
        4,
        tleads,
        {0: 0.2, 1: -0.2, 2: 0.2, 3: -0.2},
        {lead: temperature for lead in range(4)},
        50.0,
        kerntype=kerntype,
        indexing="charge",
        itype=1,
    )
    system.solve()
    return float(system.current[0]+system.current[2])


def test_RTD_ignores_roundoff_scale_tunnel_phase():
    exactly_real = _roundoff_phase_current(0.0)
    roundoff_phase = _roundoff_phase_current(2e-18)

    assert np.isclose(roundoff_phase, exactly_real, rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("kerntype", ["pyRTD", "RTD"])
def test_RTD_complex_energy_current_emits_runtime_warning(kerntype):
    with pytest.warns(
        qmeq.QmeqRuntimeWarning, match="energy_current and heat_current"
    ):
        _roundoff_phase_current(1e-3, kerntype)


@pytest.mark.parametrize("kerntype", ["pyRTD", "RTD"])
def test_RTD_missing_single_particle_amplitudes_warns(kerntype):
    reference = qmeq.Builder(
        nsingle=1,
        hsingle={(0, 0): 0.1},
        nleads=2,
        tleads={(0, 0): 0.1, (1, 0): 0.1},
        mulst=[0.1, -0.1],
        tlst=[1.0, 1.0],
        dband=1000.0,
        kerntype="pyRTD",
        itype=1,
    )
    reference.solve()
    system = qmeq.BuilderManyBody(
        Ea=reference.Ea,
        Na=[0, 1],
        Tba=reference.Tba,
        mulst=[0.1, -0.1],
        tlst=[1.0, 1.0],
        dband=1000.0,
        kerntype=kerntype,
        itype=1,
    )

    with pytest.warns(qmeq.QmeqRuntimeWarning, match="No single-particle"):
        system.solve(qdq=False, rotateq=False)


def _thermal_single_level(dband, kerntype="pyRTD"):
    return qmeq.Builder(
        nsingle=1,
        hsingle={(0, 0): 0.3},
        nleads=2,
        tleads={(0, 0): 0.05, (1, 0): 0.04},
        mulst={0: 0.0, 1: 0.0},
        tlst={0: 0.1, 1: 0.2},
        dband=dband,
        kerntype=kerntype,
        off_diag_corrections=False,
    )


def test_RTD_warns_when_unequal_temperature_cutoff_is_not_wide():
    with pytest.warns(RTDBandwidthWarning, match="cutoff-to-transport-scale"):
        _thermal_single_level(10.0).solve()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _thermal_single_level(1000.0).solve()
    assert not [warning for warning in caught
                if issubclass(warning.category, RTDBandwidthWarning)]


def test_RTD_Ozaki_expansion_covers_the_widest_lead():
    system = _thermal_single_level({0: 10.0, 1: 1000.0})
    with pytest.warns(RTDBandwidthWarning):
        system.solve()
    assert system.appr.BW_Ozaki_expansion == pytest.approx(10000.0)


def test_RTD_unequal_temperature_python_selected_backend_parity():
    python_system = _thermal_single_level(1000.0, kerntype="pyRTD")
    selected_system = _thermal_single_level(1000.0, kerntype="RTD")
    python_system.solve()
    selected_system.solve()
    np.testing.assert_allclose(
        selected_system.current, python_system.current, rtol=1e-11, atol=1e-13
    )
    np.testing.assert_allclose(
        selected_system.phi0, python_system.phi0,
        rtol=1e-11, atol=1e-13,
    )


@pytest.mark.parametrize("kerntype", ["RTD", "pyRTD"])
def test_RTD_many_body_construction_matches_Builder(kerntype):
    hybridization = 5.0
    coulomb_u = 500.0
    amplitude = 1.0/np.sqrt(2*np.pi)
    tleads = {
        (0, 0): amplitude, (0, 1): amplitude/10,
        (1, 1): amplitude, (1, 0): amplitude/10,
    }
    lead_temperatures = [20.0, 10.0]

    reference = qmeq.Builder(
        nsingle=2, nleads=2,
        hsingle={(0, 0): -30.0, (1, 1): -30.0, (0, 1): hybridization},
        coulomb={(0, 1, 1, 0): coulomb_u},
        tleads=tleads,
        mulst=[0.0, 0.0], tlst=lead_temperatures, dband=5.0e5,
        kerntype=kerntype, itype=1,
    )
    reference.solve()

    amplitude_array = np.zeros((2, 2), dtype=complex)
    for (lead, level), value in tleads.items():
        amplitude_array[lead, level] = value

    # Passing the compiled kerntype directly used to size a per-thread RTD
    # buffer from si.npauli before the many-body state indexing (Na/Ea) was
    # applied, corrupting the kernel and, in extreme cases, the heap.
    system = qmeq.BuilderManyBody(
        Ea=reference.qd.Ea, Na=[0, 1, 1, 2], Tba=reference.Tba,
        mulst=[0.0, 0.0], tlst=lead_temperatures, dband=5.0e5,
        kerntype=kerntype, itype=1,
    )
    system.nsingle = 2
    system.tleads_array = amplitude_array
    system.solve(qdq=False, rotateq=False)

    np.testing.assert_allclose(
        system.current, reference.current, rtol=1e-10, atol=1e-13,
    )
    np.testing.assert_allclose(
        system.energy_current, reference.energy_current, rtol=1e-10, atol=1e-13,
    )
