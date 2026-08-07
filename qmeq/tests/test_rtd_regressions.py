import numpy as np
import pytest
import warnings

import qmeq
from qmeq.approach.base.RTD import RTDBandwidthWarning


def _roundoff_phase_current(imaginary_part):
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
        kerntype="RTD",
        indexing="charge",
        itype=1,
    )
    system.solve()
    return float(system.current[0]+system.current[2])


def test_RTD_ignores_roundoff_scale_tunnel_phase():
    exactly_real = _roundoff_phase_current(0.0)
    roundoff_phase = _roundoff_phase_current(2e-18)

    assert np.isclose(roundoff_phase, exactly_real, rtol=1e-12, atol=1e-15)


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
