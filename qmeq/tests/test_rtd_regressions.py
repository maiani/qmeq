import numpy as np

import qmeq


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
