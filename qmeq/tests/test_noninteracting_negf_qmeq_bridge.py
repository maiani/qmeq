"""Convention bridge between the independent NEGF oracle and public QmeQ input.

The NEGF core remains independent of QmeQ. These tests are deliberately kept
in a separate module because their narrower job is to prove that
``model_from_qmeq`` translates QmeQ's public conventions without sharing any
Builder or approach implementation.
"""

import numpy as np

import qmeq

from .noninteracting_negf_solver import cumulants, model_from_qmeq


def test_adapter_converts_qmeq_tunnelling_amplitudes_and_phases():
    tleads = np.array([
        [0.11 + 0.07j, -0.03j],
        [0.04 - 0.02j, 0.09],
    ])
    model = model_from_qmeq(
        nsingle=2,
        hsingle=np.diag([-0.3, 0.4]),
        nleads=2,
        tleads=tleads,
        mulst=[0.5, -0.5],
        tlst=[1.0, 1.0],
    )

    # QmeQ's Tba electron-adding matrix element is t. The NEGF Hamiltonian
    # uses g gamma^dagger d + h.c., hence g=sqrt(2*pi)*conj(t), while the
    # broadening vector A=conj(g)=sqrt(2*pi)*t.
    expected_amplitudes = np.sqrt(2.0 * np.pi) * tleads.T
    np.testing.assert_allclose(model.couplings, expected_amplitudes.conj())
    np.testing.assert_allclose(model.amplitude_matrix(), expected_amplitudes)
    np.testing.assert_allclose(
        model.width_matrix(),
        2.0 * np.pi * tleads.T @ tleads.T.conj().T,
    )


def test_adapter_negf_matches_qmeq_rtdnoise_for_a_resonant_level():
    gamma_left, gamma_right = 0.05, 0.05
    tleads = {
        (0, 0): np.sqrt(gamma_left / (2.0 * np.pi)),
        (1, 0): np.sqrt(gamma_right / (2.0 * np.pi)),
    }
    shared = dict(
        nsingle=1,
        hsingle={(0, 0): 0.0},
        nleads=2,
        tleads=tleads,
        mulst={0: 20.0, 1: -20.0},
        tlst={0: 1.0, 1: 1.0},
    )

    reference = model_from_qmeq(**shared)
    exact = cumulants(reference, np.array([1.0, 0.0]), order=2)

    system = qmeq.Builder(
        **shared,
        dband={0: 1e4, 1: 1e4},
        kerntype="pyRTDnoise",
        countingleads=(0,),
        off_diag_corrections=False,
    )
    system.solve()

    np.testing.assert_allclose(
        system.current_noise[0].real, exact[1].real, rtol=1e-4, atol=0.0
    )
    np.testing.assert_allclose(
        system.current_noise[1].real, exact[2].real, rtol=1e-2, atol=0.0
    )
