"""Convention bridge between the independent NEGF oracle and public QmeQ input.

The NEGF core remains independent of QmeQ. These tests are deliberately kept
in a separate module because their narrower job is to prove that
``model_from_qmeq`` translates QmeQ's public conventions without sharing any
Builder or approach implementation.
"""

import numpy as np
import pytest

import qmeq

from .noninteracting_negf_solver import (
    current_analytic,
    cumulants,
    lead_weights,
    model_from_qmeq,
)


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

    # The conversion is normalisation only, g=sqrt(2*pi)*t, and the broadening
    # vector is A=g: QmeQ's tleads entry and this solver's g are both the
    # coefficient of d^dag. This test restates the convention; what *fixes* it
    # is agreement with QmeQ itself, in the two tests below -- an assertion like
    # this one cannot tell a convention from its conjugate, which is how the
    # conjugated version survived here.
    expected_couplings = np.sqrt(2.0 * np.pi) * tleads.T
    np.testing.assert_allclose(model.couplings, expected_couplings)
    np.testing.assert_allclose(model.amplitude_matrix(), expected_couplings)
    np.testing.assert_allclose(
        model.width_matrix(),
        expected_couplings @ expected_couplings.conj().T,
    )


def test_qmeq_conversion_converges_to_the_golden_rule():
    """What actually pins ``g = sqrt(2 pi) t``, conjugation included.

    At zero interaction this solver is exact, so QmeQ's golden-rule current must
    approach it as ``O(Gamma^2)``. A conjugation error flips the phase of any
    loop a single lead closes across two dot modes, which changes the current at
    leading order -- so the wrong convention does not converge at all. The model
    below is the smallest one that can tell: two modes, complex hopping, and
    each lead reaching *both* modes with complex amplitudes.

    With each lead touching one mode, or with real amplitudes, both conventions
    agree and this test is blind. That is why it is written with cross
    couplings, and why the conjugation error in the adapter survived the
    restatement test above.
    """
    hopping = 0.4 * np.exp(0.7j)
    unit = {
        (0, 0): 0.2, (0, 1): 0.05 * np.exp(-0.4j),
        (1, 0): 0.06 * np.exp(1.1j), (1, 1): 0.25,
    }
    hsingle = {(0, 0): -1.0, (1, 1): -1.2, (0, 1): hopping}
    mulst, tlst = {0: 0.3, 1: -0.3}, {0: 0.8, 1: 0.8}

    errors = []
    for scale in (1.0, 0.5, 0.25, 0.125):
        tleads = {key: scale * value for key, value in unit.items()}
        builder = qmeq.Builder(
            nsingle=2, hsingle=hsingle, coulomb={}, nleads=2, tleads=tleads,
            mulst=mulst, tlst=tlst, dband=1.0e6, kerntype="pyPauli", itype=1,
        )
        builder.solve()
        golden = float(np.asarray(builder.current)[0])
        model = model_from_qmeq(
            nsingle=2, hsingle=hsingle, nleads=2, tleads=tleads,
            mulst=mulst, tlst=tlst,
        )
        exact = float(current_analytic(model, np.array([1.0, 0.0])))
        errors.append(abs(golden - exact) / abs(golden))

    assert errors[0] < 0.2, errors
    # Halving the coupling must quarter the relative error, near enough. The
    # conjugated convention sits at a flat ~0.19 instead.
    for previous, following in zip(errors, errors[1:]):
        assert following < 0.6 * previous, errors
    assert errors[-1] < 1e-3, errors


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


def test_negf_orientation_pins_the_flux_sign():
    """Grading against the time-reversed model costs exactly one order.

    Conjugating every amplitude leaves the spectrum and every golden-rule rate
    alone, so rate-level checks and gauge-invariance checks are both blind to
    it; a two-terminal non-interacting current is even in the flux and cannot
    see it either. Three leads make the current odd in the flux at
    ``O(Gamma**2)``.

    Hence an order, not a tolerance: RTD at second order is accurate to
    ``O(Gamma**3)`` and this solver is exact, so the relative error must fall as
    ``Gamma**2``; against the conjugated model it falls as ``Gamma``. This pins
    ``hsingle``'s orientation and ``A = g`` together, since only their
    combination is observable.
    """
    scales = (0.08, 0.04, 0.02, 0.01)

    def model(scale):
        amplitude = np.array(
            [[0.9, 0.35], [0.4, 1.0], [0.55, 0.7]], dtype=complex
        )
        amplitude[0, 1] *= np.exp(0.7j)
        amplitude[1, 0] *= np.exp(-0.4j)
        amplitude[2, 1] *= np.exp(1.3j)
        amplitude = np.sqrt(scale) * amplitude / np.sqrt(2.0 * np.pi)
        return dict(
            nsingle=2,
            hsingle={(0, 0): 0.35, (1, 1): -0.22, (0, 1): 0.18 + 0.11j},
            coulomb={},
            nleads=3,
            tleads={(lead, mode): amplitude[lead, mode]
                    for lead in range(3) for mode in range(2)},
            mulst={0: 0.6, 1: -0.6, 2: 0.15},
            tlst={0: 0.5, 1: 0.5, 2: 0.5},
        )

    errors = []
    for scale in scales:
        parameters = model(scale)
        builder = qmeq.Builder(**parameters, dband=1e6, kerntype='pyRTD',
                               itype=1)
        builder.solve()
        exact_model = model_from_qmeq(**parameters)
        exact = current_analytic(
            exact_model, lead_weights(exact_model, 0)
        )
        approximate = float(np.asarray(builder.current, dtype=float)[0])
        errors.append(abs(approximate - exact) / abs(exact))

    order = float(np.polyfit(np.log(scales), np.log(errors), 1)[0])
    assert 1.7 < order < 2.3, f"relative order {order:.2f} from {errors}"


@pytest.mark.parametrize("principal_part", ["omit", "digamma", "quad"])
def test_lindblad_converges_to_the_exact_noninteracting_current(principal_part):
    """The Lindblad current must approach the exact one as ``O(Gamma**2)``.

    This solver is exact at zero interaction, and the Lindblad kernel is
    complete at first order, so its relative error must fall as ``Gamma``. The
    Lamb shift enters the current above that order, which is why every
    ``principal_part`` has to show the same rate: the shift is a correction to a
    generator that is already first-order complete, not a repair of it.

    Nothing else in the suite grades Lindblad against an exact result. The
    analytic checks in ``test_lambshift.py`` pin the Lamb shift itself, and the
    reference bundles are characterization values.
    """
    amplitude = np.array([[0.9, 0.35], [0.4, 1.0]], dtype=complex)
    amplitude[0, 1] *= np.exp(0.7j)
    amplitude[1, 0] *= np.exp(-0.4j)
    hsingle = {(0, 0): 0.35, (1, 1): -0.22, (0, 1): 0.18 + 0.11j}
    mulst, tlst = {0: 0.6, 1: -0.6}, {0: 0.5, 1: 0.5}
    scales = (0.08, 0.04, 0.02, 0.01)

    errors = []
    for scale in scales:
        tleads = {
            (lead, mode): np.sqrt(scale) * amplitude[lead, mode] / np.sqrt(2.0 * np.pi)
            for lead in range(2) for mode in range(2)
        }
        builder = qmeq.Builder(
            nsingle=2, hsingle=hsingle, coulomb={}, nleads=2, tleads=tleads,
            mulst=mulst, tlst=tlst, dband=1.0e6, kerntype="pyLindblad",
            principal_part=principal_part,
        )
        builder.solve()
        model = model_from_qmeq(
            nsingle=2, hsingle=hsingle, nleads=2, tleads=tleads,
            mulst=mulst, tlst=tlst,
        )
        exact = current_analytic(model, lead_weights(model, 0))
        errors.append(abs(float(np.asarray(builder.current)[0]) - exact) / abs(exact))

    order = float(np.polyfit(np.log(scales), np.log(errors), 1)[0])
    assert 0.8 < order < 1.3, f"relative order {order:.2f} from {errors}"
