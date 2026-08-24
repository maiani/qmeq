"""Self-validation of the non-interacting reference solver.

Six self-consistency checks plus the convention pins. Until they all pass the
solver has no trust level and must not be used to grade any QmeQ result.

Nothing here imports ``qmeq.approach`` or ``qmeq.builder``: the reference solver's value
is that it shares no conventions with the code it will judge.
"""

import numpy as np
import pytest
from scipy.integrate import quad

from .noninteracting_negf_solver import (
    NoninteractingModel,
    cumulants,
    current_analytic,
    lead_weights,
    model_from_qmeq,
)


PAULI_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
PAULI_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
PAULI_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)


def single_resonant_level(gamma_left, gamma_right, level=0.0, bias=0.0,
                          temperature=1.0):
    """Spinless one-level dot between two channels, in reference solver conventions."""
    return NoninteractingModel(
        hamiltonian=np.array([[level]], dtype=complex),
        couplings=np.array([[np.sqrt(gamma_left), np.sqrt(gamma_right)]],
                           dtype=complex),
        mu=np.array([bias / 2.0, -bias / 2.0]),
        temperature=np.array([temperature, temperature]),
        lead_of_channel=np.array([0, 1]),
    )


def spinful_double_dot(epsilon1, epsilon2, tc, b1, b2, g_left, g_right,
                       mu_left, mu_right, temperature_left, temperature_right):
    """Test fixture for a spinful DQD in the ``(1up, 1dn, 2up, 2dn)`` basis."""
    identity = np.eye(2, dtype=complex)

    def zeeman(field):
        bx, by, bz = field
        return 0.5 * (bx * PAULI_X + by * PAULI_Y + bz * PAULI_Z)

    hamiltonian = np.zeros((4, 4), dtype=complex)
    hamiltonian[0:2, 0:2] = epsilon1 * identity + zeeman(b1)
    hamiltonian[2:4, 2:4] = epsilon2 * identity + zeeman(b2)
    hamiltonian[0:2, 2:4] = tc * identity
    hamiltonian[2:4, 0:2] = np.conj(tc) * identity

    tleads = np.zeros((4, 4), dtype=complex)
    for spin in (0, 1):
        tleads[spin, spin] = g_left[0]
        tleads[spin, 2 + spin] = g_left[1]
        tleads[2 + spin, spin] = g_right[0]
        tleads[2 + spin, 2 + spin] = g_right[1]

    # The DQD is expressed through the generic adapter: it is a regression
    # fixture, not a privileged NEGF model.
    model = model_from_qmeq(
        nsingle=4, hsingle=hamiltonian, coulomb={}, nleads=4,
        tleads=tleads,
        mulst=[mu_left, mu_left, mu_right, mu_right],
        tlst=[temperature_left, temperature_left,
              temperature_right, temperature_right],
    )
    return NoninteractingModel(
        hamiltonian=model.hamiltonian,
        couplings=model.couplings,
        mu=model.mu,
        temperature=model.temperature,
        lead_of_channel=np.array([0, 0, 1, 1]),
    ), np.array([0, 1, 0, 1])


def interference_double_dot(flux_left=1.0, flux_right=0.5, hopping_phase=0.0,
                            left_shift=0.0, right_shift=0.0):
    """A DQD at prescribed plaquette fluxes, with an explicit gauge choice.

    ``Phi_alpha = a_{alpha 1} + a_{12} - a_{alpha 2}``. Given the fluxes and a
    choice of ``a_12`` and of the two ``a_{alpha 2}``, the remaining phases are
    fixed, so every call with the same fluxes describes the same physics in a
    different gauge.
    """
    a12 = hopping_phase
    a_l2, a_r2 = left_shift, right_shift
    a_l1 = flux_left - a12 + a_l2
    a_r1 = flux_right - a12 + a_r2
    return spinful_double_dot(
        epsilon1=-1.0, epsilon2=0.7,
        tc=0.4 * np.exp(1j * a12),
        b1=[0.2, 0.0, 0.5], b2=[0.0, 0.0, -0.3],
        g_left=np.array([0.22 * np.exp(1j * a_l1), 0.13 * np.exp(1j * a_l2)]),
        g_right=np.array([0.17 * np.exp(1j * a_r1), 0.25 * np.exp(1j * a_r2)]),
        mu_left=3.0, mu_right=-3.0,
        temperature_left=1.0, temperature_right=1.0,
    )


def observables(model, spin_of_channel):
    """Charge and spin current and noise at the left lead."""
    charge = cumulants(model, lead_weights(model, 0), order=2)
    spin = cumulants(
        model, lead_weights(model, 0, "spin", spin_of_channel), order=2
    )
    return np.array([
        charge[1].real, charge[2].real, spin[1].real, spin[2].real
    ])


def test_qmeq_adapter_handles_an_arbitrary_noninteracting_multiterminal_dot():
    """The QmeQ-shaped adapter is generic; a spinful DQD is only one case.

    This covers three orbitals, three leads, complex hopping, sparse QmeQ
    inputs, and unequal temperatures.  The dense input below has deliberately
    different lower-triangle entries: QmeQ's default ``herm_hs=True`` ignores
    those and completes the upper triangle by conjugation.
    """
    sparse = model_from_qmeq(
        nsingle=3,
        hsingle={(0, 0): -0.4, (1, 1): 0.2, (2, 2): 0.8,
                 (0, 1): 0.3j, (1, 2): -0.2 + 0.1j},
        coulomb={(0, 1, 1, 0): 0.0},
        nleads=3,
        tleads={(0, 0): 0.15, (0, 1): 0.04j, (1, 1): 0.12,
                (1, 2): -0.09j, (2, 0): 0.06, (2, 2): 0.11},
        mulst={0: 0.7, 1: -0.5, 2: 0.1},
        tlst=[0.6, 0.8, 1.1],
    )
    dense_hsingle = np.array([
        [-0.4, 0.3j, 0.0],
        [123.0, 0.2, -0.2 + 0.1j],
        [456.0, 789.0, 0.8],
    ])
    dense = model_from_qmeq(
        3, dense_hsingle, {}, 3,
        np.array([[0.15, 0.04j, 0.0], [0.0, 0.12, -0.09j],
                  [0.06, 0.0, 0.11]]),
        [0.7, -0.5, 0.1], [0.6, 0.8, 1.1],
    )
    np.testing.assert_allclose(sparse.hamiltonian, dense.hamiltonian)
    np.testing.assert_allclose(sparse.couplings, dense.couplings)
    for lead in range(3):
        sparse_current = current_analytic(sparse, lead_weights(sparse, lead))
        dense_current = current_analytic(dense, lead_weights(dense, lead))
        assert sparse_current == pytest.approx(dense_current, rel=1e-12)
    currents = [current_analytic(sparse, lead_weights(sparse, lead))
                for lead in range(3)]
    assert sum(currents) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("coulomb", [
    {(0, 1, 1, 0): 0.2},
    [[0, 1, 1, 0, 0.2]],
    np.array([[0, 1, 1, 0, 0.2]]),
])
def test_qmeq_adapter_refuses_interactions(coulomb):
    with pytest.raises(ValueError, match="coulomb=0"):
        model_from_qmeq(
            nsingle=1, hsingle={(0, 0): 0.0}, coulomb=coulomb,
            nleads=1, tleads={(0, 0): 0.1}, mulst={0: 0.0}, tlst={0: 1.0},
        )


# ---------------------------------------------------------------- check 1

@pytest.mark.parametrize("flux", [0.0, 0.7, np.pi])
def test_scattering_matrix_is_unitary(flux):
    """Check 1: S(E) is unitary to machine precision across the window."""
    model, _ = interference_double_dot(flux_left=flux)
    low, high = model.energy_window()
    identity = np.eye(model.nchannels)
    worst = 0.0
    for energy in np.linspace(low, high, 501):
        scattering = model.scattering_matrix(energy)
        worst = max(
            worst,
            np.max(np.abs(scattering @ scattering.conj().T - identity)),
        )
    assert worst < 1e-13, worst


def test_unitarity_holds_for_a_rank_deficient_coupling():
    """A dark mode decoupled from one lead must not break unitarity.

    Rank-one lead couplings are the interesting case for the eventual kernel
    comparison, so the reference solver is checked there too rather than only on generic
    couplings.
    """
    model = NoninteractingModel(
        hamiltonian=np.array([[0.3, 0.2], [0.2, -0.4]], dtype=complex),
        # both channels couple to the same dot combination: one dark mode.
        couplings=np.array([[0.2, 0.15], [0.2, 0.15]], dtype=complex),
        mu=np.array([2.0, -2.0]), temperature=np.array([1.0, 1.0]),
        lead_of_channel=np.array([0, 1]),
    )
    identity = np.eye(2)
    for energy in np.linspace(-20.0, 20.0, 201):
        scattering = model.scattering_matrix(energy)
        assert np.max(
            np.abs(scattering @ scattering.conj().T - identity)
        ) < 1e-13


# ---------------------------------------------------------------- check 2

def test_zeroth_cumulant_vanishes_and_cumulants_are_real():
    """Check 2: F(0) = 0, and the cumulants carry no imaginary part."""
    model, _ = interference_double_dot()
    values = cumulants(model, lead_weights(model, 0), order=3)
    assert abs(values[0]) < 1e-12
    assert np.max(np.abs(values.imag)) < 1e-11, values


def test_contour_extraction_agrees_with_the_closed_form_current():
    """The Cauchy-extracted first cumulant matches an independent derivation.

    ``current_analytic`` implements ``tr[n(Q - S^dagger Q S)]`` directly, so
    this compares two different routes through the same scattering matrix and
    catches an error in the contour bookkeeping rather than in the physics.
    """
    model, _ = interference_double_dot(flux_left=0.9, flux_right=-0.4)
    weights = lead_weights(model, 0)
    contour = cumulants(model, weights, order=2)[1].real
    closed = current_analytic(model, weights)
    assert contour == pytest.approx(closed, rel=1e-10, abs=1e-15)


@pytest.mark.parametrize("radius", [0.2, 0.35, 0.6])
def test_cumulants_do_not_depend_on_the_contour_radius(radius):
    """No step size to tune: the answer is radius-independent by analyticity."""
    model = single_resonant_level(0.05, 0.05, bias=40.0)
    weights = np.array([1.0, 0.0])
    reference = cumulants(model, weights, order=2, radius=0.35)
    value = cumulants(model, weights, order=2, radius=radius)
    assert value[1].real == pytest.approx(reference[1].real, rel=1e-11)
    assert value[2].real == pytest.approx(reference[2].real, rel=1e-9)


# ---------------------------------------------------------------- check 3

@pytest.mark.parametrize("gamma, bias", [(0.1, 40.0), (0.25, 60.0)])
def test_breit_wigner_reduction_to_machine_precision(gamma, bias):
    """Check 3: exact agreement with the closed-form resonant-level result.

    ``qmeq/tests/test_counting.py`` already compares QmeQ's RTDnoise against
    this same integrand, at ``rtol=1e-4`` for the current and ``1e-2`` for the
    noise. The reference solver must reproduce it far more tightly than that, or it
    cannot be used to grade anything.
    """
    gamma_left = gamma_right = gamma / 2.0
    temperature = 1.0
    model = single_resonant_level(
        gamma_left, gamma_right, bias=bias, temperature=temperature
    )

    def integrands(energy):
        # expit(-x), stable in both tails.
        def occupation(mu):
            x = (energy - mu) / temperature
            return (
                np.exp(-x) / (1.0 + np.exp(-x)) if x > 0.0
                else 1.0 / (1.0 + np.exp(x))
            )

        f_left = occupation(bias / 2.0)
        f_right = occupation(-bias / 2.0)
        transmission = gamma_left * gamma_right / (
            energy**2 + (gamma / 2.0) ** 2
        )
        current = transmission * (f_left - f_right)
        noise = (
            transmission * (
                f_left * (1.0 - f_left) + f_right * (1.0 - f_right)
            )
            + transmission * (1.0 - transmission) * (f_left - f_right) ** 2
        )
        return current / (2.0 * np.pi), noise / (2.0 * np.pi)

    exact_current = quad(
        lambda e: integrands(e)[0], -np.inf, np.inf, epsabs=1e-14
    )[0]
    exact_noise = quad(
        lambda e: integrands(e)[1], -np.inf, np.inf, epsabs=1e-14
    )[0]

    values = cumulants(model, np.array([1.0, 0.0]), order=2)
    assert values[1].real == pytest.approx(exact_current, rel=1e-11)
    assert values[2].real == pytest.approx(exact_noise, rel=1e-10)


# ---------------------------------------------------------------- check 4

def test_johnson_nyquist_at_zero_bias():
    """Check 4: ``S = 2 T G`` in this convention, not ``4 T G``.

    ``G`` comes from a central difference of the *current*, which is a separate
    code path from the noise, so this ties the second cumulant to the first
    through a thermodynamic identity rather than through the same formula. The
    coefficient is 2 and not 4 because ``S`` is defined as ``d Var/dt`` with no
    extra factor of two; see the reference solver's module docstring.
    """
    temperature, gamma = 1.0, 0.1
    equilibrium = single_resonant_level(
        gamma / 2, gamma / 2, bias=0.0, temperature=temperature
    )
    noise = cumulants(equilibrium, np.array([1.0, 0.0]), order=2)[2].real

    step = 1e-4
    weights = np.array([1.0, 0.0])
    forward = current_analytic(
        single_resonant_level(gamma / 2, gamma / 2, bias=+step,
                              temperature=temperature), weights
    )
    backward = current_analytic(
        single_resonant_level(gamma / 2, gamma / 2, bias=-step,
                              temperature=temperature), weights
    )
    conductance = (forward - backward) / (2.0 * step)
    assert noise == pytest.approx(2.0 * temperature * conductance, rel=1e-7)


# ---------------------------------------------------------------- check 5

def test_poisson_limit_gives_unit_fano_factor():
    """Check 5: at large bias and small transmission the Fano factor is 1."""
    model = single_resonant_level(1e-5, 0.1, bias=200.0, temperature=1.0)
    values = cumulants(model, np.array([1.0, 0.0]), order=2)
    fano = values[2].real / abs(values[1].real)
    assert fano == pytest.approx(1.0, abs=1e-3)


def test_symmetric_barrier_gives_one_half_fano_factor():
    """The classic double-barrier suppression, as a second normalisation pin.

    At large bias a symmetric single level carries ``I = Gamma_L Gamma_R /
    (Gamma_L + Gamma_R)`` and Fano ``(Gamma_L^2 + Gamma_R^2) /
    (Gamma_L + Gamma_R)^2``, which is ``1/2`` when symmetric. The current value
    pins the reference solver's amplitude normalisation analytically: it fixes both
    ``Gamma_c = |g_c|^2`` and the absorbed ``2 pi``, with no QmeQ import.
    """
    gamma_left, gamma_right = 0.03, 0.03
    model = single_resonant_level(
        gamma_left, gamma_right, level=0.0, bias=400.0, temperature=1.0
    )
    values = cumulants(model, np.array([1.0, 0.0]), order=2)
    expected_current = gamma_left * gamma_right / (gamma_left + gamma_right)
    expected_fano = (gamma_left**2 + gamma_right**2) / (
        gamma_left + gamma_right
    ) ** 2
    assert values[1].real == pytest.approx(expected_current, rel=1e-4)
    assert values[2].real / values[1].real == pytest.approx(
        expected_fano, rel=1e-3
    )


# ---------------------------------------------------------------- check 6

@pytest.mark.parametrize(
    "hopping_phase, left_shift, right_shift",
    [
        (0.3, 0.0, 0.0),      # rephase dot state 1
        (0.5, 0.5, 0.5),      # rephase dot state 2
        (0.0, 0.4, 0.4),      # global lead phase
        (-0.8, 0.25, -0.6),   # both dot states, independently
    ],
)
def test_gauge_invariance_of_observables(hopping_phase, left_shift,
                                        right_shift):
    """Check 6a: only the plaquette fluxes may enter an observable.

    See ``NoninteractingModel.amplitude_matrix`` for the conjugation this pins.
    """
    reference, spin = interference_double_dot()
    rotated, _ = interference_double_dot(
        hopping_phase=hopping_phase,
        left_shift=left_shift,
        right_shift=right_shift,
    )
    np.testing.assert_allclose(
        observables(rotated, spin), observables(reference, spin),
        rtol=1e-10, atol=1e-15,
    )


@pytest.mark.parametrize("lead", ["left", "right"])
def test_observables_are_periodic_in_each_flux(lead):
    """Check 6b: a shift of one flux by 2 pi changes nothing."""
    reference, spin = interference_double_dot()
    if lead == "left":
        shifted, _ = interference_double_dot(flux_left=1.0 + 2.0 * np.pi)
    else:
        shifted, _ = interference_double_dot(flux_right=0.5 + 2.0 * np.pi)
    np.testing.assert_allclose(
        observables(shifted, spin), observables(reference, spin),
        rtol=1e-10, atol=1e-15,
    )


def test_flux_actually_changes_the_observables():
    """Negative control for check 6.

    Gauge invariance and periodicity would both pass on a reference solver that ignored
    every phase. This asserts the flux is not being dropped.
    """
    reference, spin = interference_double_dot()
    shifted, _ = interference_double_dot(flux_left=1.0 + np.pi)
    difference = np.abs(
        observables(shifted, spin) - observables(reference, spin)
    )
    assert np.max(difference / np.abs(observables(reference, spin))) > 0.1


# ---------------------------------------------- convention pins (exit gate)

def test_current_is_positive_into_the_dot_from_the_higher_potential():
    """Sign convention: currents are positive inward."""
    model = single_resonant_level(0.05, 0.05, bias=40.0)
    assert current_analytic(model, np.array([1.0, 0.0])) > 0.0
    assert current_analytic(model, np.array([0.0, 1.0])) < 0.0


def test_lead_currents_sum_to_zero_in_the_stationary_state():
    """Charge conservation, which no single-lead formula can fake."""
    model, _ = interference_double_dot()
    left = cumulants(model, lead_weights(model, 0), order=1)[1].real
    right = cumulants(model, lead_weights(model, 1), order=1)[1].real
    assert left + right == pytest.approx(0.0, abs=1e-12 * abs(left))


def test_two_terminal_noise_is_symmetric_between_leads():
    """``S_LL = S_RR`` for two terminals, from charge conservation."""
    model, _ = interference_double_dot()
    left = cumulants(model, lead_weights(model, 0), order=2)[2].real
    right = cumulants(model, lead_weights(model, 1), order=2)[2].real
    assert left == pytest.approx(right, rel=1e-10)


def test_spin_weighting_reduces_to_the_channel_difference():
    """Spin convention: ``I^Sz = (I_up - I_down) / 2``, evaluated per channel."""
    model, spin = interference_double_dot()
    up = np.array([
        1.0 if (model.lead_of_channel[c] == 0 and spin[c] == 0) else 0.0
        for c in range(model.nchannels)
    ])
    down = np.array([
        1.0 if (model.lead_of_channel[c] == 0 and spin[c] == 1) else 0.0
        for c in range(model.nchannels)
    ])
    spin_current = cumulants(
        model, lead_weights(model, 0, "spin", spin), order=1
    )[1].real
    expected = 0.5 * (
        cumulants(model, up, order=1)[1].real
        - cumulants(model, down, order=1)[1].real
    )
    assert spin_current == pytest.approx(expected, rel=1e-10, abs=1e-16)


def test_equilibrium_carries_no_current_at_unequal_couplings():
    """Zero bias and zero thermal bias give zero current in every channel."""
    model, spin = spinful_double_dot(
        epsilon1=-0.5, epsilon2=0.4, tc=0.3 * np.exp(0.7j),
        b1=[0.1, 0.0, 0.2], b2=[0.0, 0.0, -0.1],
        g_left=np.array([0.3, 0.1 * np.exp(0.9j)]),
        g_right=np.array([0.05, 0.4]),
        mu_left=0.0, mu_right=0.0,
        temperature_left=1.0, temperature_right=1.0,
    )
    for lead in (0, 1):
        value = cumulants(model, lead_weights(model, lead), order=1)[1].real
        assert abs(value) < 1e-12


def test_energy_window_is_wide_enough():
    """The default window is a choice; widening it must not move the answer."""
    model = single_resonant_level(0.05, 0.05, bias=40.0)
    weights = np.array([1.0, 0.0])
    default = cumulants(model, weights, order=2)
    low, high = model.energy_window()
    widened = cumulants(
        model, weights, order=2, window=(2.0 * low, 2.0 * high)
    )
    assert widened[1].real == pytest.approx(default[1].real, rel=1e-10)
    assert widened[2].real == pytest.approx(default[2].real, rel=1e-9)


def _thermal_bias_level(level):
    return NoninteractingModel(
        hamiltonian=np.array([[level]], dtype=complex),
        couplings=np.array([[np.sqrt(0.05), np.sqrt(0.05)]], dtype=complex),
        mu=np.array([0.0, 0.0]), temperature=np.array([2.0, 1.0]),
        lead_of_channel=np.array([0, 1]),
    )


def test_thermocurrent_vanishes_at_the_particle_hole_symmetric_point():
    """A thermal bias alone drives no current when ``eps = mu``.

    At the symmetric point the electron- and hole-like contributions cancel
    exactly, so this is a symmetry the reference solver must respect rather than a
    tolerance it must meet. Measured at 1.8e-18 against a current scale of
    1e-3 for the off-symmetric case below.
    """
    values = cumulants(_thermal_bias_level(0.0), np.array([1.0, 0.0]), order=2)
    assert abs(values[1].real) < 1e-15
    assert values[2].real > 0.0


def test_unequal_lead_temperatures_are_supported():
    """The reference solver has no equal-temperature restriction, unlike RTD's integrals.

    Recorded because it is what lets the reference solver answer the ``dband``
    convergence question the RTD unequal-temperature warning currently raises.
    Away from the symmetric point a pure thermal bias drives a finite current
    at zero voltage bias, and it must reverse when the level crosses ``mu``.
    """
    above = cumulants(_thermal_bias_level(3.0), np.array([1.0, 0.0]), order=2)
    below = cumulants(_thermal_bias_level(-3.0), np.array([1.0, 0.0]), order=2)
    assert abs(above[1].real) > 1e-6
    assert abs(below[1].real) > 1e-6
    assert np.sign(above[1].real) == -np.sign(below[1].real)
    assert above[2].real > 0.0 and below[2].real > 0.0
