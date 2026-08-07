"""Regression and independent checks for zero-frequency counting statistics."""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import expit

import qmeq
from qmeq.approach.counting import markovian_current_noise
from qmeq.approach.counting import markovian_current_noise_matrix
from qmeq.approach.counting import stationary_projected_pseudoinverse
from qmeq.tests.data_counting import (
    FIRST_ORDER_POINTS,
    FIRST_ORDER_REFERENCE,
    RTD_POINTS,
    RTD_REFERENCE,
)


def _first_order_system(kerntype, gate, bias, countingleads=(0,)):
    return qmeq.Builder(
        nsingle=2,
        hsingle={(0, 0): -10 + gate, (1, 1): -12 + gate, (0, 1): 20},
        coulomb={(0, 1, 1, 0): 30},
        nleads=2,
        tleads={(0, 0): 2.0, (1, 1): 1.0, (0, 1): 0.6, (1, 0): 0.1},
        mulst={0: bias / 2, 1: -bias / 2},
        tlst={0: 25.0, 1: 25.0},
        dband={0: 1000.0, 1: 1000.0},
        kerntype=kerntype,
        itype=2,
        countingleads=countingleads,
    )


def _rtd_system(gate, bias, countingleads=(0,)):
    return qmeq.Builder(
        nsingle=1,
        hsingle={(0, 0): gate},
        nleads=2,
        tleads={
            (0, 0): np.sqrt(0.08 / (2 * np.pi)),
            (1, 0): np.sqrt(0.12 / (2 * np.pi)),
        },
        mulst={0: bias / 2, 1: -bias / 2},
        tlst={0: 1.0, 1: 1.0},
        dband={0: 100.0, 1: 100.0},
        kerntype="pyRTDnoise",
        countingleads=countingleads,
        off_diag_corrections=False,
    )


@pytest.mark.parametrize("kerntype", FIRST_ORDER_REFERENCE)
@pytest.mark.parametrize("point_index", range(len(FIRST_ORDER_POINTS)))
def test_first_order_matches_simon_reference(kerntype, point_index):
    gate, bias = FIRST_ORDER_POINTS[point_index]
    system = _first_order_system(kerntype, gate, bias)
    system.solve()

    np.testing.assert_allclose(
        system.current_noise,
        FIRST_ORDER_REFERENCE[kerntype][point_index],
        rtol=1e-11,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        system.current_noise[0], system.current[0], rtol=1e-11, atol=1e-12
    )


@pytest.mark.parametrize("kerntype", FIRST_ORDER_REFERENCE)
def test_first_order_python_and_selected_backend_agree(kerntype):
    python_system = _first_order_system(f"py{kerntype}", 0.0, 5.0)
    selected_system = _first_order_system(kerntype, 0.0, 5.0)
    python_system.solve()
    selected_system.solve()

    for python_value, selected_value in (
        (python_system.appr._counting_kernel,
         selected_system.appr._counting_kernel),
        (python_system.appr.Lpm, selected_system.appr.Lpm),
        (python_system.phi0, selected_system.phi0),
        (python_system.current_noise, selected_system.current_noise),
        (python_system.current_noise_matrix,
         selected_system.current_noise_matrix),
    ):
        np.testing.assert_allclose(
            selected_value, python_value, rtol=1e-12, atol=1e-13
        )


@pytest.mark.parametrize("point_index", range(len(RTD_POINTS)))
def test_rtd_matches_simon_reference(point_index):
    gate, bias = RTD_POINTS[point_index]
    expected = RTD_REFERENCE[point_index]
    system = _rtd_system(gate, bias)
    system.solve()

    np.testing.assert_allclose(
        system.current_noise, expected["full"], rtol=1e-8, atol=1e-10
    )
    np.testing.assert_allclose(
        system.current_noise_first,
        expected["first"],
        rtol=1e-8,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        system.current_noise_o4trunc,
        expected["o4trunc"],
        rtol=1e-8,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        system.current_noise[0].real,
        system.current[0],
        rtol=1e-8,
        atol=1e-10,
    )


def test_rtdnoise_alias_is_the_same_python_implementation():
    python_name = _rtd_system(0.0, 6.0)
    alias = _rtd_system(0.0, 6.0)
    alias.kerntype = "RTDnoise"
    assert alias.appr.__class__ is python_name.appr.__class__


def _single_level(countingleads=(0,), bias=4.0, gamma_left=0.1,
                  gamma_right=0.2):
    return qmeq.Builder(
        nsingle=1,
        hsingle={(0, 0): 0.3},
        nleads=2,
        tleads={
            (0, 0): np.sqrt(gamma_left / (2 * np.pi)),
            (1, 0): np.sqrt(gamma_right / (2 * np.pi)),
        },
        mulst={0: bias / 2, 1: -bias / 2},
        tlst={0: 1.0, 1: 1.0},
        dband={0: 100.0, 1: 100.0},
        kerntype="Pauli",
        itype=2,
        countingleads=countingleads,
    )


def test_counting_is_opt_in_and_reusable():
    disabled = _single_level(countingleads=None)
    disabled.solve()
    assert disabled.current_noise is None
    assert disabled.current_noise_matrix is None
    assert disabled.appr.Lpm is None
    assert disabled.appr._counting_kernel is None

    system = _single_level(countingleads=(0,))
    system.solve()
    left = system.current_noise.copy()
    np.testing.assert_allclose(system.current, disabled.current, rtol=0, atol=0)
    system.solve()
    np.testing.assert_allclose(system.current_noise, left, rtol=0, atol=1e-14)

    system.countingleads = (1,)
    system.solve()
    assert system.current_noise_matrix.shape == (1, 1)
    np.testing.assert_allclose(system.current_noise[0], system.current[1])
    np.testing.assert_allclose(system.current_noise[1], left[1])
    assert np.sign(system.current_noise[0]) == -np.sign(left[0])

    system.change(countingleads=None)
    system.solve()
    assert system.current_noise is None
    assert system.current_noise_matrix is None
    assert system.appr.Lpm is None
    assert system.appr._counting_kernel is None


def test_nonsquare_kernel_uses_the_physical_square_block():
    square = _single_level()
    nonsquare = _single_level()
    nonsquare.symq = False
    square.solve()
    nonsquare.solve()
    np.testing.assert_allclose(
        nonsquare.current_noise, square.current_noise,
        rtol=1e-12, atol=1e-13,
    )


def test_equilibrium_and_all_leads_counting():
    equilibrium = _single_level(bias=0.0)
    equilibrium.solve()
    assert abs(equilibrium.current_noise[0]) < 1e-13
    assert equilibrium.current_noise[1] > 0

    all_leads = _single_level(countingleads=(0, 1))
    all_leads.solve()
    np.testing.assert_allclose(all_leads.current_noise, 0.0, atol=1e-12)
    np.testing.assert_allclose(
        all_leads.current_noise_matrix,
        all_leads.current_noise_matrix.T,
        rtol=0,
        atol=0,
    )
    assert np.max(np.abs(all_leads.current_noise_matrix)) > 1e-3
    np.testing.assert_allclose(
        np.sum(all_leads.current_noise_matrix), 0.0, atol=1e-12
    )


@pytest.mark.parametrize("kerntype", FIRST_ORDER_REFERENCE)
def test_first_order_noise_matrix_reconstructs_counting_fields(kerntype):
    individual = []
    for lead in (0, 1):
        system = _first_order_system(
            kerntype, gate=0.0, bias=5.0, countingleads=(lead,)
        )
        system.solve()
        individual.append(system.current_noise.copy())

    joint = _first_order_system(
        kerntype, gate=0.0, bias=5.0, countingleads=(0, 1)
    )
    joint.solve()
    matrix = joint.current_noise_matrix

    np.testing.assert_allclose(matrix, matrix.T, rtol=0, atol=0)
    np.testing.assert_allclose(
        np.diag(matrix), [value[1] for value in individual],
        rtol=1e-11, atol=1e-12,
    )
    np.testing.assert_allclose(
        joint.current_noise,
        [joint.current[0] + joint.current[1], np.sum(matrix)],
        rtol=1e-11, atol=1e-12,
    )

    spin_weights = np.array([0.5, -0.5])
    direct_spin_noise = spin_weights @ matrix @ spin_weights
    reconstructed_spin_noise = (
        2 * individual[0][1] + 2 * individual[1][1]
        - joint.current_noise[1]
    ) / 4
    np.testing.assert_allclose(
        direct_spin_noise, reconstructed_spin_noise,
        rtol=1e-11, atol=1e-12,
    )


def test_rtd_noise_matrices_reconstruct_counting_fields():
    individual = []
    for lead in (0, 1):
        system = _rtd_system(0.0, 6.0, countingleads=(lead,))
        system.solve()
        individual.append((
            system.current_noise.copy(),
            system.current_noise_first.copy(),
        ))

    joint = _rtd_system(0.0, 6.0, countingleads=(0, 1))
    joint.solve()

    for scalar, matrix, scalar_index in (
        (joint.current_noise, joint.current_noise_matrix, 0),
        (joint.current_noise_first, joint.current_noise_matrix_first, 1),
    ):
        np.testing.assert_allclose(matrix, matrix.T, rtol=0, atol=0)
        np.testing.assert_allclose(
            np.diag(matrix),
            [value[scalar_index][1] for value in individual],
            rtol=1e-9, atol=1e-11,
        )
        np.testing.assert_allclose(
            scalar[1], np.sum(matrix), rtol=1e-9, atol=1e-11
        )


def test_rtd_matrix_sum_matches_aggregate_nonmarkovian_formula():
    system = qmeq.Builder(
        nsingle=1,
        hsingle={(0, 0): 0.2},
        nleads=3,
        tleads={
            (0, 0): np.sqrt(0.08 / (2 * np.pi)),
            (1, 0): np.sqrt(0.12 / (2 * np.pi)),
            (2, 0): np.sqrt(0.05 / (2 * np.pi)),
        },
        mulst={0: 3.0, 1: -3.0, 2: 1.0},
        tlst={0: 1.0, 1: 1.0, 2: 1.0},
        dband={0: 100.0, 1: 100.0, 2: 100.0},
        kerntype="pyRTDnoise",
        countingleads=(0, 2),
        off_diag_corrections=False,
    )
    system.solve()
    approach = system.appr
    counted = system.countingleads
    L0, Lp1, Lp2, Lm2, Lm1 = approach.build_counting_kernels(
        approach.Lpm_first, approach.Lpm_second, counted
    )
    L0p, Lp1p, Lp2p, Lm2p, Lm1p = approach.build_counting_kernels(
        approach.Lpm_first_dot, approach.Lpm_second_dot, counted
    )
    P, O, _, R = stationary_projected_pseudoinverse(
        approach.kern, approach.phi0, approach.norm_vec
    )
    Jp = 1j * (Lp1 - Lm1 + 2 * Lp2 - 2 * Lm2)
    Jpp = -Lp1 - Lm1 - 4 * Lp2 - 4 * Lm2
    Jdot = L0p + Lp1p + Lp2p + Lm2p + Lm1p
    Jdotp = 1j * (Lp1p - Lm1p + 2 * Lp2p - 2 * Lm2p)
    current = -1j * (O @ Jp @ P)
    noise = (
        -(O @ (Jpp - 2 * Jp @ R @ Jp) @ P)
        + 2 * current * (O @ (Jdotp - Jp @ R @ Jdot) @ P)
    )

    np.testing.assert_allclose(
        system.current_noise,
        [current.item(), noise.item()],
        rtol=1e-10,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        system.current_noise[1],
        np.sum(system.current_noise_matrix),
        rtol=1e-10,
        atol=1e-12,
    )


def test_noise_matrix_follows_countinglead_order():
    forward = _single_level(countingleads=(0, 1))
    reverse = _single_level(countingleads=(1, 0))
    forward.solve()
    reverse.solve()

    np.testing.assert_allclose(
        reverse.current_noise_matrix,
        forward.current_noise_matrix[::-1, ::-1],
        rtol=0,
        atol=1e-14,
    )


@pytest.mark.parametrize(
    "countingleads, exception",
    [
        ([], ValueError),
        ([0, 0], ValueError),
        ([-1], ValueError),
        ([2], ValueError),
        ([0.0], TypeError),
        ([True], TypeError),
        ("0", TypeError),
    ],
)
def test_counting_lead_validation(countingleads, exception):
    with pytest.raises(exception):
        qmeq.Builder(nleads=2, countingleads=countingleads)


def test_nonunique_stationary_state_is_rejected():
    system = qmeq.Builder(
        nsingle=1,
        nleads=1,
        tleads={},
        mulst={0: 0.0},
        tlst={0: 1.0},
        dband={0: 10.0},
        kerntype="Pauli",
        countingleads=(0,),
    )
    with pytest.raises(np.linalg.LinAlgError, match="unique stationary state"):
        system.solve()


def test_weak_but_nonzero_coupling_remains_well_defined():
    system = _single_level(gamma_left=1e-20, gamma_right=2e-20)
    system.solve()
    assert np.all(np.isfinite(system.current_noise))
    np.testing.assert_allclose(
        system.current_noise[0], system.current[0], rtol=1e-11, atol=1e-32
    )


def test_markovian_formula_matches_tilted_liouvillian_derivatives():
    rate_in, rate_out = 0.7, 1.3
    kernel = np.array([[-rate_in, rate_out], [rate_in, -rate_out]])
    stationary = np.array([rate_out, rate_in]) / (rate_in + rate_out)
    lminus = np.array([[0.0, 0.5], [0.0, 0.0]])
    lplus = np.array([[0.0, 0.0], [0.4, 0.0]])
    result = markovian_current_noise(
        kernel, stationary, np.ones(2), np.array([lminus, lplus])
    )

    def eigenvalue(chi):
        tilted = (
            kernel
            + (np.exp(1j * chi) - 1) * lplus
            + (np.exp(-1j * chi) - 1) * lminus
        )
        values = np.linalg.eigvals(tilted)
        return values[np.argmin(np.abs(values))]

    step = 3e-4
    lambda_minus, lambda_zero, lambda_plus = (
        eigenvalue(-step), eigenvalue(0.0), eigenvalue(step)
    )
    numerical = np.array([
        ((lambda_plus - lambda_minus) / (2j * step)).real,
        (-(lambda_plus - 2 * lambda_zero + lambda_minus) / step**2).real,
    ])
    np.testing.assert_allclose(result, numerical, rtol=2e-7, atol=2e-9)


def test_markovian_noise_matrix_matches_mixed_tilted_derivatives():
    lead_lpm = np.array([
        [
            [[0.0, 0.5], [0.0, 0.0]],
            [[0.0, 0.0], [0.4, 0.0]],
        ],
        [
            [[0.0, 0.8], [0.0, 0.0]],
            [[0.0, 0.0], [0.3, 0.0]],
        ],
    ])
    rate_in, rate_out = 0.7, 1.3
    kernel = np.array([[-rate_in, rate_out], [rate_in, -rate_out]])
    stationary = np.array([rate_out, rate_in]) / (rate_in + rate_out)
    currents, matrix = markovian_current_noise_matrix(
        kernel, stationary, np.ones(2), lead_lpm
    )

    def eigenvalue(fields):
        tilted = kernel.astype(complex)
        for field, (lminus, lplus) in zip(fields, lead_lpm):
            tilted += (np.exp(1j * field) - 1) * lplus
            tilted += (np.exp(-1j * field) - 1) * lminus
        values = np.linalg.eigvals(tilted)
        return values[np.argmin(np.abs(values))]

    step = 3e-4
    numerical_currents = np.empty(2)
    numerical_matrix = np.empty((2, 2))
    zero = np.zeros(2)
    for i in range(2):
        shift = np.zeros(2)
        shift[i] = step
        numerical_currents[i] = (
            (eigenvalue(shift) - eigenvalue(-shift)) / (2j * step)
        ).real
        numerical_matrix[i, i] = (
            -(eigenvalue(shift) - 2 * eigenvalue(zero)
              + eigenvalue(-shift)) / step**2
        ).real
        for j in range(i + 1, 2):
            shift_j = np.zeros(2)
            shift_j[j] = step
            mixed = -(
                eigenvalue(shift + shift_j)
                - eigenvalue(shift - shift_j)
                - eigenvalue(-shift + shift_j)
                + eigenvalue(-shift - shift_j)
            ) / (4 * step**2)
            numerical_matrix[i, j] = mixed.real
            numerical_matrix[j, i] = mixed.real

    np.testing.assert_allclose(
        currents, numerical_currents, rtol=2e-7, atol=2e-9
    )
    np.testing.assert_allclose(
        matrix, numerical_matrix, rtol=2e-7, atol=2e-9
    )


def test_unsupported_counting_modes_raise():
    with pytest.raises(NotImplementedError, match="Matrix-free"):
        system = _single_level()
        system.mfreeq = True
        system.solve()

    with pytest.raises(NotImplementedError, match="2vN"):
        qmeq.Builder(
            nleads=1, dband={0: 10.0}, kpnt=5,
            kerntype="2vN", countingleads=(0,),
        ).solve(niter=1)

    with pytest.raises(NotImplementedError, match="electron-phonon"):
        qmeq.BuilderElPh(nleads=1, countingleads=(0,)).solve()

    with pytest.raises(NotImplementedError, match="pyRTD"):
        qmeq.Builder(
            nleads=1, kerntype="pyRTD", countingleads=(0,)
        ).solve()

    with pytest.raises(NotImplementedError, match="off-diagonal"):
        qmeq.Builder(
            nleads=1, kerntype="pyRTDnoise", countingleads=(0,)
        ).solve()


def test_rtd_sequential_limit_agrees_with_pauli():
    gamma = 1e-4
    params = dict(
        nsingle=1,
        hsingle={(0, 0): 0.4},
        nleads=2,
        tleads={
            (0, 0): np.sqrt(0.4 * gamma / (2 * np.pi)),
            (1, 0): np.sqrt(0.6 * gamma / (2 * np.pi)),
        },
        mulst={0: 3.0, 1: -3.0},
        tlst={0: 1.0, 1: 1.0},
        dband={0: 100.0, 1: 100.0},
        countingleads=(0,),
    )
    pauli = qmeq.Builder(**params, kerntype="Pauli", itype=1)
    rtd = qmeq.Builder(
        **params, kerntype="pyRTDnoise", off_diag_corrections=False
    )
    pauli.solve()
    rtd.solve()
    np.testing.assert_allclose(
        rtd.current_noise_first.real, pauli.current_noise,
        rtol=3e-5, atol=1e-12,
    )


def _thermal_rtdnoise_system(dband):
    temperature = 100.0
    gamma_left, gamma_right = 1.5, 0.5
    return qmeq.Builder(
        nsingle=2,
        hsingle={(0, 0): -3500.0, (1, 1): -3500.0},
        coulomb={(0, 1, 1, 0): 2000.0},
        nleads=4,
        tleads={
            (0, 0): np.sqrt(gamma_left / (2*np.pi)),
            (1, 0): np.sqrt(gamma_right / (2*np.pi)),
            (2, 1): np.sqrt(gamma_left / (2*np.pi)),
            (3, 1): np.sqrt(gamma_right / (2*np.pi)),
        },
        mulst={lead: 0.0 for lead in range(4)},
        tlst={0: temperature, 1: 2*temperature,
              2: temperature, 3: 2*temperature},
        dband=dband,
        countingleads=(1, 3),
        kerntype="pyRTDnoise",
        off_diag_corrections=False,
    )


def test_rtdnoise_unequal_temperature_cutoff_convergence():
    lower_cutoff = _thermal_rtdnoise_system(2e6)
    higher_cutoff = _thermal_rtdnoise_system(2e7)
    lower_cutoff.solve()
    higher_cutoff.solve()
    np.testing.assert_allclose(
        lower_cutoff.current_noise[0], higher_cutoff.current_noise[0],
        rtol=1e-5, atol=1e-12,
    )
    np.testing.assert_allclose(
        lower_cutoff.current_noise[1], higher_cutoff.current_noise[1],
        rtol=5e-3, atol=1e-12,
    )


@pytest.mark.parametrize("gamma, bias", [(0.1, 40.0), (0.25, 60.0)])
def test_single_resonant_level_matches_exact_scattering(gamma, bias):
    gamma_left = gamma_right = gamma / 2
    system = qmeq.Builder(
        nsingle=1,
        hsingle={(0, 0): 0.0},
        nleads=2,
        tleads={
            (0, 0): np.sqrt(gamma_left / (2 * np.pi)),
            (1, 0): np.sqrt(gamma_right / (2 * np.pi)),
        },
        mulst={0: bias / 2, 1: -bias / 2},
        tlst={0: 1.0, 1: 1.0},
        dband={0: 1e4, 1: 1e4},
        kerntype="pyRTDnoise",
        countingleads=(0,),
        off_diag_corrections=False,
    )
    system.solve()

    def integrands(energy):
        f_left = expit(-(energy - bias / 2))
        f_right = expit(-(energy + bias / 2))
        transmission = gamma_left * gamma_right / (
            energy**2 + (gamma / 2)**2
        )
        current = transmission * (f_left - f_right) / (2 * np.pi)
        noise = (
            transmission
            * (f_left * (1 - f_left) + f_right * (1 - f_right))
            + transmission * (1 - transmission) * (f_left - f_right)**2
        ) / (2 * np.pi)
        return current, noise

    exact_current = quad(
        lambda energy: integrands(energy)[0], -np.inf, np.inf,
        epsabs=1e-12,
    )[0]
    exact_noise = quad(
        lambda energy: integrands(energy)[1], -np.inf, np.inf,
        epsabs=1e-12,
    )[0]
    np.testing.assert_allclose(
        system.current_noise[0].real, exact_current, rtol=1e-4, atol=0
    )
    np.testing.assert_allclose(
        system.current_noise[1].real, exact_noise, rtol=1e-2, atol=0
    )
