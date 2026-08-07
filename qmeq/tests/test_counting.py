"""Regression and independent checks for zero-frequency counting statistics."""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import expit

import qmeq
from qmeq.approach.counting import markovian_current_noise
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


def _rtd_system(gate, bias):
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
        countingleads=(0,),
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
    np.testing.assert_allclose(system.current_noise[0], system.current[1])
    np.testing.assert_allclose(system.current_noise[1], left[1])
    assert np.sign(system.current_noise[0]) == -np.sign(left[0])

    system.change(countingleads=None)
    system.solve()
    assert system.current_noise is None
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
