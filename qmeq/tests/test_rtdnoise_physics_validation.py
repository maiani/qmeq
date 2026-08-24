"""Independent numerical and analytic validation of RTDnoise internals."""

import warnings

import numpy as np

import qmeq
from qmeq.tests.noninteracting_negf_solver import cumulants, model_from_qmeq
from qmeq.tests.rtdnoise_reference_models import build_rtdnoise_scenario


def _solve_derivative(step):
    system = build_rtdnoise_scenario("double_dot_equal_temperature")
    system.appr.lpm_h = step
    system.solve()
    return system.appr.Lpm_second_dot.copy()


def _relative_maximum(actual, expected):
    scale = np.max(np.abs(expected))
    return np.max(np.abs(actual - expected)) / scale


def test_laplace_derivative_step_sweep_has_a_measured_accuracy_floor():
    """Richardson controls truncation error and exposes the roundoff floor."""
    steps = (1e-4, 5e-5, 2e-5, 1e-5, 1e-6, 1e-8, 1e-9)
    values = {step: _solve_derivative(step) for step in steps}
    reference = 2.0 * values[5e-5] - values[1e-4]
    independent = 2.0 * values[1e-5] - values[2e-5]

    assert _relative_maximum(independent, reference) < 1e-6
    errors = {
        step: _relative_maximum(values[step], reference)
        for step in (1e-4, 1e-6, 1e-8, 1e-9)
    }
    assert errors[1e-4] > errors[1e-6] > errors[1e-8]
    assert 1e-9 < errors[1e-8] < 2e-7
    assert errors[1e-9] > errors[1e-8]


def _coherent_inputs(scale, flux):
    amplitude = np.sqrt(scale / (2.0 * np.pi))
    return dict(
        nsingle=2,
        hsingle={(0, 0): -1.0, (1, 1): 0.7, (0, 1): 0.4},
        nleads=2,
        tleads={
            (0, 0): amplitude,
            (0, 1): 0.8 * amplitude * np.exp(1j * flux),
            (1, 0): 0.7 * amplitude,
            (1, 1): 1.1 * amplitude,
        },
        mulst={0: 1.5, 1: -1.5},
        tlst={0: 1.0, 1: 1.0},
    )


def _solve_rtd(scale, flux, off_diag_corrections):
    system = qmeq.Builder(
        **_coherent_inputs(scale, flux),
        dband=1e5,
        kerntype="RTD",
        itype=1,
        off_diag_corrections=off_diag_corrections,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*energy_current and heat_current.*",
            category=qmeq.QmeqRuntimeWarning,
        )
        system.solve()
    return system


def _solve_rtdnoise(scale, flux):
    system = qmeq.Builder(
        **_coherent_inputs(scale, flux),
        dband=1e5,
        kerntype="RTDnoise",
        itype=1,
        countingleads=(0,),
        off_diag_corrections=False,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*energy_current and heat_current.*",
            category=qmeq.QmeqRuntimeWarning,
        )
        system.solve()
    return system


def _log_slope(scales, errors):
    return np.polyfit(np.log(scales), np.log(errors), 1)[0]


def test_noninteracting_residuals_have_the_expected_coupling_orders():
    """An exact NEGF oracle separates missing second- and third-order terms.

    Scaling every width by ``Gamma`` makes the retained RTD coherence
    correction an order-``Gamma**2`` contribution.  Its omission therefore
    leaves order-``Gamma**2`` errors in both current and noise.  Once included
    in the ordinary-current calculation, its remaining error starts at
    order ``Gamma**3``.
    """
    scales = np.array([0.02, 0.01, 0.005, 0.0025])
    corrected_current_errors = []
    population_current_errors = []
    noise_errors = []
    for scale in scales:
        exact = cumulants(
            model_from_qmeq(**_coherent_inputs(scale, 0.45)),
            np.array([1.0, 0.0]),
            order=2,
        )
        corrected = _solve_rtd(scale, 0.45, True)
        population = _solve_rtdnoise(scale, 0.45)
        corrected_current_errors.append(abs(corrected.current[0] - exact[1].real))
        population_current_errors.append(abs(population.current[0] - exact[1].real))
        noise_errors.append(abs(population.current_noise[1].real - exact[2].real))

    corrected_order = _log_slope(scales, corrected_current_errors)
    population_order = _log_slope(scales, population_current_errors)
    noise_order = _log_slope(scales, noise_errors)
    assert 2.8 < corrected_order < 3.2
    assert 1.9 < population_order < 2.5
    assert 1.8 < noise_order < 2.2


def _standard_rtd_second_order_kernel(scale, flux):
    full = qmeq.Builder(
        **_coherent_inputs(scale, flux),
        dband=1e5,
        kerntype="pyRTD",
        itype=1,
        off_diag_corrections=False,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*energy_current and heat_current.*",
            category=qmeq.QmeqRuntimeWarning,
        )
        full.solve()
    sequential = qmeq.Builder(
        **_coherent_inputs(scale, flux),
        dband=1e5,
        kerntype="pyRTD",
        itype=1,
        off_diag_corrections=False,
    )
    original = sequential.appr.generate_col_diag_kern_2nd_order
    sequential.appr.generate_col_diag_kern_2nd_order = lambda *args: None
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r".*energy_current and heat_current.*",
                category=qmeq.QmeqRuntimeWarning,
            )
            sequential.solve()
    finally:
        sequential.appr.generate_col_diag_kern_2nd_order = original
    return np.sum(full.appr.Wdd - sequential.appr.Wdd, axis=0)


def test_complex_flux_second_order_kernel_defect_is_reproduced_directly():
    """The counting traversal currently drops a physical imaginary channel."""
    scale = 0.02
    real_standard = _standard_rtd_second_order_kernel(scale, 0.0)
    real_counting = np.sum(
        _solve_rtdnoise(scale, 0.0).appr.Lpm_second.real,
        axis=(0, 1, 2, 3),
    )
    assert _relative_maximum(real_counting, real_standard) < 1e-6

    flux_standard = _standard_rtd_second_order_kernel(scale, np.pi / 2.0)
    flux_blocks = np.sum(
        _solve_rtdnoise(scale, np.pi / 2.0).appr.Lpm_second,
        axis=(0, 1, 2, 3),
    )
    scale_standard = np.max(np.abs(flux_standard))
    mismatch = np.max(np.abs(flux_blocks.real - flux_standard)) / scale_standard
    omitted_imaginary = np.max(np.abs(flux_blocks.imag)) / scale_standard
    assert mismatch > 0.5
    assert omitted_imaginary > 0.4


def test_interacting_deep_blockade_matches_elastic_cotunnelling_limit():
    """Current and noise approach the bidirectional-Poisson cotunnelling limit.

    At particle-hole symmetry, elastic potential and exchange cotunnelling of
    the spin-degenerate Anderson dot give three equal squared-denominator
    contributions.  Forward and backward rare events then give
    ``S/I = coth(bias/(2*T))`` in QmeQ's ``d Var/dt`` noise convention.
    """
    gamma = 0.02
    epsilon = -50.0
    interaction = 100.0
    bias = 1.0
    temperature = 0.1
    tunnel = np.sqrt(gamma / (2.0 * np.pi))
    system = qmeq.Builder(
        nsingle=2,
        hsingle={(0, 0): epsilon, (1, 1): epsilon},
        coulomb={(0, 1, 1, 0): interaction},
        nleads=4,
        tleads={
            (0, 0): tunnel,
            (1, 1): tunnel,
            (2, 0): tunnel,
            (3, 1): tunnel,
        },
        mulst={0: bias / 2, 1: bias / 2, 2: -bias / 2, 3: -bias / 2},
        tlst={lead: temperature for lead in range(4)},
        dband=1e6,
        kerntype="pyRTDnoise",
        itype=1,
        countingleads=(0, 1),
        off_diag_corrections=False,
    )
    # The sequential kernel has two disconnected spin sectors in this limit,
    # so its standalone first-order noise is undefined.  Generate only the
    # full, uniquely stationary cotunnelling observables under test.
    system.solve(currentq=False)
    system.appr.generate_current_noise()
    system.appr.generate_current()

    denominator = 1.0 / epsilon**2 + 1.0 / (epsilon + interaction) ** 2
    expected_current = 3.0 * gamma**2 * bias * denominator / (2.0 * np.pi)
    expected_noise = expected_current / np.tanh(bias / (2.0 * temperature))
    np.testing.assert_allclose(
        system.current_noise[0].real, expected_current, rtol=2e-3, atol=0.0,
    )
    np.testing.assert_allclose(
        system.current_noise[1].real, expected_noise, rtol=1e-2, atol=0.0,
    )
