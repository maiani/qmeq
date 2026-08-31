"""Independent numerical and analytic validation of RTDnoise internals."""

from collections.abc import Callable
import warnings

import numpy as np
import pytest
from pytest import MonkeyPatch

import qmeq
import qmeq.approach.base.RTDnoise as rtdnoise_module
from qmeq.approach.rtd_blocks import FirstOrderTerm
from qmeq.approach.rtd_blocks import _coordinate_charges
from qmeq.approach.rtd_blocks import first_order_block_value
from qmeq.specfunc.specfunc import _lpm_derivative_step
from qmeq.specfunc.specfunc import integralD_lpm
from qmeq.specfunc.specfunc import integralX_lpm
from qmeq.tests.noninteracting_negf_solver import cumulants, model_from_qmeq
from qmeq.tests.rtdnoise_reference_models import build_rtdnoise_scenario


LpmIntegral = Callable[
    [bool, int, int, int, float, float, float, float, float,
     float, float, float, np.ndarray, bool],
    complex,
]

DEFAULT_RTD_TEST_BANDWIDTH = 1e5
PRACTICAL_RTD_TEST_BANDWIDTH = 50.0
BANDWIDTH_SWEEP_RATIO = 4.0
ZERO_FIELD_KERNEL_ATOL = 2e-13
BANDWIDTH_INVARIANCE_RTOL = 2e-12
BANDWIDTH_INVARIANCE_ATOL = 2e-15
CORRECTED_NOISE_ORDER_MIN = 2.9
CORRECTED_NOISE_ORDER_MAX = 3.6
NONMARKOVIAN_RESIDUAL_FRACTION_MAX = 1e-3
FIVE_POINT_DERIVATIVE_RELATIVE_ERROR_MAX = 5e-8
DEEP_BLOCKADE_GAMMA = 0.02
DEEP_BLOCKADE_LEVEL_ENERGY = -100.0
DEEP_BLOCKADE_INTERACTION = 200.0
DEEP_BLOCKADE_BIAS = 1.0
DEEP_BLOCKADE_TEMPERATURE = 0.1
DEEP_BLOCKADE_CURRENT_RTOL = 2e-3
DEEP_BLOCKADE_BANDWIDTH = 1e6


def _solve_derivative() -> np.ndarray:
    system = build_rtdnoise_scenario("double_dot_equal_temperature")
    system.solve()
    return system.appr.Lpm_second_dz.copy()


def _relative_maximum(actual: np.ndarray, expected: np.ndarray) -> float:
    scale = np.max(np.abs(expected))
    return float(np.max(np.abs(actual - expected)) / scale)


def _fourth_order_derivative(integral: LpmIntegral) -> LpmIntegral:
    """Return an independent five-point derivative for a full-kernel check."""
    def derivative(
            lpm_imaginary_2nd: bool, p1: int, eta0: int, eta1: int,
            E1: float, E2: float, E3: float, T1: float, T2: float,
            mu1: float, mu2: float, D: float, b_and_R: np.ndarray,
            ImGamma: bool) -> complex:
        step = 8.0*_lpm_derivative_step(E1, E2, E3, T1, T2)

        def shifted(offset: float) -> complex:
            return integral(
                lpm_imaginary_2nd, p1, eta0, eta1,
                E1+offset, E2+offset, E3+offset,
                T1, T2, mu1, mu2, D, b_and_R, ImGamma,
            )

        return (
            -shifted(-2.0*step) + 8.0*shifted(-step)
            - 8.0*shifted(step) + shifted(2.0*step)
        )/(12.0*step)

    return derivative


def test_scale_aware_laplace_derivative_matches_five_point_reference(
        monkeypatch: MonkeyPatch) -> None:
    """The production centered derivative matches an independent stencil."""
    production = _solve_derivative()
    monkeypatch.setattr(
        rtdnoise_module.ApproachPyRTDnoise,
        "integralD_lpm_derivative",
        staticmethod(_fourth_order_derivative(integralD_lpm)),
    )
    monkeypatch.setattr(
        rtdnoise_module.ApproachPyRTDnoise,
        "integralX_lpm_derivative",
        staticmethod(_fourth_order_derivative(integralX_lpm)),
    )
    reference = _solve_derivative()

    assert _relative_maximum(production, reference) < (
        FIVE_POINT_DERIVATIVE_RELATIVE_ERROR_MAX
    )


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


def _solve_rtd(
        scale, flux, off_diag_corrections,
        dband=DEFAULT_RTD_TEST_BANDWIDTH):
    system = qmeq.Builder(
        **_coherent_inputs(scale, flux),
        dband=dband,
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


def _solve_rtdnoise(
        scale, flux, off_diag_corrections=False,
        dband=DEFAULT_RTD_TEST_BANDWIDTH):
    system = qmeq.Builder(
        **_coherent_inputs(scale, flux),
        dband=dband,
        kerntype="RTDnoise",
        itype=1,
        countingleads=(0,),
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


def _log_slope(scales, errors):
    return np.polyfit(np.log(scales), np.log(errors), 1)[0]


def test_noninteracting_residuals_have_the_expected_coupling_orders():
    """An exact NEGF oracle separates missing second- and third-order terms.

    Scaling every width by ``Gamma`` makes the retained RTD coherence
    correction an order-``Gamma**2`` contribution.  Its omission therefore
    leaves order-``Gamma**2`` errors in both current and noise.  Once included
    in the ordinary-current calculation, its remaining error starts at
    order ``Gamma**3``.  This gate uses a real-amplitude gauge so it isolates
    the omitted coherence block from complex-amplitude partner assembly, which
    is covered independently below.
    """
    scales = np.array([0.02, 0.01, 0.005, 0.0025])
    corrected_current_errors = []
    population_current_errors = []
    noise_errors = []
    for scale in scales:
        exact = cumulants(
            model_from_qmeq(**_coherent_inputs(scale, np.pi)),
            np.array([1.0, 0.0]),
            order=2,
        )
        corrected = _solve_rtd(scale, np.pi, True)
        population = _solve_rtdnoise(scale, np.pi)
        corrected_current_errors.append(abs(corrected.current[0] - exact[1].real))
        population_current_errors.append(abs(population.current[0] - exact[1].real))
        noise_errors.append(abs(population.current_noise[1].real - exact[2].real))

    corrected_order = _log_slope(scales, corrected_current_errors)
    population_order = _log_slope(scales, population_current_errors)
    noise_order = _log_slope(scales, noise_errors)
    assert 2.8 < corrected_order < 3.2
    assert 1.9 < population_order < 2.5
    assert 1.8 < noise_order < 2.2


def test_counting_resolved_coherence_correction_reduces_to_standard_rtd():
    """At practical bandwidth, zero-field counting exactly recovers RTD."""
    standard = _solve_rtd(
        0.01, 0.0, True, dband=PRACTICAL_RTD_TEST_BANDWIDTH,
    )
    counted = _solve_rtdnoise(
        0.01, 0.0, True, dband=PRACTICAL_RTD_TEST_BANDWIDTH,
    )
    correction = counted.appr.coherence_correction
    correction_dz = counted.appr.coherence_correction_dz
    standard_wdd = (
        standard.appr.Wdd
        if standard.appr.Wdd is not None
        else standard.appr.Wdd2[0]
    )

    # The correction is real in every transfer sector and its Laplace
    # derivative purely imaginary.  The kernel's imaginary part is bitwise zero
    # because ``.real`` is taken at the write site; the derivative's real part
    # is only roundoff-small, because it is the surviving-channel expression
    # ``-1j*product_dz`` rather than a projection that constructs a zero.
    # ``test_correction_projection_keeps_the_only_nonzero_channel`` is where
    # that distinction is derived and asserted.
    np.testing.assert_array_equal(correction.imag, 0.0)
    assert np.max(np.abs(correction_dz.real)) < (
        1e-14*np.max(np.abs(correction_dz))
    )
    np.testing.assert_allclose(
        counted.appr.kern, np.sum(standard_wdd, axis=0),
        rtol=0.0, atol=ZERO_FIELD_KERNEL_ATOL,
    )
    np.testing.assert_allclose(
        counted.appr.Wdd, standard_wdd,
        rtol=0.0, atol=ZERO_FIELD_KERNEL_ATOL,
    )
    np.testing.assert_allclose(
        counted.current, standard.current,
        rtol=0.0, atol=ZERO_FIELD_KERNEL_ATOL,
    )
    np.testing.assert_allclose(
        counted.current_noise[0].real, standard.current[0],
        rtol=0.0, atol=ZERO_FIELD_KERNEL_ATOL,
    )
    np.testing.assert_allclose(counted.current_noise.imag, 0.0, atol=1e-12)
    np.testing.assert_allclose(
        np.sum(correction, axis=(1, 2, 3, 4)), 0.0, atol=3e-20,
    )
    np.testing.assert_allclose(np.sum(counted.current), 0.0, atol=1e-15)


def test_corrected_rtdnoise_has_zero_equilibrium_current():
    inputs = _coherent_inputs(0.01, 0.0)
    inputs["mulst"] = {0: 0.0, 1: 0.0}
    system = qmeq.Builder(
        **inputs,
        dband=1e5,
        kerntype="pyRTDnoise",
        itype=1,
        countingleads=(0, 1),
        off_diag_corrections=True,
    )
    system.solve()

    np.testing.assert_allclose(system.current, 0.0, atol=2e-15)
    np.testing.assert_allclose(system.current_noise[0], 0.0, atol=2e-15)
    np.testing.assert_allclose(
        np.sum(system.appr.Wdd, axis=1), 0.0, atol=3e-15,
    )
    np.testing.assert_allclose(
        system.current_noise_matrix,
        system.current_noise_matrix.T,
        atol=2e-15,
    )


def test_real_amplitude_corrected_noise_is_superquadratic_against_negf():
    """The counted Schur block removes the leading quadratic NEGF residual."""
    scales = np.array([0.04, 0.02, 0.01, 0.005])
    population_noise_errors = []
    corrected_current_errors = []
    corrected_noise_errors = []

    for scale in scales:
        exact = cumulants(
            model_from_qmeq(**_coherent_inputs(scale, 0.0)),
            np.array([1.0, 0.0]),
            order=2,
        )
        population = _solve_rtdnoise(
            scale, 0.0, False, dband=PRACTICAL_RTD_TEST_BANDWIDTH,
        )
        corrected = _solve_rtdnoise(
            scale, 0.0, True, dband=PRACTICAL_RTD_TEST_BANDWIDTH,
        )
        population_noise_errors.append(
            abs(population.current_noise[1].real - exact[2].real)
        )
        corrected_current_errors.append(
            abs(corrected.current_noise[0].real - exact[1].real)
        )
        corrected_noise_errors.append(
            abs(corrected.current_noise[1].real - exact[2].real)
        )

    population_order = _log_slope(scales, population_noise_errors)
    corrected_current_order = _log_slope(scales, corrected_current_errors)
    corrected_noise_order = _log_slope(scales, corrected_noise_errors)
    assert 1.9 < population_order < 2.4
    assert 2.8 < corrected_current_order < 3.2
    assert CORRECTED_NOISE_ORDER_MIN < corrected_noise_order
    assert corrected_noise_order < CORRECTED_NOISE_ORDER_MAX
    assert np.all(
        np.asarray(corrected_noise_errors)
        < np.asarray(population_noise_errors)
    )


def _weak_coupling_quadratic_scale(
        dband: float, temperature: float = 1.0) -> float:
    """Return the weakest-point residual normalized by ``Gamma**2``.

    This is a comparison scale, not a claim that the complete non-Markovian
    result is quadratic: after the equal-temperature integral repair its
    residual is cubic, whereas deliberately removing the Laplace terms leaves
    the expected quadratic error.
    """
    scales = np.array([1.25e-3, 6.25e-4, 3.125e-4])
    errors = []
    for scale in scales:
        inputs = _coherent_inputs(scale, 0.0)
        inputs["tlst"] = {0: temperature, 1: temperature}
        system = qmeq.Builder(
            **inputs, dband=dband, kerntype="pyRTDnoise", itype=1,
            countingleads=(0,), off_diag_corrections=True,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            system.solve()
        exact = cumulants(
            model_from_qmeq(**inputs), np.array([1.0, 0.0]), order=2
        )
        errors.append(
            abs(system.current_noise[1].real - exact[2].real)
        )
    return float(errors[-1]/scales[-1]**2)


@pytest.mark.parametrize("off_diag_corrections", [False, True])
def test_equal_temperature_noise_is_auxiliary_bandwidth_independent(
        off_diag_corrections: bool) -> None:
    """Equal-temperature RTDnoise uses one analytic zero-field kernel.

    Individual Appendix-D integrals retain ``ln(dband)`` terms, but their
    diagram sum cancels them.  Using a finite-pole approximation for only the
    counted copy spoiled that cancellation and moved the stationary projector,
    current, and noise.  A practical fourfold sweep now agrees at roundoff in
    both the historical population-only and coherence-corrected modes.
    """
    lower = _solve_rtdnoise(
        0.01, 0.0, off_diag_corrections,
        dband=PRACTICAL_RTD_TEST_BANDWIDTH,
    )
    higher = _solve_rtdnoise(
        0.01, 0.0, off_diag_corrections,
        dband=PRACTICAL_RTD_TEST_BANDWIDTH*BANDWIDTH_SWEEP_RATIO,
    )
    for left, right in [
            (lower.appr.kern, higher.appr.kern),
            (lower.phi0, higher.phi0),
            (lower.current, higher.current),
            (lower.current_noise, higher.current_noise),
            (lower.current_noise_matrix, higher.current_noise_matrix)]:
        np.testing.assert_allclose(
            left, right,
            rtol=BANDWIDTH_INVARIANCE_RTOL,
            atol=BANDWIDTH_INVARIANCE_ATOL,
        )


def test_non_markovian_correction_cancels_the_markovian_quadratic_error():
    """Gate the non-Markovian term by how much of the O(Gamma**2) error it kills.

    Dropping the Laplace-derivative arrays leaves the purely Markovian noise of
    the same counting kernel, whose residual is ``O(Gamma**2)`` by construction.
    Restoring them must cancel that term almost entirely; what is left over is
    the wide-band residue of the test above.  This is the sharp gate on the
    non-Markovian machinery: a wrong chain-rule factor, a wrong sign, or a
    dropped channel shows up here as a cancellation that is far from complete,
    whereas an exponent fit only reports it indirectly.
    """
    complete_nonmarkovian = _weak_coupling_quadratic_scale(
        PRACTICAL_RTD_TEST_BANDWIDTH,
    )

    original = rtdnoise_module.nonmarkovian_current_noise_matrix

    def without_laplace_terms(kernel, state, trace, first, second,
                              kernel_dz, first_dz):
        return original(
            kernel, state, trace, first, second,
            np.zeros_like(np.asarray(kernel_dz)),
            np.zeros_like(np.asarray(first_dz)),
        )

    rtdnoise_module.nonmarkovian_current_noise_matrix = without_laplace_terms
    try:
        markovian_only = _weak_coupling_quadratic_scale(
            PRACTICAL_RTD_TEST_BANDWIDTH,
        )
    finally:
        rtdnoise_module.nonmarkovian_current_noise_matrix = original

    assert (
        complete_nonmarkovian
        < NONMARKOVIAN_RESIDUAL_FRACTION_MAX*markovian_only
    )


def _solve_rtdnoise_at_temperature(
        scale: float, temperature: float,
        off_diag_corrections: bool) -> tuple[qmeq.Builder, dict[str, object]]:
    inputs = _coherent_inputs(scale, 0.0)
    inputs["tlst"] = {0: temperature, 1: temperature}
    system = qmeq.Builder(
        **inputs, dband=1e5, kerntype="RTDnoise", itype=1,
        countingleads=(0,), off_diag_corrections=off_diag_corrections,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        system.solve()
    return system, inputs


def test_temperature_half_noise_has_cubic_residual_in_calibrated_window(
        ) -> None:
    """Run the exact non-interacting gate where ``1/T_lead`` is not ``1``.

    Every other residual-order gate here happens to sit at ``tlst == 1``, where
    a Laplace derivative differentiated with respect to ``z/T`` instead of
    ``z`` is numerically indistinguishable from the correct one.  At
    ``T = 0.5`` that error puts the leading ``O(Gamma**2)`` term back into the
    corrected noise and the exponent collapses to ``2``.
    """
    scales = np.array([0.04, 0.02, 0.01, 0.005])
    corrected_noise_errors = []
    corrected_current_errors = []

    for scale in scales:
        corrected, inputs = _solve_rtdnoise_at_temperature(scale, 0.5, True)
        exact = cumulants(
            model_from_qmeq(**inputs), np.array([1.0, 0.0]), order=2,
        )
        corrected_current_errors.append(
            abs(corrected.current_noise[0].real - exact[1].real)
        )
        corrected_noise_errors.append(
            abs(corrected.current_noise[1].real - exact[2].real)
        )

    assert 2.7 < _log_slope(scales, corrected_noise_errors) < 3.4
    assert 2.8 < _log_slope(scales, corrected_current_errors) < 3.2


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


def test_complex_flux_second_order_kernel_matches_stationary_rtd():
    """Conjugate partners recover the stationary kernel at generic flux."""
    scale = 0.02
    real_standard = _standard_rtd_second_order_kernel(scale, 0.0)
    real_counting = np.sum(
        _solve_rtdnoise(scale, 0.0).appr.Lpm_second.real,
        axis=(0, 1, 2, 3),
    )
    assert _relative_maximum(real_counting, real_standard) < 1e-6

    flux_standard = _standard_rtd_second_order_kernel(scale, np.pi / 2.0)
    counted = _solve_rtdnoise(scale, np.pi / 2.0)
    resolved_blocks = counted.appr.Lpm_second
    resolved_derivatives = counted.appr.Lpm_second_dz
    flux_blocks = np.sum(resolved_blocks, axis=(0, 1, 2, 3))
    scale_standard = np.max(np.abs(flux_standard))
    mismatch = np.max(np.abs(flux_blocks.real - flux_standard)) / scale_standard
    residual_imaginary = np.max(np.abs(flux_blocks.imag)) / scale_standard
    assert mismatch < 1e-6
    assert residual_imaginary < 1e-14
    # The identity holds before summing transfer sectors: cancellation between
    # incorrectly labelled conjugate partners must not make this test pass.
    assert np.max(np.abs(resolved_blocks.imag)) < 1e-14*scale_standard
    assert np.max(np.abs(resolved_derivatives.real)) < (
        1e-14*np.max(np.abs(resolved_derivatives))
    )


def test_interacting_deep_blockade_matches_elastic_cotunnelling_current():
    """Current approaches the symmetric-Anderson cotunnelling limit.

    At particle-hole symmetry, elastic potential and exchange cotunnelling of
    the spin-degenerate Anderson dot give three equal squared-denominator
    contributions.

    No Poisson noise identity is imposed here. The QmeQ model has no intrinsic
    spin-relaxation bath and is therefore in the strong-cotunnelling regime of
    Sukhorukov, Burkard, and Loss, where the dot state retains memory between
    events. Their weak-cotunnelling identity ``S/I = coth(bias/(2*T))`` requires
    an intrinsic relaxation rate larger than the cotunnelling rate and does not
    apply to this model.
    """
    gamma = DEEP_BLOCKADE_GAMMA
    epsilon = DEEP_BLOCKADE_LEVEL_ENERGY
    interaction = DEEP_BLOCKADE_INTERACTION
    bias = DEEP_BLOCKADE_BIAS
    temperature = DEEP_BLOCKADE_TEMPERATURE
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
        dband=DEEP_BLOCKADE_BANDWIDTH,
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
    np.testing.assert_allclose(
        system.current_noise[0].real, expected_current,
        rtol=DEEP_BLOCKADE_CURRENT_RTOL, atol=0.0,
    )
    assert np.isfinite(system.current_noise[1])


def _rescaled_inputs(lam: float) -> dict[str, object]:
    """The calibrated model with every energy scale multiplied by ``lam``.

    ``E``, ``mu``, ``T`` and ``Gamma ~ |t|**2`` all carry one power of energy,
    so every second-order observable must be ``lam`` times a function of the
    dimensionless ratios alone.
    """
    amplitude = np.sqrt(0.01*lam/(2.0*np.pi))
    return dict(
        nsingle=2,
        hsingle={(0, 0): -1.0*lam, (1, 1): 0.7*lam, (0, 1): 0.4*lam},
        nleads=2,
        tleads={
            (0, 0): amplitude,
            (0, 1): 0.8*amplitude,
            (1, 0): 0.7*amplitude,
            (1, 1): 1.1*amplitude,
        },
        mulst={0: 1.5*lam, 1: -1.5*lam},
        tlst={0: 1.0*lam, 1: 1.3*lam},
    )


def _solve_rescaled(
        lam: float, off_diag_corrections: bool) -> qmeq.Builder:
    system = qmeq.Builder(
        **_rescaled_inputs(lam),
        dband=1e5*lam,
        kerntype="pyRTDnoise",
        itype=1,
        countingleads=(0,),
        off_diag_corrections=off_diag_corrections,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        system.solve()
    return system


@pytest.mark.parametrize("off_diag_corrections", [False, True])
def test_noise_is_covariant_under_an_overall_energy_rescaling(
        off_diag_corrections: bool) -> None:
    """``noise/lam`` is invariant under a common energy rescaling.

    This is a closed self-consistency gate requiring no external reference:
    rescaling is an exact symmetry of the second-order theory rather than an
    approximation to it. It catches a Laplace derivative missing its
    ``1/T_lead`` factor, because that term scales as ``lam**2`` while every
    legitimate one scales as ``lam``. Every other physics gate here runs at
    ``tlst == 1``, where that error is invisible.

    The sweep runs far *below* ``lam == 1`` as well as above, because a second
    unit-covariance defect is one-sided: a numerical differentiation step with
    an absolute floor tracks the model upward but not downward. The model's
    largest scale is about ``1.9`` at ``lam == 1``, so the small-``lam`` points
    are the ones a ``max(1.0, ...)`` clamp would have differentiated with a
    step orders of magnitude too coarse relative to their own features -- at
    ``lam = 1e-3`` the clamped step is some ``500`` times the correct one.
    """
    reference = _solve_rescaled(1.0, off_diag_corrections)
    baseline = reference.current_noise[1].real
    for lam in (7.0, 4.0, 2.0, 0.25, 0.05, 1e-3):
        scaled = _solve_rescaled(lam, off_diag_corrections)
        np.testing.assert_allclose(
            scaled.current_noise[1].real/lam, baseline,
            rtol=1e-13, atol=0.0,
        )
        np.testing.assert_allclose(
            scaled.current[0]/lam, reference.current[0], rtol=1e-13, atol=0.0,
        )


def _diagonal_limit_first_order_blocks(
        system: qmeq.Builder) -> tuple[np.ndarray, np.ndarray]:
    """Assemble the ``dn`` block with its two coherence indices collapsed.

    Setting ``a1 == b1`` makes the population/coherence block, diagram by
    diagram, the *diagonal* first-order kernel: identical vertices, energies
    and tunnel products.  So both the value and the Laplace derivative must
    reproduce ``Lpm_first`` / ``Lpm_first_dz``, which the exact non-interacting
    gate below constrains directly.  Nothing else pins the coherence block's
    derivative, whose own contribution to the noise sits at ``O(Gamma**3)``.
    """
    approach = system.appr
    si = approach.si
    size = approach.get_kern_size()
    value = np.zeros((si.nleads, size, size), dtype=complex)
    derivative = np.zeros((si.nleads, size, size), dtype=complex)

    def collect(
            direction: str, value_real: float, value_imag: float,
            derivative_real: float, derivative_imag: float,
            lead: int, b: int, bp: int, bcharge: int,
            a: int, ap: int, acharge: int) -> None:
        row = si.get_ind_dm0(b, bp, bcharge)
        column = si.get_ind_dm0(a, ap, acharge)
        value[lead, row, column] += value_real + 1j*value_imag
        derivative[lead, row, column] += (
            derivative_real + 1j*derivative_imag
        )

    original = approach._add_coherence_element
    approach._add_coherence_element = collect
    try:
        for charge in range(si.ncharge):
            for state in si.statesdm[charge]:
                approach.generate_col_nondiag_kern_1st_order_dn(
                    state, state, charge
                )
    finally:
        approach._add_coherence_element = original
    return value, derivative


def test_coherence_block_reduces_to_the_diagonal_first_order_kernel() -> None:
    """The ``a1 == b1`` limit of the block is the diagonal kernel, exactly.

    Run at *unequal* lead temperatures so the reduction also pins the
    per-lead ``1/T`` of the Laplace derivative, and so that the real
    ``pi*f'`` channel -- which cancels only when the two Keldysh partners share
    an energy -- is exercised on the way in.
    """
    inputs = _coherent_inputs(0.01, 0.0)
    inputs["tlst"] = {0: 0.7, 1: 0.91}
    system = qmeq.Builder(
        **inputs, dband=1e5, kerntype="pyRTDnoise", itype=1,
        countingleads=(0,), off_diag_corrections=True,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        system.solve()

    value, derivative = _diagonal_limit_first_order_blocks(system)
    np.testing.assert_allclose(
        value, np.sum(system.appr.Lpm_first, axis=1), rtol=0.0, atol=1e-17,
    )
    np.testing.assert_allclose(
        derivative, np.sum(system.appr.Lpm_first_dz, axis=1),
        rtol=0.0, atol=1e-17,
    )
    # The diagonal kernel is real and its Laplace derivative imaginary; the
    # collapse must not leak between those channels.
    np.testing.assert_allclose(value.imag, 0.0, atol=1e-18)
    np.testing.assert_allclose(derivative.real, 0.0, atol=1e-18)


def _three_level_inputs() -> dict[str, object]:
    amplitude = 0.1/np.sqrt(2.0*np.pi)
    return dict(
        nsingle=3,
        hsingle={(0, 0): -1.0, (1, 1): 0.3, (2, 2): 0.9,
                 (0, 1): 0.2, (1, 2): 0.15},
        coulomb={(0, 1, 1, 0): 0.5},
        nleads=2,
        tleads={(lead, state): amplitude*(1.0 + 0.1*state + 0.2*lead)
                for lead in range(2) for state in range(3)},
        mulst={0: 1.5, 1: -1.5},
        tlst={0: 1.0, 1: 1.0},
        dband=1e5,
    )


def test_corrections_support_charge_sectors_two_electrons_apart() -> None:
    """Three levels put populations two charges away from a coherence.

    Those coordinate pairs are not connected by a first-order vertex, so their
    block is structurally zero and carries no transfer to resolve.  Rejecting
    them by charge alone -- rather than by a nonzero amplitude -- made
    ``off_diag_corrections=True`` unusable for any dot with more than two
    levels, which no two-level scenario can detect.
    """
    counted = qmeq.Builder(
        **_three_level_inputs(), kerntype="pyRTDnoise", itype=1,
        countingleads=(0,), off_diag_corrections=True,
    )
    population = qmeq.Builder(
        **_three_level_inputs(), kerntype="pyRTD", itype=1,
        off_diag_corrections=True,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        counted.solve()
        population.solve()

    assert counted.success
    correction = counted.appr.coherence_correction
    # Every retained sector is a single-electron transfer, so summing them
    # must return the ordinary RTD correction.
    np.testing.assert_allclose(
        counted.appr.Wdd, population.appr.Wdd, atol=2e-13,
    )
    np.testing.assert_allclose(counted.current, population.current, atol=2e-13)
    np.testing.assert_allclose(correction.imag, 0.0)
    np.testing.assert_allclose(
        np.sum(correction, axis=(1, 2, 3, 4)), 0.0, atol=1e-18,
    )


def test_complex_flux_rtdnoise_observables_have_cubic_residuals():
    """RTDnoise retains the full second-order current and noise at generic flux.

    Ordinary RTD keeps its full ``O(Gamma**2)`` accuracy at a generic flux, so
    the ``.real`` projections in its second-order assembly are an identity
    there rather than a truncation.  Completing RTDnoise's transfer-resolved
    conjugate partners must give the same cubic residual against the exact
    non-interacting current instead of dropping a physical ``O(Gamma**2)``
    contribution.
    """
    scales = np.array([0.04, 0.02, 0.01, 0.005])
    flux = np.pi/2.0
    rtd_errors = []
    counting_errors = []
    noise_errors = []
    for scale in scales:
        exact = cumulants(
            model_from_qmeq(**_coherent_inputs(scale, flux)),
            np.array([1.0, 0.0]), order=2,
        )
        rtd_errors.append(
            abs(_solve_rtd(scale, flux, True).current[0] - exact[1].real)
        )
        counted = _solve_rtdnoise(scale, flux, True)
        counting_errors.append(abs(counted.current[0] - exact[1].real))
        noise_errors.append(abs(counted.current_noise[1].real - exact[2].real))

    assert 2.8 < _log_slope(scales, rtd_errors) < 3.2
    assert 2.8 < _log_slope(scales, counting_errors) < 3.2
    assert 2.7 < _log_slope(scales, noise_errors) < 3.4


def test_complex_flux_observables_are_invariant_under_orbital_rephasing():
    """Gauge redistribution and a full flux period change no observable."""
    inputs = _coherent_inputs(0.02, np.pi/2.0)
    reference = _solve_rtdnoise(0.02, np.pi/2.0, True)
    periodic = _solve_rtdnoise(0.02, np.pi/2.0 + 2.0*np.pi, True)

    phases = np.array([0.37, -0.22])
    rephased = dict(inputs)
    # For d'_i = exp(i*phase_i) d_i, covariance requires
    # h'_ij = exp(i*(phase_i-phase_j))*h_ij and
    # t'_{ri} = exp(-i*phase_i)*t_{ri}.
    rephased["hsingle"] = {
        (0, 0): -1.0,
        (1, 1): 0.7,
        (0, 1): 0.4*np.exp(1j*(phases[0] - phases[1])),
    }
    rephased["tleads"] = {
        (lead, orbital): amplitude*np.exp(-1j*phases[orbital])
        for (lead, orbital), amplitude in inputs["tleads"].items()
    }
    transformed = qmeq.Builder(
        **rephased, dband=1e5, kerntype="pyRTDnoise", itype=1,
        countingleads=(0,), off_diag_corrections=True,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        transformed.solve()

    np.testing.assert_allclose(
        transformed.current, reference.current, rtol=0.0, atol=1e-15,
    )
    np.testing.assert_allclose(
        transformed.current_noise, reference.current_noise,
        rtol=0.0, atol=1e-14,
    )
    np.testing.assert_allclose(
        periodic.current_noise, reference.current_noise,
        rtol=0.0, atol=1e-14,
    )


def test_transfer_resolved_correction_survives_complex_amplitudes():
    """Re-bucketing the correction by lead and transfer is exact at any flux.

    The correction itself is built from RTD's own ``nd``/``dn`` blocks, so this
    independently gates the counting-resolved *composition* after the
    population traversal's conjugate-partner repair. Note the companion
    diagonal-limit test cannot reach here: setting the two coherence indices
    equal makes the tunnel product ``|T|**2``, real by construction.
    """
    def population_kernel(flux, off_diag_corrections):
        system = qmeq.Builder(
            **_coherent_inputs(0.02, flux), dband=1e5, kerntype="pyRTD",
            itype=1, off_diag_corrections=off_diag_corrections,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            system.solve()
        return system.appr.Wdd

    for flux in (0.0, np.pi/2.0):
        counted = _solve_rtdnoise(0.02, flux, True)
        reference = (
            population_kernel(flux, True) - population_kernel(flux, False)
        )
        correction = counted.appr.coherence_correction
        np.testing.assert_allclose(
            np.sum(correction.real, axis=(1, 2, 3)), reference,
            rtol=0.0, atol=1e-13*np.max(np.abs(reference)),
        )
        np.testing.assert_array_equal(correction.imag, 0.0)


def _record_first_order_blocks(
        system: qmeq.Builder) -> list[tuple[object, ...]]:
    """Re-run the block traversal, capturing term tables with coordinates.

    ``_coherence_block_derivative`` receives the term table and
    ``_add_coherence_element`` receives the insertion coordinates, and the two
    are called in strict pairs, so zipping them in call order recovers both.
    The production arrays are deliberately left untouched: the replacement
    ``_add_coherence_element`` records and returns without inserting.
    """
    approach = system.appr
    si = approach.si
    handler = approach.kernel_handler
    records: list[tuple[object, ...]] = []
    pending: list[tuple[object, ...]] = []

    original_derivative = approach._coherence_block_derivative
    original_add = approach._add_coherence_element

    def capture_terms(tunnel_product, temperature, *terms):
        pending.append((tunnel_product, temperature, terms))
        return original_derivative(tunnel_product, temperature, *terms)

    def capture_coordinates(direction, value_real, value_imag,
                            derivative_real, derivative_imag,
                            lead, b, bp, bcharge, a, ap, acharge):
        tunnel_product, temperature, terms = pending.pop()
        records.append((
            direction, tunnel_product, temperature, terms, lead,
            b, bp, bcharge, a, ap, acharge, value_real + 1j*value_imag,
        ))

    approach._coherence_block_derivative = capture_terms
    approach._add_coherence_element = capture_coordinates
    try:
        for charge in range(si.ncharge):
            for state in si.statesdm[charge]:
                if handler.is_unique(state, state, charge):
                    approach.generate_col_nondiag_kern_1st_order_nd(
                        state, charge
                    )
        for charge in range(si.ncharge):
            for state in si.statesdm[charge]:
                for partner in si.statesdm[charge]:
                    if state != partner:
                        approach.generate_col_nondiag_kern_1st_order_dn(
                            state, partner, charge
                        )
    finally:
        approach._coherence_block_derivative = original_derivative
        approach._add_coherence_element = original_add
    return records


def _blocks_at_laplace_energy(
        system: qmeq.Builder, records: list[tuple[object, ...]],
        laplace: float) -> tuple[np.ndarray, np.ndarray]:
    """Assemble ``Wdn(z)`` and ``Wnd(z)`` in the packed coherence layout."""
    approach = system.appr
    handler = approach.kernel_handler
    npauli = approach.get_kern_size()
    ncoherences = approach.Lnn_inv.shape[0]
    bands = approach.leads.dlst
    dn = np.zeros((approach.si.nleads, npauli, ncoherences), dtype=complex)
    nd = np.zeros((approach.si.nleads, ncoherences, npauli), dtype=complex)
    for record in records:
        (direction, tunnel_product, temperature, terms, lead,
         b, bp, bcharge, a, ap, acharge, _value) = record
        value = first_order_block_value(
            tunnel_product, temperature,
            bands[lead, 0]/temperature, bands[lead, 1]/temperature,
            tuple(FirstOrderTerm(*term) for term in terms), laplace,
        )
        handler.add_matrix_element_to(
            dn if direction == "dn" else nd, value,
            lead, b, bp, bcharge, a, ap, acharge,
        )
    return dn, nd


def _finite_laplace_correction(
        system: qmeq.Builder, records: list[tuple[object, ...]],
        laplace: float, resolvent_sign: float) -> np.ndarray:
    """Transfer-resolved ``W_corr(z)``, built independently of the production
    product rule.

    ``resolvent_sign = +1`` means the bare coherence resolvent is
    ``1/(dE + z)``, i.e. ``dG/dz = -G**2``, which is what
    ``counting_resolved_coherence_correction`` assumes.
    """
    approach = system.appr
    si = approach.si
    npauli = approach.get_kern_size()
    ncoherences = approach.Lnn_inv.shape[0]
    population_charge, coherence_charge = _coordinate_charges(si)
    dn, nd = _blocks_at_laplace_energy(system, records, laplace)

    resolved_dn = np.zeros(
        (si.nleads, 3, npauli, ncoherences), dtype=complex
    )
    resolved_nd = np.zeros(
        (si.nleads, 3, ncoherences, npauli), dtype=complex
    )
    for population in range(npauli):
        for coherence in range(ncoherences):
            transfer = (
                population_charge[population] - coherence_charge[coherence]
            )
            if abs(transfer) > 1:
                continue
            resolved_dn[:, transfer, population, coherence] = (
                dn[:, population, coherence]
            )
            resolved_nd[:, -transfer, coherence, population] = (
                nd[:, coherence, population]
            )

    splitting = 1.0/np.diag(approach.Lnn_inv)
    resolvent = np.diag(1.0/(splitting + resolvent_sign*laplace))
    correction = np.zeros(
        (si.nleads, si.nleads, 3, 3, npauli, npauli), dtype=complex
    )
    for lead_dn in range(si.nleads):
        for lead_nd in range(si.nleads):
            for transfer_dn in (-1, 0, 1):
                for transfer_nd in (-1, 0, 1):
                    correction[lead_dn, lead_nd, transfer_dn, transfer_nd] = (
                        -1j*(
                            resolved_dn[lead_dn, transfer_dn]
                            @ resolvent
                            @ resolved_nd[lead_nd, transfer_nd]
                        )
                    )
    return correction


def _corrected_system(temperatures=(0.7, 0.91)):
    inputs = _coherent_inputs(0.01, 0.0)
    inputs["tlst"] = {0: temperatures[0], 1: temperatures[1]}
    system = qmeq.Builder(
        **inputs, dband=1e5, kerntype="pyRTDnoise", itype=1,
        countingleads=(0,), off_diag_corrections=True,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        system.solve()
    return system


def test_finite_laplace_energy_reproduces_the_stored_blocks():
    """The term decomposition is pinned against the historical expressions.

    ``first_order_block_value`` at ``z = 0`` must reproduce the block element
    that the untouched ``temp1 + 1j*temp2`` expressions insert.  This is the
    non-circular anchor for every finite-``z`` statement below: the
    decomposition is not asserted from a derivation, it is matched element by
    element against the code that has always computed the value.
    """
    system = _corrected_system()
    approach = system.appr
    reference_dn = (approach.ReWdn + 1j*approach.ImWdn).copy()
    reference_nd = (approach.ReWnd + 1j*approach.ImWnd).copy()

    records = _record_first_order_blocks(system)
    assert records
    for record in records:
        rebuilt = first_order_block_value(
            record[1], record[2],
            approach.leads.dlst[record[4], 0]/record[2],
            approach.leads.dlst[record[4], 1]/record[2],
            tuple(FirstOrderTerm(*term) for term in record[3]), 0.0,
        )
        # Summation order differs from the historical expression, so this is a
        # last-bit comparison rather than a bitwise one.
        np.testing.assert_allclose(rebuilt, record[-1], rtol=1e-14, atol=0.0)

    dn, nd = _blocks_at_laplace_energy(system, records, 0.0)
    np.testing.assert_allclose(dn, reference_dn, rtol=0.0, atol=1e-17)
    np.testing.assert_allclose(nd, reference_nd, rtol=0.0, atol=1e-17)


def test_correction_projection_keeps_the_only_nonzero_channel():
    """The real/imaginary projections are identities, not truncations.

    Two structural facts fix the convention, and both are measured here rather
    than assumed.  The zero-field Schur product ``Wdn G Wnd`` is purely
    imaginary, so ``(-1j*product).real`` discards nothing; its Laplace
    derivative is purely real, so ``-1j*product_dz`` likewise discards nothing.
    Taking ``.imag`` of the derivative instead -- as an apparent symmetry with
    the value would suggest -- keeps the identically zero channel and silently
    drops the whole term.

    The pair is therefore one analytic object, ``W_corr(z) = -1j*Wdn(z) G(z)
    Wnd(z)``, and it matches the package-wide convention asserted below: every
    kernel array is real and every Laplace-derivative array purely imaginary.
    That convention is not cosmetic -- ``nonmarkovian_current_noise_matrix``
    forms ``currents*responses`` from them, and a real-valued noise depends on
    it ([Emary2009, Eqs. (40)-(41)]).
    """
    approach = _corrected_system().appr
    dn = np.sum(approach.ReWdn + 1j*approach.ImWdn, axis=0)
    nd = np.sum(approach.ReWnd + 1j*approach.ImWnd, axis=0)
    dn_dz = np.sum(approach.ReWdn_dz + 1j*approach.ImWdn_dz, axis=0)
    nd_dz = np.sum(approach.ReWnd_dz + 1j*approach.ImWnd_dz, axis=0)
    resolvent = approach.Lnn_inv

    product = dn @ resolvent @ nd
    product_dz = (
        dn_dz @ resolvent @ nd
        - dn @ (resolvent @ resolvent) @ nd
        + dn @ resolvent @ nd_dz
    )
    assert np.max(np.abs(product.real)) < 1e-15*np.max(np.abs(product))
    assert np.max(np.abs(product_dz.imag)) < 1e-15*np.max(np.abs(product_dz))

    for kernel, derivative in (
        (approach.Lpm_first, approach.Lpm_first_dz),
        (approach.Lpm_second, approach.Lpm_second_dz),
        (approach.coherence_correction, approach.coherence_correction_dz),
    ):
        kernel = np.asarray(kernel)
        derivative = np.asarray(derivative)
        assert np.max(np.abs(kernel.imag)) < 1e-14*np.max(np.abs(kernel))
        assert np.max(np.abs(derivative.real)) < (
            1e-14*np.max(np.abs(derivative))
        )
    assert np.max(np.abs(approach.coherence_correction_dz)) > 0.0


def test_bare_resolvent_uses_oriented_coherence_bohr_frequencies():
    """Map every RTD coherence slot to its derived free resolvent.

    The free molecular line is ``Pi0(z_LW) = 1j*(z_LW - L)^-1``
    [LeijnseWegewijs2008, Eq. (49)].  For ``|a><b|``, ``L`` contributes the
    oriented Bohr frequency ``E[a] - E[b]``.  QmeQ stores the reversed
    coherence in a separate RTD slot (layout rule L9), so its splitting changes
    sign while the common continuation ``z_LW = -z`` does not.  Extracting the
    line's ``-1j`` leaves ``G_ab(z) = 1/(E[a] - E[b] + z)``.
    """
    approach = _corrected_system().appr
    si = approach.si
    npauli = approach.get_kern_size()
    imag_offset = si.ndm0 - si.npauli
    expected_splittings = np.full(approach.Lnn_inv.shape[0], np.nan)

    for charge in range(si.ncharge):
        for a in si.statesdm[charge]:
            for b in si.statesdm[charge]:
                if a == b:
                    continue
                slot = si.get_ind_dm0(a, b, charge) - npauli
                if a > b:
                    slot += imag_offset
                splitting = approach.qd.Ea[a] - approach.qd.Ea[b]
                if 0.0 <= splitting < 1e-10:
                    splitting = 1e-10
                elif -1e-10 < splitting <= 0.0:
                    splitting = -1e-10
                expected_splittings[slot] = splitting

    assert np.all(np.isfinite(expected_splittings))
    stored = np.diag(approach.Lnn_inv)
    np.testing.assert_allclose(
        stored, 1.0/expected_splittings, rtol=0.0, atol=0.0,
    )

    # This is also the local analytic input used by the product rule.  The
    # negative sign comes from differentiating the derived +z denominator.
    derivative = np.diag(-1.0/expected_splittings**2)
    np.testing.assert_allclose(
        derivative, -(approach.Lnn_inv @ approach.Lnn_inv),
        rtol=0.0, atol=0.0,
    )


def test_finite_laplace_correction_derivative_is_directly_gated():
    """Gate ``coherence_correction_dz`` against a finite-``z`` reference.

    The reference re-derives every transfer sector from
    ``W_corr(z) = -1j*Wdn(z) G(z) Wnd(z)`` and central-differences it, using no
    part of the production product rule.  Neither the zero-field reduction to
    ordinary RTD, nor the diagonal limit, nor the non-interacting residual
    order constrains this quantity: it enters the noise only at
    ``O(Gamma**3)``.

    The bare-resolvent orientation is ``G(z) = 1/(dE + z)``. It follows from
    ``Pi0(z_LW) = 1j/(z_LW - dE)`` with QmeQ's ``z_LW = -z`` and the line's
    ``-1j`` extracted into the Schur product
    [LeijnseWegewijs2008, Eq. (49)]. The negative control shows that the
    opposite orientation differs by more than a factor of two.
    """
    system = _corrected_system()
    records = _record_first_order_blocks(system)
    splitting = np.abs(1.0/np.diag(system.appr.Lnn_inv))
    step = 1e-5*float(np.max(splitting))

    def numerical(resolvent_sign):
        upper = _finite_laplace_correction(
            system, records, step, resolvent_sign
        )
        lower = _finite_laplace_correction(
            system, records, -step, resolvent_sign
        )
        return (upper - lower)/(2.0*step)

    stored = system.appr.coherence_correction_dz
    matching = numerical(+1.0)
    scale = np.max(np.abs(matching))
    assert scale > 0.0
    np.testing.assert_allclose(matching, stored, rtol=0.0, atol=1e-7*scale)

    opposite = numerical(-1.0)
    assert np.max(np.abs(opposite - stored)) > scale


def test_corrected_noise_is_cubic_at_practical_bandwidth():
    """The independent U=0 oracle rejects a residual quadratic term.

    With the analytic equal-temperature counted integral, ``dband`` is only the
    regulator appearing in the Appendix-D expressions and their logarithms
    cancel in the diagram sum.  The cubic RTD truncation error must therefore
    already be visible at a practical bandwidth; requiring ``dband=1e7`` would
    merely hide a mismatched zero-field kernel behind an expensive pole sum.
    """
    scales = np.array([0.04, 0.02, 0.01, 0.005])
    errors = []
    for scale in scales:
        inputs = _coherent_inputs(scale, 0.0)
        system = qmeq.Builder(
            **inputs, dband=PRACTICAL_RTD_TEST_BANDWIDTH,
            kerntype="pyRTDnoise", itype=1,
            countingleads=(0,), off_diag_corrections=True,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            system.solve()
        exact = cumulants(
            model_from_qmeq(**inputs), np.array([1.0, 0.0]), order=2
        )
        errors.append(abs(system.current_noise[1].real - exact[2].real))

    corrected_noise_order = _log_slope(scales, errors)
    assert CORRECTED_NOISE_ORDER_MIN < corrected_noise_order
    assert corrected_noise_order < CORRECTED_NOISE_ORDER_MAX
