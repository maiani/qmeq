"""The diagram records assemble exactly what the nested RTD loops assemble.

RTD and RTDnoise evaluate the records of :mod:`qmeq.approach.rtd_diagrams`.
Their ``_legacy_*`` methods enumerate the same diagrams with hand-written
nested loops. Every array either path assembles must be equal bit for bit, on
each scenario of the RTD and RTDnoise reference matrices and on the
arbitrary-system stress cases, for real and complex amplitudes.
"""

import warnings

import numpy as np
import pytest

from qmeq.tests.qmeq_11_reference_models import (
    RTD_REFERENCE_SCENARIOS,
    build_rtd_reference_system,
)
from qmeq.tests.rtdnoise_reference_models import (
    LIVE_ARBITRARY_SYSTEM_SCENARIOS,
    RTDNOISE_LIVE_INVARIANT_SCENARIOS,
    build_rtdnoise_arbitrary_system_scenario,
    build_rtdnoise_scenario,
)

_TRAVERSALS = (
    "generate_row_1st_order_kernel",
    "generate_col_diag_kern_2nd_order",
    "generate_row_1st_order_kernel_lpm",
    "generate_col_diag_kern_2nd_order_lpm",
)

_RTD_ARRAYS = ("Wdd", "WE1", "WE2", "kern", "phi0", "current")

_RTDNOISE_ARRAYS = (
    "Wdd", "kern", "kern_first", "kern_second", "phi0", "Lpm", "Lpm_first",
    "Lpm_first_dz", "Lpm_second", "Lpm_second_dz", "current_noise",
    "current_noise_first", "current_noise_o4trunc", "current_noise_matrix",
    "current_noise_matrix_first",
)


def _solve(build, legacy, **solve_options):
    system = build()
    if legacy:
        appr = system.appr
        for name in _TRAVERSALS:
            if hasattr(appr, "_legacy_" + name):
                setattr(appr, name, getattr(appr, "_legacy_" + name))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        system.solve(**solve_options)
    return system.appr


def _assert_bitwise_equal(build, names, **solve_options):
    records = _solve(build, legacy=False, **solve_options)
    loops = _solve(build, legacy=True, **solve_options)
    for name in names:
        expected, actual = getattr(loops, name), getattr(records, name)
        assert np.array_equal(actual, expected, equal_nan=True), name


@pytest.mark.parametrize("scenario", RTD_REFERENCE_SCENARIOS)
def test_rtd_records_reproduce_the_loops(scenario):
    # Many-body input is already diagonal, as in solve_rtd_reference_system.
    options = {"qdq": False, "rotateq": False} if scenario == "many_body" else {}
    _assert_bitwise_equal(
        lambda: build_rtd_reference_system(scenario), _RTD_ARRAYS, **options)


@pytest.mark.parametrize("selected", [False, True], ids=["pyRTDnoise", "RTDnoise"])
@pytest.mark.parametrize("scenario", RTDNOISE_LIVE_INVARIANT_SCENARIOS)
def test_rtdnoise_records_reproduce_the_loops(scenario, selected):
    _assert_bitwise_equal(
        lambda: build_rtdnoise_scenario(scenario, use_selected_backend=selected),
        _RTDNOISE_ARRAYS)


@pytest.mark.parametrize("scenario", LIVE_ARBITRARY_SYSTEM_SCENARIOS)
def test_arbitrary_system_records_reproduce_the_loops(scenario):
    _assert_bitwise_equal(
        lambda: build_rtdnoise_arbitrary_system_scenario(scenario),
        _RTDNOISE_ARRAYS)
