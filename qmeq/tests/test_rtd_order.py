"""Perturbative-order selection for the population-space RTD approaches."""

import numpy as np
import pytest

import qmeq


RTD_KERNTYPES = ["pyRTD", "RTD", "pyRTDnoise", "RTDnoise"]

_SINGLE_LEVEL = dict(
    nsingle=1,
    hsingle={(0, 0): 0.2},
    nleads=2,
    tleads={
        (0, 0): np.sqrt(0.08/(2*np.pi)),
        (1, 0): np.sqrt(0.12/(2*np.pi)),
    },
    mulst={0: 2.0, 1: -2.0},
    tlst={0: 1.0, 1: 1.0},
    dband=100.0,
)

# Interacting, multi-level, cross-coupled and (for the second) thermally biased,
# so the sequential equivalence below is not a one-state coincidence.
_TWO_LEVEL = dict(
    nsingle=2,
    hsingle={(0, 0): -0.35, (1, 1): 0.55},
    coulomb={(0, 1, 1, 0): 2.4},
    nleads=2,
    tleads={(0, 0): 0.08, (0, 1): 0.05, (1, 0): 0.03, (1, 1): 0.07},
    mulst={0: 1.3, 1: -0.7},
    tlst={0: 0.9, 1: 0.9},
    dband=800.0,
)

_THREE_LEVEL = dict(
    nsingle=3,
    hsingle={(0, 0): -0.5, (1, 1): 0.2, (2, 2): 0.75},
    coulomb={(0, 1, 1, 0): 1.8, (1, 2, 2, 1): 1.1, (0, 2, 2, 0): 0.9},
    nleads=2,
    tleads={(0, 0): 0.06, (0, 1): 0.04, (0, 2): 0.02,
            (1, 0): 0.03, (1, 1): 0.05, (1, 2): 0.07},
    mulst={0: 0.9, 1: -0.9},
    tlst={0: 1.1, 1: 0.6},
    dband=900.0,
)

SEQUENTIAL_MODELS = {
    "single_level": _SINGLE_LEVEL,
    "two_level_interacting": _TWO_LEVEL,
    "three_level_thermal_bias": _THREE_LEVEL,
}


def _system(kerntype, *, order=2, off_diag_corrections=False, **overrides):
    kwargs = dict(_SINGLE_LEVEL, kerntype=kerntype,
                  off_diag_corrections=off_diag_corrections)
    kwargs.update(overrides)
    if "RTDnoise" in kerntype:
        kwargs.setdefault("countingleads", (0,))
    system = qmeq.Builder(**kwargs)
    system.appr.rtd_order = order
    return system


@pytest.mark.parametrize(
    "kerntype", ["Pauli", "Lindblad", "Redfield", "1vN", "2vN"]
)
def test_rtd_order_exists_only_on_rtd_approaches(kerntype):
    """A fixed-order approach must not appear to offer a truncation choice."""
    system = qmeq.Builder(kerntype=kerntype)

    assert not hasattr(system.appr, "rtd_order")
    assert not hasattr(system, "rtd_order")


@pytest.mark.parametrize("kerntype", RTD_KERNTYPES)
def test_rtd_order_defaults_to_the_highest_implemented_order(kerntype):
    """The default reproduces the behaviour that predates the selector."""
    kwargs = dict(_SINGLE_LEVEL, kerntype=kerntype)
    if "RTDnoise" in kerntype:
        kwargs["countingleads"] = (0,)
    approach = qmeq.Builder(**kwargs).appr

    assert approach.rtd_order == 2
    assert approach.rtd_order == approach.rtd_max_order


@pytest.mark.parametrize("kerntype", RTD_KERNTYPES)
def test_rtd_order_rejects_unimplemented_and_noninteger_values(kerntype):
    approach = _system(kerntype).appr

    for value in (True, False, 1.0, "1", None):
        with pytest.raises(TypeError):
            approach.rtd_order = value
    for value in (-1, 0, approach.rtd_max_order + 1):
        with pytest.raises(ValueError):
            approach.rtd_order = value
    assert approach.rtd_order == 2


@pytest.mark.parametrize("kerntype", ["pyRTD", "RTD"])
@pytest.mark.parametrize(
    "model", SEQUENTIAL_MODELS.values(), ids=list(SEQUENTIAL_MODELS)
)
def test_order_one_matches_independent_sequential_averages(kerntype, model):
    """Order 1 is the golden-rule kernel, so Pauli is an independent check.

    Every average the two approaches share must agree: the stationary
    populations, and the particle, energy and heat currents at both leads.
    Pauli carries no noise, so the counting cumulants are out of scope here.

    ``itype=1`` gives Pauli the wide-band principal-value convention that RTD
    forces on itself, which is what makes the two comparable at all. The
    interacting multi-level models keep this from being a single-state
    coincidence, and one of them runs at a thermal bias, which order 1 accepts.
    """
    rtd = _system(kerntype, order=1, **model)
    pauli = qmeq.Builder(**model, kerntype="Pauli", itype=1)

    rtd.solve()
    pauli.solve()

    for name in ("phi0", "current", "energy_current", "heat_current"):
        np.testing.assert_allclose(
            getattr(rtd, name), getattr(pauli, name),
            rtol=0.0, atol=1e-14,
            err_msg=f"order-1 RTD and Pauli disagree on {name}",
        )


@pytest.mark.parametrize("kerntype", ["pyRTD", "RTD"])
def test_order_two_adds_physics_that_order_one_does_not_carry(kerntype):
    """The selector must change the answer, not merely the code path."""
    first = _system(kerntype, order=1)
    second = _system(kerntype, order=2)
    first.solve()
    second.solve()

    assert not np.allclose(first.current, second.current)


@pytest.mark.parametrize("kerntype", RTD_KERNTYPES)
def test_order_one_omits_the_energy_current_corrections(kerntype):
    """``WE1`` and ``WE2`` are two contractions of one O(Gamma^2) correction.

    Neither is a first-order block despite the method names, so order 1 leaves
    both empty and takes the whole energy current from the ``Wdd`` contraction.
    """
    system = _system(kerntype, order=1)
    system.solve()

    np.testing.assert_array_equal(system.appr.WE1, 0.0)
    np.testing.assert_array_equal(system.appr.WE2, 0.0)


@pytest.mark.parametrize("kerntype", ["pyRTDnoise", "RTDnoise"])
def test_order_one_selects_the_first_order_counting_results(kerntype):
    system = _system(kerntype, order=1)
    system.solve()

    np.testing.assert_array_equal(
        system.current_noise, system.current_noise_first
    )
    np.testing.assert_array_equal(
        system.current_noise_matrix, system.current_noise_matrix_first
    )


@pytest.mark.parametrize("kerntype", ["pyRTD", "RTD"])
def test_changing_rtd_order_restarts_the_approach(kerntype):
    system = _system(kerntype, order=2)
    system.solve()
    second_order_current = system.current.copy()
    assert system.appr.is_prepared

    system.appr.rtd_order = 1
    assert not system.appr.is_prepared

    system.solve(qdq=False, rotateq=False)
    assert system.appr.is_prepared
    assert not np.allclose(system.current, second_order_current)

    # Assigning the value it already holds must not discard a solved kernel.
    system.appr.rtd_order = 1
    assert system.appr.is_prepared


@pytest.mark.parametrize("order", [1, 2])
def test_both_backends_agree_at_each_order(order):
    """The compiled twin re-declares the property; it must not drift."""
    python = _system("pyRTD", order=order)
    compiled = _system("RTD", order=order)
    python.solve()
    compiled.solve()

    np.testing.assert_array_equal(python.phi0, compiled.phi0)
    np.testing.assert_array_equal(python.current, compiled.current)


@pytest.mark.parametrize("kerntype", ["pyRTD", "RTD"])
def test_order_one_accepts_a_thermal_bias(kerntype):
    """Only the four-vertex integrals need equal lead temperatures.

    The two-vertex rate reads its own lead's Fermi function, so order 1 is
    defined at unequal temperatures and still reproduces the golden rule.
    """
    unequal = {"tlst": {0: 1.0, 1: 0.4}}
    rtd = _system(kerntype, order=1, **unequal)
    pauli = qmeq.Builder(
        **{**_SINGLE_LEVEL, **unequal}, kerntype="Pauli", itype=1
    )

    rtd.solve()
    pauli.solve()

    np.testing.assert_allclose(rtd.current, pauli.current, rtol=0.0, atol=1e-15)


@pytest.mark.parametrize("kerntype", RTD_KERNTYPES)
def test_coherence_elimination_stays_orthogonal_to_the_order(kerntype):
    """``off_diag_corrections`` is a separate switch, not part of the order.

    It is an O(Gamma^2) term, so enabling it at order 1 is a diagnostic control
    rather than a consistent truncation -- but it must still be the term that
    order 2 adds, and it must still reach the kernel.
    """
    coupled = dict(
        nsingle=2,
        hsingle={(0, 0): -0.4, (1, 1): 0.6, (0, 1): 0.15},
        coulomb={(0, 1, 1, 0): 2.0},
        nleads=2,
        tleads={(0, 0): 0.08, (0, 1): 0.05, (1, 0): 0.03, (1, 1): 0.07},
        mulst={0: 1.5, 1: -1.5},
        tlst={0: 1.0, 1: 1.0},
        dband=1000.0,
    )
    corrected = _system(kerntype, order=1, off_diag_corrections=True, **coupled)
    population_only = _system(
        kerntype, order=1, off_diag_corrections=False, **coupled
    )
    corrected.solve()
    population_only.solve()

    assert not np.allclose(corrected.current, population_only.current)
