"""Structure of the RTD population-diagram records.

The records of :mod:`qmeq.approach.rtd_diagrams` feed both RTD and RTDnoise.
These tests check what a record must satisfy independently of how it is
evaluated: charge conservation of its counting labels, consistent population
endpoints, and the lead its stationary contribution is attributed to. The
compiled RTD traversal is held to the record-based Python path through the
stationary state and current.
"""

import functools
import warnings

import numpy as np
import pytest

from qmeq.approach.rtd_diagrams import DIRECT
from qmeq.approach.rtd_diagrams import EXCHANGE
from qmeq.approach.rtd_diagrams import GAIN
from qmeq.approach.rtd_diagrams import LOSS
from qmeq.approach.rtd_diagrams import counting_labels
from qmeq.approach.rtd_diagrams import first_order_diagrams
from qmeq.approach.rtd_diagrams import second_order_diagrams
from qmeq.tests.qmeq_11_reference_models import (
    RTD_REFERENCE_SCENARIOS,
    build_rtd_reference_system,
)
from qmeq.tests.rtdnoise_reference_models import (
    LIVE_ARBITRARY_SYSTEM_SCENARIOS,
    build_rtdnoise_arbitrary_system_scenario,
)


@functools.cache
def _solved(scenario):
    system = build_rtdnoise_arbitrary_system_scenario(scenario)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        system.solve()
    return system.appr


def _populations(appr):
    """Yield ``(state, charge)`` for each population the kernel carries."""
    si, kh = appr.si, appr.kernel_handler
    for charge, sector in enumerate(si.statesdm):
        for state in sector:
            if kh.is_unique(state, state, charge):
                yield state, charge


def _charge_of_population(appr):
    si = appr.si
    return {
        si.get_ind_dm0(b, b, charge): charge
        for charge, sector in enumerate(si.statesdm) for b in sector
    }


@pytest.mark.parametrize("scenario", LIVE_ARBITRARY_SYSTEM_SCENARIOS)
def test_counting_labels_conserve_charge(scenario):
    """At every insertion the charge counted in the leads enters the dot.

    The labels are the vertex factors of the two contractions; their sum
    must equal the charge of the row population minus that of the column,
    for each of the four outer-Keldysh insertions of every diagram. This is
    charge conservation, not a restatement of the label formula.
    """
    appr = _solved(scenario)
    seen = 0
    for a0, charge in _populations(appr):
        for d in second_order_diagrams(appr, a0, charge):
            labels = counting_labels(d.topology, 1, d.eta1, d.p1, d.p2)
            rows = (d.final_charge, d.flipped_charge,
                    d.final_charge, d.flipped_charge)
            columns = (charge, charge, charge + 1, charge + 1)
            for (q0, q1), row, column in zip(labels, rows, columns):
                assert q0 in (-1, 0, 1) and q1 in (-1, 0, 1)
                assert q0 + q1 == row - column, d
                seen += 1
    assert seen > 0


@pytest.mark.parametrize("scenario", LIVE_ARBITRARY_SYSTEM_SCENARIOS)
def test_second_order_records_have_population_endpoints(scenario):
    appr = _solved(scenario)
    si = appr.si
    charge_of = _charge_of_population(appr)
    topologies = set()
    for a0, charge in _populations(appr):
        for d in second_order_diagrams(appr, a0, charge):
            topologies.add(d.topology)
            a1 = d.states[1]
            assert d.states[0] == a0
            assert d.initial == si.get_ind_dm0(a0, a0, charge)
            assert d.initial_flipped == si.get_ind_dm0(a1, a1, charge + 1)
            assert charge_of[d.initial_flipped] == charge + 1
            for state, state_charge in ((d.final_state, d.final_charge),
                                        (d.flipped_state, d.flipped_charge)):
                assert state in si.statesdm[state_charge]
                assert abs(state_charge - charge) <= 2
            assert d.p1 in (-1, 1) and d.p2 in (-1, 1) and d.eta1 in (-1, 1)
            assert d.lead == (d.r0 if d.topology == DIRECT else d.r1)
            assert abs(d.tunnel_product) > 1e-20*max(appr.leads.tlst)**2
    assert topologies == {DIRECT, EXCHANGE}


@pytest.mark.parametrize("scenario", LIVE_ARBITRARY_SYSTEM_SCENARIOS)
def test_first_order_records_pair_every_gain_with_a_loss(scenario):
    """Each neighbouring state and lead gives one gain and one loss diagram.

    A gain diagram's column is the neighbour's population, one charge away;
    a loss diagram stays on the row. Both name the same transition, and
    ``gamma`` is the squared modulus of its amplitude.
    """
    appr = _solved(scenario)
    si, E, Tba = appr.si, appr.qd.Ea, appr.leads.Tba
    charge_of = _charge_of_population(appr)
    for b, charge in _populations(appr):
        records = list(first_order_diagrams(appr, b, charge))
        gains = [d for d in records if d.kind == GAIN]
        losses = [d for d in records if d.kind == LOSS]
        key = lambda d: (d.other, d.lead)  # noqa: E731
        assert sorted(map(key, gains)) == sorted(map(key, losses))
        for d in records:
            other_charge = charge - 1 if d.lower else charge + 1
            assert d.other in si.statesdm[other_charge]
            upper, lower = (b, d.other) if d.lower else (d.other, b)
            assert d.pair == si.get_ind_dm1(upper, lower, min(charge, other_charge))
            assert d.energy == E[upper] - E[lower]
            np.testing.assert_allclose(
                d.gamma, abs(Tba[d.lead, d.other, b])**2, rtol=1e-12, atol=0)
            if d.kind == GAIN:
                assert charge_of[d.column] == other_charge
            else:
                assert d.column == d.row


@pytest.mark.parametrize("scenario", RTD_REFERENCE_SCENARIOS)
def test_compiled_rtd_matches_the_record_based_python_rtd(scenario):
    """The compiled traversal and the records give one stationary answer.

    Under ``QMEQ_BACKEND=python`` both names resolve to the Python approach
    and the comparison is trivial; under the compiled backend it holds the
    hand-written Cython loops to the record stream.
    """
    options = {"qdq": False, "rotateq": False} if scenario == "many_body" else {}
    results = []
    for selected in (False, True):
        system = build_rtd_reference_system(scenario, use_selected_backend=selected)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            system.solve(**options)
        results.append(system)
    python, selected = results
    np.testing.assert_allclose(selected.phi0, python.phi0, rtol=0, atol=1e-13)
    scale = max(np.max(np.abs(python.current)), np.finfo(float).tiny)
    np.testing.assert_allclose(
        selected.current, python.current, rtol=0, atol=1e-12*scale + 1e-18)
