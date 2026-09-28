"""Tests for inputs at the edge of what the approaches can evaluate.

These are the cases where an approach has to refuse rather than return a
number, plus the neighbouring cases it must still accept. Both live here so
that tightening one does not silently narrow the other.
"""

import warnings

import numpy as np
import pytest

import qmeq

APPROACHES = ["Pauli", "Lindblad", "Redfield", "1vN", "RTD"]


def _system(temperature=0.1, dband=1e5, level=-1.0, kerntype="1vN", itype=1):
    """A spinful double dot small enough to solve in every approach."""
    tunnel = 0.02
    return qmeq.Builder(
        nsingle=2,
        hsingle={(0, 0): level, (1, 1): -0.6},
        coulomb={(0, 1, 1, 0): 2.0},
        nleads=2,
        tleads={
            (0, 0): tunnel, (0, 1): tunnel,
            (1, 0): tunnel, (1, 1): tunnel,
        },
        mulst={0: 0.5, 1: -0.5},
        tlst={0: temperature, 1: temperature},
        dband=dband,
        kerntype=kerntype,
        itype=itype,
        principal_part="digamma" if kerntype == "Lindblad" else None,
    )


# --- lead temperatures -----------------------------------------------------

@pytest.mark.parametrize("temperature", [0.0, -0.1, float("nan")])
def test_non_positive_lead_temperature_is_refused(temperature):
    """Zero, negative, and nan temperatures are rejected at construction.

    Every kernel divides by the temperature. Left to the approaches the three
    disagree about how to fail: Pauli and Lindblad return ``success=False``
    from a singular kernel, Redfield, 1vN and RTD report ``success=True``
    beside an all-``nan`` stationary solution, and a negative temperature is
    accepted outright and returns a confidently wrong current. Refusing the
    input is the only answer that is the same in every approach.
    """
    with pytest.raises(ValueError, match="lead temperatures must be positive"):
        _system(temperature=temperature)


@pytest.mark.parametrize("kerntype", APPROACHES)
def test_small_positive_lead_temperature_is_accepted(kerntype):
    """The refusal above must not creep towards ordinary temperatures."""
    system = _system(temperature=1e-8, kerntype=kerntype)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        system.solve()
    assert np.all(np.isfinite(system.current))


@pytest.mark.parametrize(
    "update",
    [
        pytest.param(lambda system: system.change(tlst={0: 0.0}), id="change"),
        pytest.param(lambda system: system.add(tlst={0: -1.0}), id="add"),
    ],
)
def test_rejected_temperature_update_leaves_the_leads_untouched(update):
    """A refused update must not have half-applied itself.

    ``make_array`` writes a dictionary update straight into the array it is
    handed, so validating the requested value is not enough on its own: the
    candidate has to be built from a copy.
    """
    system = _system()
    before = np.array(system.leads.tlst)
    with pytest.raises(ValueError, match="lead temperatures must be positive"):
        update(system)
    np.testing.assert_array_equal(system.leads.tlst, before)


def test_partial_temperature_dict_cannot_leave_leads_at_zero():
    """A tlst naming some leads must not leave the others at the zero default."""
    with pytest.raises(ValueError, match=r"tlst\[1\]=0.0 \(not given\)"):
        qmeq.Builder(
            nsingle=1, hsingle={(0, 0): 0.0}, nleads=2,
            tleads={(0, 0): 0.1, (1, 0): 0.1}, mulst={0: 0.1, 1: -0.1},
            tlst={0: 1.0}, dband={0: 10.0, 1: 10.0},
        )
    # An empty tlst is still accepted, for systems that never reach a solve.
    qmeq.Builder(nsingle=1, hsingle={(0, 0): 0.0}, nleads=2)


def test_assigned_temperatures_are_validated_and_leave_the_leads_untouched():
    system = _system()
    before = np.array(system.leads.tlst)
    with pytest.raises(ValueError, match="lead temperatures must be positive"):
        system.tlst = [0.0, 0.1]
    with pytest.raises(ValueError, match="needs shape"):
        system.tlst = [0.1, 0.1, 0.1]
    np.testing.assert_array_equal(system.leads.tlst, before)


# --- assigned lead parameters ------------------------------------------------

def _single_level(kerntype):
    return qmeq.Builder(
        nsingle=1, hsingle={(0, 0): 0.1}, nleads=2,
        tleads={(0, 0): 0.1, (1, 0): 0.1}, mulst={0: 0.3, 1: -0.3},
        tlst={0: 1.0, 1: 1.0}, dband={0: 50.0, 1: 50.0}, kerntype=kerntype,
    )


@pytest.mark.parametrize("kerntype", ["Pauli", "pyPauli", "1vN", "py1vN"])
@pytest.mark.parametrize(
    ("attribute", "value"),
    [
        ("mulst", [0.6, -0.6]),
        ("tlst", [0.5, 0.5]),
        ("dlst", [[-50.0, 50.0], [-0.05, 0.05]]),
    ],
)
def test_assigned_lead_parameters_reach_a_prepared_approach(
        kerntype, attribute, value):
    """Assigning a lead array must act like change() after the first solve.

    The compiled approaches bind views of these arrays when they prepare, so
    replacing the array left them computing with the old values -- in QmeQ
    1.1 as well -- while the Python approaches, which read the attribute
    afresh, followed the assignment.
    """
    assigned = _single_level(kerntype)
    assigned.solve()
    setattr(assigned, attribute, value)
    assigned.solve()
    changed = _single_level(kerntype)
    changed.leads.change(**{attribute: value})
    changed.solve()

    np.testing.assert_array_equal(assigned.current, changed.current)


@pytest.mark.parametrize("kerntype", ["Pauli", "pyPauli", "1vN"])
def test_assigned_bath_temperatures_reach_a_prepared_approach(kerntype):
    """The phonon bath arrays are bound the same way as the lead arrays."""
    def build():
        return qmeq.BuilderElPh(
            nsingle=2, hsingle={(0, 0): 0.1, (1, 1): -0.1, (0, 1): 0.05},
            coulomb={(0, 1, 1, 0): 3.0}, nleads=2,
            tleads={(0, 0): 0.05, (1, 1): 0.05}, mulst={0: 0.3, 1: -0.3},
            tlst={0: 0.5, 1: 0.5}, dband={0: 50.0, 1: 50.0},
            nbaths=1, velph={(0, 0, 0): 0.3, (0, 1, 1): -0.3},
            tlst_ph={0: 0.2}, dband_ph={0: [1e-8, 10.0]},
            kerntype=kerntype, itype=2, itype_ph=2,
        )

    assigned = build()
    assigned.solve()
    assigned.baths.tlst_ph = [1.0]
    assigned.solve()
    changed = build()
    changed.baths.change(tlst_ph={0: 1.0})
    changed.solve()

    np.testing.assert_array_equal(assigned.current, changed.current)


def test_lead_arrays_keep_the_spin_symmetric_layout():
    """With spin symmetry the stored array is doubled; reading it back and
    assigning it, or add(), must not double it again."""
    # Under spin symmetry nleads counts both spin channels of each lead.
    system = qmeq.Builder(
        nsingle=2, hsingle={(0, 0): 0.1}, nleads=4,
        tleads={(0, 0): 0.1, (1, 0): 0.1}, mulst={0: 0.3, 1: -0.3},
        tlst={0: 1.0, 1: 1.0}, dband={0: 50.0, 1: 50.0},
        symmetry="spin", kerntype="Pauli",
    )
    stored = system.leads.mulst
    assert stored.shape == (4,)

    system.mulst = np.array(stored)
    system.leads.add(mulst={0: 0.1})
    system.mulst = {0: 0.4, 1: -0.3}

    assert system.leads.mulst is stored
    np.testing.assert_array_equal(stored, [0.4, -0.3, 0.4, -0.3])


# --- itype=0 band edges ----------------------------------------------------

@pytest.mark.parametrize("edge", [1.0, 50.0])
def test_itype0_refuses_a_transition_energy_on_the_band_edge(edge):
    """The principal value does not exist when the pole sits on the boundary.

    ``scipy``'s Cauchy weight refuses ``wvar`` on an integration limit, and
    the accompanying ``log|(Rm - alpha)/(Rp - alpha)|`` is ``log(0)`` in the
    same breath, so this is not a hard integral but an undefined one. The
    error must name the input rather than quote scipy: the two band widths
    here show it is the coincidence and not the width that triggers it.
    """
    system = _system(level=-edge, dband=edge, itype=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="exactly on a band edge"):
            system.solve()


def test_itype0_accepts_a_band_edge_next_to_a_transition_energy():
    """Only exact coincidence is refused; a neighbouring cutoff still solves.

    The current is genuinely discontinuous across the edge, so the guard must
    not widen into a tolerance that silently picks one side.
    """
    system = _system(level=-1.0, dband=1.0 + 1e-9, itype=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        system.solve()
    assert np.all(np.isfinite(system.current))
    assert system.current[0] != 0.0
