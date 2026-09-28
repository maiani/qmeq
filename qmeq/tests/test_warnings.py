"""Regression tests for QmeQ's structured warning interface."""

import warnings

import numpy as np
import pytest

import qmeq
from qmeq.builder.funcprop import FunctionProperties
from qmeq.builder.validation import resolve_transport_options
from qmeq.builder.validation import validate_indexing
from qmeq.builder.validation import validate_itype_ph
from qmeq.builder.validation import validate_kerntype
from qmeq.builder.validation import validate_mfreeq
from qmeq.approach.base.RTDnoise import ApproachPyRTDnoise
from qmeq.indexing import StateIndexing
from qmeq.qdot import QuantumDot, ssquare_eigenstates


def test_warning_categories_are_public_and_filterable_as_a_group():
    assert issubclass(qmeq.QmeqRuntimeWarning, qmeq.QmeqWarning)
    assert issubclass(qmeq.QmeqRuntimeWarning, RuntimeWarning)


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda: validate_kerntype("Lindbald"), "Unknown kerntype 'Lindbald'"),
        (
            lambda: resolve_transport_options(9, None, None, "Pauli"),
            "itype must be 0, 1, 2, or 3",
        ),
        (lambda: validate_itype_ph(1), "itype_ph must be 0 or 2"),
        (lambda: validate_mfreeq("RTD", True), "mfreeq=True"),
        (
            lambda: validate_indexing("invalid", None, "Pauli"),
            "indexing must be",
        ),
        (lambda: validate_indexing(None, "Spin", "Pauli"), "symmetry must be"),
        (lambda: StateIndexing(2, indexing="invalid"), "indexing must be"),
    ],
    ids=["kerntype", "itype", "itype_ph", "mfreeq", "indexing", "symmetry",
         "StateIndexing"],
)
def test_invalid_input_raises_instead_of_falling_back(call, match):
    """A misspelled option must not silently select a different calculation.

    Each of these used to warn and substitute a default: an unknown kerntype
    ran Pauli, an unknown indexing ran charge or Lin, and a misspelled
    symmetry was ignored. mfreeq with RTD crashed deep in the kernel handler.
    """
    with pytest.raises(ValueError, match=match):
        call()


def test_builder_refuses_invalid_input():
    with pytest.raises(ValueError, match="Unknown kerntype"):
        qmeq.Builder(nsingle=0, kerntype="Lindbald")
    with pytest.raises(ValueError, match="mfreeq=True"):
        qmeq.Builder(nsingle=0, kerntype="RTD", mfreeq=True)
    system = qmeq.Builder(nsingle=0, kerntype="Pauli")
    with pytest.raises(ValueError, match="Unknown kerntype"):
        system.kerntype = "Lindbald"
    with pytest.raises(ValueError, match="itype must be"):
        system.itype = 7
    rtd = qmeq.Builder(nsingle=0, kerntype="RTD")
    with pytest.raises(ValueError, match="mfreeq=True"):
        rtd.mfreeq = True


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (
            lambda: resolve_transport_options(0, None, None, "RTD"),
            "Only itype=1",
        ),
        (
            lambda: validate_indexing(None, "spin", "RTD"),
            "symmetry='spin'",
        ),
        (
            lambda: validate_indexing("sz", None, "2vN"),
            "2vN approach",
        ),
        (
            lambda: validate_indexing("Lin", None, "RTD"),
            "RTD approach",
        ),
    ],
)
def test_validation_fallbacks_emit_qmeq_warning(call, match):
    with pytest.warns(qmeq.QmeqWarning, match=match):
        call()


def test_state_indexing_fallbacks_emit_qmeq_warning():
    with pytest.warns(qmeq.QmeqWarning, match="nsingle has to be even"):
        StateIndexing(3, indexing="sz")

    charge_indexing = StateIndexing(2, indexing="charge")
    with pytest.warns(qmeq.QmeqWarning, match="Returning charge list"):
        charge_indexing.get_lst(charge=1, sz=1)

    spin_indexing = StateIndexing(2, indexing="ssq")
    with pytest.warns(qmeq.QmeqWarning, match="removal of Fock states"):
        spin_indexing.remove_fock_states([0])


def test_quantum_dot_fallbacks_emit_qmeq_warning():
    si = StateIndexing(2)
    with pytest.warns(qmeq.QmeqWarning, match="same parity"):
        assert ssquare_eigenstates(1, 0, si) == 0

    qd = QuantumDot({}, {}, si)
    with pytest.warns(qmeq.QmeqWarning, match="No indexing"):
        assert qd.diagonalise_charge(charge=1, sz=1) is None


def test_function_properties_warnings_are_typed_and_suppressed():
    funcp = FunctionProperties()
    with pytest.warns(qmeq.QmeqWarning, match="missing initial state"):
        funcp.print_warning(0, "missing initial state")
    with pytest.warns(qmeq.QmeqRuntimeWarning, match="solver exploded"):
        funcp.print_error(RuntimeError("solver exploded"))

    # These legacy helpers intentionally report each condition once per object.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        funcp.print_warning(0, "missing initial state")
        funcp.print_error(RuntimeError("solver exploded"))
    assert not caught


def test_builder_runtime_fallbacks_emit_qmeq_warning():
    system = qmeq.Builder(
        nsingle=1,
        nleads=0,
        indexing="Lin",
        symq=False,
        solmethod="solve",
    )
    with pytest.warns(qmeq.QmeqWarning, match="Cannot change indexing"):
        system.indexing = "charge"
    with pytest.warns(qmeq.QmeqWarning, match="solmethod=lsqr"):
        system.solve(qdq=False, rotateq=False)


@pytest.mark.parametrize("solve_method", ["solve_kern_first", "solve_kern_second"])
def test_rtdnoise_solver_failure_warns_without_dumping_kernel(
    solve_method, capsys
):
    approach = object.__new__(ApproachPyRTDnoise)
    approach.funcp = FunctionProperties(solmethod="solve", symq=True)
    approach.kern_first = np.zeros((2, 2))
    approach.kern_second = np.zeros((2, 2))
    approach.bvec = np.zeros(2)
    approach.replaced_eq = np.zeros(2)
    approach.norm_vec = np.ones(2)
    approach.phi0_first = np.zeros(2)
    approach.phi0_second = np.zeros(2)

    with pytest.warns(qmeq.QmeqRuntimeWarning, match="Singular matrix"):
        getattr(approach, solve_method)()

    captured = capsys.readouterr()
    assert captured.out == ""


def _spinful_double_dot(indexing, t_up, t_down, kerntype="1vN", mixing=0.0):
    """Orbitals 0, 1 spin up and 2, 3 spin down; leads 0, 1 (up), 2, 3 (down).

    ``mixing`` lets lead 0 also reach the spin-down orbital 2, which breaks
    S_z conservation of the tunnelling.
    """
    tleads = {(0, 0): t_up, (1, 1): t_up, (2, 2): t_down, (3, 3): t_down}
    if mixing:
        tleads[(0, 2)] = mixing
    return qmeq.Builder(
        nsingle=4,
        hsingle={(0, 0): 0.0, (1, 1): 0.2, (0, 1): 0.1,
                 (2, 2): 0.0, (3, 3): 0.2, (2, 3): 0.1},
        coulomb={(0, 2, 2, 0): 2.0, (1, 3, 3, 1): 2.0, (0, 1, 1, 0): 1.0,
                 (2, 3, 3, 2): 1.0, (0, 3, 3, 0): 1.0, (1, 2, 2, 1): 1.0},
        nleads=4, tleads=tleads,
        mulst={0: 1.0, 1: -1.0, 2: 1.0, 3: -1.0},
        tlst={lead: 0.5 for lead in range(4)},
        dband={lead: 20.0 for lead in range(4)},
        kerntype=kerntype, indexing=indexing,
    )


def _symmetry_warnings(system):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        system.solve()
        system.solve()
    return [w for w in caught if "assumes the leads" in str(w.message)]


@pytest.mark.parametrize(
    ("indexing", "t_down", "mixing", "kerntype", "expected"),
    [
        ("ssq", 0.1, 0.0, "1vN", False),     # SU(2) symmetric
        ("ssq", 0.05, 0.0, "1vN", True),     # spin-dependent tunnelling
        ("ssq", 0.05, 0.0, "Pauli", True),   # ssq reduces multiplets even for Pauli
        ("sz", 0.05, 0.0, "1vN", False),     # S_z is still conserved
        ("sz", 0.1, 0.03, "1vN", True),      # a lead couples to both spins
        ("sz", 0.1, 0.03, "Pauli", False),   # Pauli keeps no coherences
    ],
)
def test_couplings_that_break_the_indexing_symmetry_warn(
        indexing, t_down, mixing, kerntype, expected):
    """'sz'/'ssq' indexing silently drops what symmetry-breaking couplings do.

    With spin-dependent tunnelling 'ssq' moved this current by 8% while
    'sz' and 'charge' agreed to machine precision; nothing said so.
    """
    system = _spinful_double_dot(indexing, 0.1, t_down, kerntype, mixing)
    caught = _symmetry_warnings(system)
    assert len(caught) == (1 if expected else 0)
    if expected:
        assert issubclass(caught[0].category, qmeq.QmeqWarning)


def test_spin_dependent_phonons_break_ssq():
    from qmeq.tests.test_elph_backend_parity import _spin_dependent_ssq_double_dot

    caught = _symmetry_warnings(_spin_dependent_ssq_double_dot("pyLindblad"))
    assert len(caught) == 1
    assert "phonon bath 0" in str(caught[0].message)
