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
        (lambda: validate_kerntype("unknown"), "Allowed kerntype"),
        (
            lambda: resolve_transport_options(9, None, None, "Pauli"),
            "itype needs to be",
        ),
        (
            lambda: resolve_transport_options(0, None, None, "RTD"),
            "Only itype=1",
        ),
        (lambda: validate_itype_ph(1), "itype_ph needs"),
        (lambda: validate_mfreeq("RTD", True), "mfreeq=True"),
        (
            lambda: validate_indexing(None, "spin", "RTD"),
            "symmetry='spin'",
        ),
        (
            lambda: validate_indexing("invalid", None, "Pauli"),
            "Allowed indexing",
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
    with pytest.warns(qmeq.QmeqWarning, match="indexing has to be"):
        StateIndexing(2, indexing="invalid")

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
