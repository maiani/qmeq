"""Tests for the Lamb shift of the Lindblad approach."""

import itertools

import numpy as np
from numpy.linalg import norm
from scipy.special import psi
import pytest

import qmeq
from qmeq.specfunc.specfunc import func_1vN
from qmeq.tests.test_builder import (ParametersDoubleDotSpinful,
                                     ParametersSingleOrbitalSpinful)

EPS = 1e-10


def build(p, kerntype='pyLindblad', itype=1, principal_part="digamma"):
    system = qmeq.Builder(p.nsingle, p.hsingle, p.coulomb, p.nleads, p.tleads,
                          p.mulst, p.tlst, p.dlst, kerntype=kerntype, itype=itype,
                          principal_part=principal_part)
    system.solve()
    return system


def total_HLS(system):
    """Lamb shift Hamiltonian summed over the leads."""
    return system.appr.HLS.sum(axis=0)


@pytest.mark.parametrize('kerntype', ['Lindblad', 'pyLindblad'])
def test_HLS_is_hermitian_and_charge_block_diagonal(kerntype):
    """HLS is a Hamiltonian and the total charge is a good quantum number."""
    p = ParametersDoubleDotSpinful()
    system = build(p, kerntype=kerntype, itype=1)
    HLS, si = system.appr.HLS, system.si

    assert norm(HLS - HLS.conjugate().transpose((0, 2, 1))) < EPS
    assert norm(HLS) > 0.0

    charge_of = {b: charge for charge in range(si.ncharge) for b in si.statesdm[charge]}
    for b, bp in itertools.product(charge_of, charge_of):
        if charge_of[b] != charge_of[bp]:
            assert norm(HLS[:, b, bp]) == 0.0


@pytest.mark.parametrize('kerntype', ['Lindblad', 'pyLindblad'])
@pytest.mark.parametrize('itype', [0, 1, 2, 3])
def test_HLS_vanishes_when_disabled(kerntype, itype):
    """The Lamb shift is independent of the legacy integral selector."""
    p = ParametersDoubleDotSpinful()
    system = build(p, kerntype=kerntype, itype=itype, principal_part="omit")
    assert norm(system.appr.HLS) == 0.0


@pytest.mark.parametrize('kerntype', ['Lindblad', 'pyLindblad'])
def test_lamb_shift_changes_the_current(kerntype):
    """Changing only principal_part isolates the Lamb-shift contribution."""
    p = ParametersDoubleDotSpinful()
    with_ls = build(p, kerntype=kerntype, itype=1,
                    principal_part="digamma")
    without_ls = build(p, kerntype=kerntype, itype=1,
                       principal_part="omit")

    assert norm(with_ls.current - without_ls.current) > 1e-4
    # Current conservation is not affected by the Lamb shift
    assert abs(with_ls.current.sum()) < EPS


@pytest.mark.parametrize('kerntype', ['Lindblad', 'pyLindblad'])
@pytest.mark.parametrize(('old_itype', 'equivalent_itype'), [(0, 2), (1, 3)])
def test_legacy_itype_keeps_pre_lamb_shift_results(
        kerntype, old_itype, equivalent_itype):
    """Legacy itype calls do not opt into the newly implemented Lamb shift."""
    legacy = build(
        ParametersDoubleDotSpinful(), kerntype=kerntype, itype=old_itype,
        principal_part="omit"
    )
    equivalent = build(
        ParametersDoubleDotSpinful(), kerntype=kerntype,
        itype=equivalent_itype, principal_part="omit"
    )

    assert norm(legacy.appr.HLS) == 0.0
    assert norm(legacy.current - equivalent.current) < EPS
    assert norm(legacy.energy_current - equivalent.energy_current) < EPS


@pytest.mark.parametrize('kerntype', ['Lindblad', 'pyLindblad'])
@pytest.mark.parametrize('itype', [0, 1, 2, 3])
def test_lamb_shift_kernel_preserves_the_trace(kerntype, itype):
    """The Lamb shift enters through a commutator with a Hermitian Hamiltonian, so it
       must not contribute to the sum of the diagonal rows of the kernel. This fixes the
       relative sign of the two Lamb shift terms of the kernel."""
    p = ParametersDoubleDotSpinful()
    system = qmeq.Builder(p.nsingle, p.hsingle, p.coulomb, p.nleads, p.tleads,
                          p.mulst, p.tlst, p.dlst, kerntype=kerntype, itype=itype,
                          principal_part="digamma")
    # Diagonalise and rotate, but generate the kernel without solving, because the
    # compiled solver overwrites the kernel with its factorisation.
    system.solve(masterq=False)
    appr = system.appr
    appr.prepare_kern()
    appr.generate_fct()
    appr.generate_kern()

    # norm_vec picks the rows of the diagonal density matrix elements
    trace_of_columns = appr.norm_vec @ appr.kern
    assert norm(trace_of_columns) < EPS*np.abs(appr.kern).max()


def test_lamb_shift_backends_agree():
    """The compiled and the pure python Lindblad kernels give the same currents."""
    p = ParametersDoubleDotSpinful()
    for itype in [0, 1, 2, 3]:
        compiled = build(p, kerntype='Lindblad', itype=itype,
                         principal_part="digamma")
        pure = build(p, kerntype='pyLindblad', itype=itype,
                     principal_part="digamma")
        assert norm(compiled.appr.HLS - pure.appr.HLS) < EPS
        assert norm(compiled.current - pure.current) < EPS
        assert norm(compiled.energy_current - pure.energy_current) < EPS


@pytest.mark.parametrize('kerntype', ['Lindblad', 'pyLindblad'])
def test_lamb_shift_with_other_solution_methods(kerntype):
    """The Lamb shift is inserted through the kernel handler, so it has to give the same
       stationary state for the matrix free and the least-squares solution methods."""
    p = ParametersDoubleDotSpinful()
    reference = build(p, kerntype=kerntype, itype=1)

    lsqr = qmeq.Builder(p.nsingle, p.hsingle, p.coulomb, p.nleads, p.tleads,
                        p.mulst, p.tlst, p.dlst, kerntype=kerntype, itype=1,
                        principal_part="digamma", symq=False)
    lsqr.solve()
    assert norm(lsqr.current - reference.current) < EPS

    mfree = qmeq.Builder(p.nsingle, p.hsingle, p.coulomb, p.nleads, p.tleads,
                        p.mulst, p.tlst, p.dlst, kerntype=kerntype, itype=1,
                        principal_part="digamma", mfreeq=True)
    mfree.solve()
    assert norm(mfree.current - reference.current) < 1e-6


@pytest.mark.parametrize('kerntype', ['Lindblad', 'pyLindblad'])
def test_HLS_single_spinless_level(kerntype):
    """For a single spinless level the Lamb shift is known analytically. Both charge
       states are shifted by the same amount, so the transition energy is not
       renormalised in the wide band limit and the current stays the same."""
    t, eps, mulst, tlst, dband = 0.3, 1.5, [2.0, -2.0], [0.4, 0.4], [100.0, 100.0]
    system = qmeq.Builder(1, {(0, 0): eps}, {}, 2, {(0, 0): t, (1, 0): t},
                          mulst, tlst, dband, kerntype=kerntype, itype=1,
                          principal_part="digamma")
    system.solve()

    HLS = system.appr.HLS
    for l in range(2):
        expected = t**2*psi(0.5 + 1.0j*(eps-mulst[l])/(2*np.pi*tlst[l])).real
        # the empty state is shifted by the sum over the states with one electron
        # more, the occupied state by the sum over the states with one electron less
        assert HLS[l, 0, 0].real == pytest.approx(expected, abs=EPS)
        assert HLS[l, 1, 1].real == pytest.approx(expected, abs=EPS)

    without_ls = qmeq.Builder(1, {(0, 0): eps}, {}, 2, {(0, 0): t, (1, 0): t},
                              mulst, tlst, dband, kerntype=kerntype, itype=1,
                              principal_part="omit")
    without_ls.solve()
    assert norm(system.current - without_ls.current) < EPS


@pytest.mark.parametrize('kerntype', ['Lindblad', 'pyLindblad'])
def test_HLS_spin_degenerate_orbital(kerntype):
    """A spin degenerate orbital is shifted uniformly within each charge sector, hence
       the Lamb shift commutes with the density matrix and the current is unchanged."""
    p = ParametersSingleOrbitalSpinful()
    with_ls = build(p, kerntype=kerntype, itype=1)
    without_ls = build(p, kerntype=kerntype, itype=1,
                       principal_part="omit")

    HLS, si = total_HLS(with_ls), with_ls.si
    for charge in range(si.ncharge):
        block = HLS[np.ix_(si.statesdm[charge], si.statesdm[charge])]
        shifts = np.diag(block).real
        assert norm(block - np.diag(shifts)) < EPS
        assert norm(shifts - shifts[0]) < EPS

    assert norm(with_ls.current - without_ls.current) < EPS


@pytest.mark.parametrize('kerntype', ['Lindblad', 'pyLindblad'])
def test_HLS_level_renormalisation_matches_1vN_principal_part(kerntype):
    """The energy renormalisation of the Lamb shift has to agree with the principal
       parts of the 1vN and Redfield kernels, which are the same integrals in the
       same sign convention. Only the differences of the diagonal entries are
       compared, because the bandwidth constant dropped by func_lambshift shifts all
       states of a charge sector by the same amount."""
    p = ParametersDoubleDotSpinful()
    system = build(p, kerntype=kerntype, itype=1)

    si, appr = system.si, system.appr
    E, Tba = appr.qd.Ea, appr.leads.Tba
    mulst, tlst, dlst = appr.leads.mulst, appr.leads.tlst, appr.leads.dlst
    HLS = total_HLS(system)

    for bcharge in range(si.ncharge):
        acharge, ccharge = bcharge-1, bcharge+1
        shifts_1vN = {}
        for b in si.statesdm[bcharge]:
            shift = 0.0
            for l in range(si.nleads):
                args = (mulst[l], tlst[l], dlst[l, 0], dlst[l, 1], 1, 10000)
                for a in si.statesdm[acharge]:
                    # hole factor, the intermediate state has one electron less
                    fct = func_1vN(E[b]-E[a], *args)[1]
                    shift += (Tba[l, b, a]*Tba[l, a, b]*fct).real
                for c in si.statesdm[ccharge]:
                    # particle factor, the intermediate state has one electron more
                    fct = func_1vN(E[c]-E[b], *args)[0]
                    shift += (Tba[l, b, c]*Tba[l, c, b]*fct).real
            shifts_1vN[b] = shift

        for b, bp in itertools.combinations(si.statesdm[bcharge], 2):
            assert ((HLS[b, b]-HLS[bp, bp]).real
                    == pytest.approx(shifts_1vN[b]-shifts_1vN[bp], abs=1e-8))


@pytest.mark.parametrize('kerntype', ['Lindblad', 'pyLindblad'])
def test_lamb_shift_of_elph_lindblad(kerntype):
    """The electron-phonon Lindblad approach reuses the electron-lead kernel and
       therefore also picks up the Lamb shift of the leads."""
    from qmeq.tests.test_builder_elph import SpinfulDoubleDotWithElPh

    with_ls = SpinfulDoubleDotWithElPh(
        kerntype=kerntype, itype=1, itype_ph=2, vbias=0.5,
        principal_part="digamma"
    )
    with_ls.solve()
    without_ls = SpinfulDoubleDotWithElPh(
        kerntype=kerntype, itype=1, itype_ph=2, vbias=0.5,
        principal_part="omit"
    )
    without_ls.solve()

    assert norm(with_ls.appr.HLS) > 0.0
    assert norm(without_ls.appr.HLS) == 0.0
    assert with_ls.success and without_ls.success
    # The currents of this weakly coupled model are tiny, so compare relative to them
    assert abs(with_ls.current.sum()) < 1e-4*np.abs(with_ls.current).max()
    assert norm(with_ls.current - without_ls.current) > 0.0
