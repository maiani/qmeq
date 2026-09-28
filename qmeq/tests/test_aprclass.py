from types import SimpleNamespace

import numpy as np
import pytest

import qmeq
from qmeq.approach.base.neumann2 import Approach2vN
from qmeq.approach.base.neumann2 import get_dm0_transpose_index
from qmeq.approach.base.neumann2 import get_htransf_phi1k
from qmeq.specfunc.specfunc import hilbert_fredriksen


def test_get_htransf_phi1k_matches_scalar_transforms():
    rng = np.random.default_rng(3821)
    phi1k = (
        rng.standard_normal((5, 3, 7, 7))
        + 1j*rng.standard_normal((5, 3, 7, 7))
    )
    original = phi1k.copy()
    funcp = SimpleNamespace(kpnt_left=2, kpnt_right=1, ht_ker=None)

    padded, transformed = get_htransf_phi1k(phi1k, funcp)
    expected = np.empty_like(transformed)
    for index in np.ndindex(padded.shape[1:]):
        trace = (slice(None),) + index
        expected[trace] = hilbert_fredriksen(
            padded[trace], funcp.ht_ker
        )

    assert np.array_equal(phi1k, original)
    # The batched transform and the per-slice reference loop are the same
    # computation, so they agree exactly on x86-64. They are only required to
    # agree to floating-point roundoff, because the FFT pair underneath may
    # associate operations differently between a batched and a 1-D call: on
    # linux-aarch64 this shows up as a last-bit disagreement. The tolerance is
    # therefore set a few orders of magnitude above the ~1e-16 relative scale
    # of a double-precision ULP, and deliberately nowhere near loose enough to
    # accept a genuine error in the batching itself.
    np.testing.assert_allclose(transformed, expected, rtol=1e-13, atol=1e-15)
    assert len(funcp.ht_ker) == 2*len(padded)


def test_Approach2vN_kpnt():
    system = qmeq.Builder(nleads=1, dband={0: 1000}, kpnt=5, kerntype='2vN')
    appr = Approach2vN(system)
    appr.make_Ek_grid()
    assert appr.Ek_grid.tolist() == [-1000, -500, 0, 500, 1000]
    appr.funcp.kpnt = 6
    appr.make_Ek_grid()
    assert appr.Ek_grid.tolist() == [-1000, -600,  -200, 200, 600, 1000]
    #
    system = qmeq.Builder(1, {}, {}, 1, {}, {}, {}, {0: 1000}, kpnt=5, kerntype='2vN')
    system.appr.make_Ek_grid()
    assert system.appr.Ek_grid.tolist() == [-1000, -500, 0, 500, 1000]
    system.kpnt = 6
    system.appr.make_Ek_grid()
    assert system.appr.Ek_grid.tolist() == [-1000, -600,  -200, 200, 600, 1000]


def test_Approach2vN_make_Ek_grid():
    system = qmeq.Builder(nleads=2, dband={0: [-1000, 1000], 1: [-1000, 1000]}, kpnt=5, kerntype='2vN')
    appr = Approach2vN(system)
    appr.make_Ek_grid()
    assert appr.Ek_grid.tolist() == [-1000, -500, 0, 500, 1000]
    appr.leads.change(dlst={0: [-1400, 1000], 1: [-1000, 1000]})
    with pytest.warns(qmeq.QmeqWarning, match="bandwidth and Ek_grid"):
        appr.make_Ek_grid()
    assert appr.Ek_grid.tolist() == [-1400.0, -800.0, -200.0, 400.0, 1000.0]


def test_Approach2vN_warns_when_changed_grid_restarts():
    system = qmeq.Builder(
        nleads=1, dband={0: 1000}, kpnt=5, kerntype="2vN"
    )
    appr = Approach2vN(system)
    appr.make_Ek_grid()
    appr.niter = 0
    appr.funcp.kpnt = 6

    with pytest.warns(qmeq.QmeqWarning, match="Restarting"):
        appr.make_Ek_grid()


_REPHASING_LEADS = dict(mulst={0: 0.6, 1: -0.4}, tlst={0: 0.5, 1: 0.3},
                        dband={0: 10.0, 1: 10.0})


def _double_dot_many_body_input():
    """Eigenbasis of a spinless double dot whose leads both reach both dots.

    The complex hopping and tunnelling phase make the stationary same-charge
    coherence complex, which is what the rephasing test below needs.
    """
    system = qmeq.Builder(
        nsingle=2,
        hsingle={(0, 0): -0.1, (1, 1): 0.25, (0, 1): 0.15 + 0.05j},
        coulomb={(0, 1, 1, 0): 2.0},
        nleads=2,
        tleads={(0, 0): 0.3, (0, 1): 0.15*np.exp(0.9j),
                (1, 1): 0.25, (1, 0): 0.1},
        kerntype='Pauli', **_REPHASING_LEADS,
    )
    system.solve(masterq=False)
    return np.array(system.qd.Ea), [0, 1, 1, 2], np.array(system.leads.Tba)


def test_get_dm0_transpose_index_is_the_orientation_swap():
    Ea, Na, Tba = _double_dot_many_body_input()
    system = qmeq.BuilderManyBody(Ea=Ea, Na=Na, Tba=Tba, kpnt=8,
                                  kerntype='2vN', **_REPHASING_LEADS)
    si = system.si
    transpose = get_dm0_transpose_index(si)
    for charge in range(si.ncharge):
        for b in si.statesdm[charge]:
            for bp in si.statesdm[charge]:
                assert (transpose[si.get_ind_dm0(b, bp, charge)]
                        == si.get_ind_dm0(bp, b, charge))
    assert np.array_equal(transpose[transpose], np.arange(si.ndm0))


@pytest.mark.parametrize("kerntype", ["2vN", "py2vN"])
def test_2vN_is_covariant_under_eigenstate_rephasing(kerntype):
    """A rephasing |b> -> exp(i theta_b)|b> of the many-body eigenbasis maps
    Tba -> exp(-i theta_b) Tba exp(i theta_a) and
    Phi[0]_{bb'} -> exp(-i theta_b) Phi[0]_{bb'} exp(i theta_b'); the currents
    must not change. The first iteration (local approximation) contains no
    conjugated Phi[1] and is covariant on its own; later iterations are
    covariant only if the conjugated Phi[1] acts on the transposed Phi[0]
    element (get_dm0_transpose_index).
    """
    Ea, Na, Tba = _double_dot_many_body_input()
    theta = np.array([0.0, 1.1, -2.3, 0.4])
    phase = np.exp(1j*theta)
    rephased_Tba = phase.conj()[None, :, None]*Tba*phase[None, None, :]

    systems = []
    for amplitudes in (Tba, rephased_Tba):
        system = qmeq.BuilderManyBody(
            Ea=Ea.copy(), Na=Na, Tba=amplitudes.copy(), kpnt=64,
            kerntype=kerntype, **_REPHASING_LEADS,
        )
        system.solve(qdq=False, rotateq=False, niter=3)
        systems.append(system)
    reference, rephased = systems

    si = reference.si
    coherence = reference.phi0[si.get_ind_dm0(1, 2, 1)]
    # Blind if the coherence the transpose acts on were real or zero.
    assert abs(coherence.imag) > 1e-4
    np.testing.assert_allclose(rephased.current, reference.current,
                               rtol=0, atol=1e-13)
    np.testing.assert_allclose(rephased.energy_current,
                               reference.energy_current, rtol=0, atol=1e-13)
    for charge in range(si.ncharge):
        for b in si.statesdm[charge]:
            for bp in si.statesdm[charge]:
                index = si.get_ind_dm0(b, bp, charge)
                covariant = (reference.phi0[index]
                             * np.exp(-1j*theta[b] + 1j*theta[bp]))
                assert abs(rephased.phi0[index] - covariant) < 1e-13
