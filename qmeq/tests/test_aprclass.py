from types import SimpleNamespace

import numpy as np

import qmeq
from qmeq.approach.base.neumann2 import Approach2vN
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
    appr.make_Ek_grid()
    assert appr.Ek_grid.tolist() == [-1400.0, -800.0, -200.0, 400.0, 1000.0]
