"""Compiled electron-phonon approaches against their pure-Python twins."""

import numpy as np
import pytest

from qmeq import BuilderElPh
from qmeq.specfunc import Func


class _OhmicBath(Func):
    def eval(self, energy):
        return 0.02*energy


def _hybridised_double_dot(kerntype):
    """Spinless double dot whose one-electron sector carries a coherence.

    The complex off-diagonal phonon element makes the jump operator
    ``tLbbp[l, b, a, q]`` connect different eigenstates, so kernel columns of
    the coherence ``(a, ap)``, ``a != ap``, receive a nonzero phonon term.
    """
    return BuilderElPh(
        nsingle=2, hsingle={(0, 0): 0.5, (1, 1): -0.5, (0, 1): 0.3},
        coulomb={(0, 1, 1, 0): 2.0}, nleads=2,
        tleads={(0, 0): 0.2*np.exp(0.3j), (1, 1): 0.15, (0, 1): 0.05j},
        mulst={0: 1.0, 1: -1.0}, tlst={0: 0.8, 1: 0.5},
        dband={0: 20.0, 1: 20.0}, nbaths=1,
        velph={(0, 0, 0): 0.1, (0, 1, 1): -0.07*np.exp(0.7j),
               (0, 0, 1): 0.05*np.exp(1.1j), (0, 1, 0): 0.03*np.exp(-0.4j)},
        tlst_ph={0: 0.4}, dband_ph={0: [1e-8, 10.0]}, bath_func=[_OhmicBath()],
        kerntype=kerntype, principal_part="omit",
    )


def _spin_dependent_ssq_double_dot(kerntype):
    """Spinful double dot under ``'ssq'`` indexing with a spin-dependent velph.

    Single-particle states 0, 1 carry spin up and 2, 3 spin down. The phonon
    coupling differs between the two spins, so it conserves S_z but not the
    total spin, and ``Vbbp`` couples singlet and triplet states of equal
    charge. The ``si_elph`` layout pairs only states of equal (S, S_z), so
    these jump-operator elements lie outside it. The model is outside the
    validity of ``'ssq'``; the two backends must still compute the same
    approximation.
    """
    return BuilderElPh(
        nsingle=4,
        hsingle={(0, 0): 0.2, (1, 1): -0.1, (0, 1): 0.05,
                 (2, 2): 0.2, (3, 3): -0.1, (2, 3): 0.05},
        coulomb={(0, 2, 2, 0): 3.0, (1, 3, 3, 1): 3.0, (0, 1, 1, 0): 1.0,
                 (2, 3, 3, 2): 1.0, (0, 3, 3, 0): 1.0, (1, 2, 2, 1): 1.0},
        nleads=4, tleads={(0, 0): 0.1, (1, 1): 0.07, (2, 2): 0.1, (3, 3): 0.07},
        mulst={0: 0.5, 1: -0.5, 2: 0.5, 3: -0.5},
        tlst={0: 0.3, 1: 0.3, 2: 0.3, 3: 0.3}, dband=20.0, nbaths=1,
        velph={(0, 0, 0): 0.03, (0, 0, 1): 0.05,
               (0, 2, 2): -0.03, (0, 2, 3): 0.01},
        tlst_ph={0: 0.2}, dband_ph={0: [1e-8, 10.0]}, bath_func=[_OhmicBath()],
        indexing="ssq", kerntype=kerntype, principal_part="omit",
    )


def _generator(system):
    """The unsolved kernel; a compiled solve factorises ``kern`` in place."""
    system.solve(masterq=False)
    appr = system.appr
    appr.prepare_kern()
    appr.generate_fct()
    appr.generate_kern()
    return np.array(appr.kern, copy=True)


@pytest.mark.parametrize("model", [_hybridised_double_dot,
                                   _spin_dependent_ssq_double_dot])
@pytest.mark.parametrize("kerntype", ["Pauli", "Lindblad", "Redfield", "1vN"])
def test_electron_phonon_backend_parity(kerntype, model):
    kerns, solved = {}, {}
    for implementation in (kerntype, "py"+kerntype):
        kerns[implementation] = _generator(model(implementation))
        system = model(implementation)
        system.solve()
        solved[implementation] = system

    np.testing.assert_allclose(kerns[kerntype], kerns["py"+kerntype],
                               rtol=1e-12, atol=1e-14)
    for field in ("phi0", "current", "energy_current"):
        np.testing.assert_allclose(getattr(solved[kerntype], field),
                                   getattr(solved["py"+kerntype], field),
                                   rtol=1e-10, atol=1e-13, err_msg=field)


@pytest.mark.parametrize("kerntype", ["Lindblad", "pyLindblad"])
def test_electron_phonon_lindblad_generator_preserves_trace(kerntype):
    """A GKSL generator conserves the trace: under charge indexing the
    population rows of every kernel column sum to zero, coherence columns
    included, and the leads conserve particles, I_L = -I_R."""
    system = _hybridised_double_dot(kerntype)
    kern = _generator(system)
    npauli = system.si.npauli
    np.testing.assert_allclose(kern[:npauli].sum(axis=0), 0.0, atol=1e-14)

    system = _hybridised_double_dot(kerntype)
    system.solve()
    assert np.all(system.phi0[:npauli] > 0.0)
    np.testing.assert_allclose(system.current.sum(), 0.0, atol=1e-14)
