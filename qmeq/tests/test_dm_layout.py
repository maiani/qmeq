"""Tests pinning the packed real density-matrix layout.

Every test names the rule from :mod:`qmeq.approach.dm_layout` that it pins, so
a reviewer can check a new kernel handler or diagram generator against a
written contract instead of against nested loops. Nothing here builds a model
or solves a master equation: the layout is a property of the indexing, and it
is tested as one.
"""

import itertools
import os

import numpy as np
import pytest

from qmeq.approach.dm_layout import NO_INDEX
from qmeq.approach.dm_layout import DensityMatrixLayout
from qmeq.approach.dm_layout import LiouvilleState
from qmeq.approach.kernel_handler import KernelHandler
from qmeq.approach.kernel_handler import KernelHandlerMatrixFree
from qmeq.indexing import StateIndexingDM


def make_si(nsingle, indexing='charge', removed=()):
    si = StateIndexingDM(nsingle, indexing=indexing)
    if removed:
        si.remove_fock_states(list(removed))
    return si


#: Layouts covering the normal models plus the degenerate shapes that ordinary
#: reference models never produce: a sector with no coherences, empty sectors at
#: both ends, and the symmetry-reduced modes.
LAYOUTS = [
    (1, 'charge', ()),          # no coherence sector at all
    (2, 'charge', ()),
    (3, 'charge', ()),
    (4, 'charge', ()),
    (4, 'sz', ()),
    (4, 'ssq', ()),
    (3, 'charge', (0, 1, 2, 3)),  # empty leading and trailing sectors
    (3, 'charge', (0,)),
]

LAYOUT_IDS = ['n%d-%s%s' % (n, m, '-cut%d' % len(r) if r else '') for n, m, r in LAYOUTS]


@pytest.fixture(params=LAYOUTS, ids=LAYOUT_IDS)
def layout(request):
    nsingle, indexing, removed = request.param
    return DensityMatrixLayout(make_si(nsingle, indexing, removed))


def random_hermitian(layout, seed):
    """A random Hermitian matrix supported only on carried elements."""
    rng = np.random.default_rng(seed)
    n = layout.si.nmany
    full = rng.normal(size=(n, n)) + 1j*rng.normal(size=(n, n))
    rho = np.zeros((n, n), dtype=complex)
    for state in layout.states():
        rho[state.ket, state.bra] = full[state.ket, state.bra]
    return rho + rho.conj().T


# --- Tier A: structure of the layout itself -------------------------------

def test_L1_only_same_charge_elements_are_carried(layout):
    """L1: elements exist only within a charge sector."""
    si = layout.si
    for charge in range(si.ncharge):
        for other in range(si.ncharge):
            if other == charge:
                continue
            for b in si.statesdm[charge]:
                for bp in si.statesdm[other]:
                    assert b != bp


def test_L2_reduced_indices_are_exactly_contiguous(layout):
    """L2: the carried elements occupy range(ndm0) with no gaps."""
    indices = sorted({layout.index(s) for s in layout.states()})
    assert indices == list(range(layout.ndm0))


def test_L2_populations_precede_coherences(layout):
    """L2: populations fill [0, npauli), coherences [npauli, ndm0)."""
    for state in layout.states():
        index = layout.index(state)
        if state.is_population:
            assert index < layout.npauli
        else:
            assert index >= layout.npauli


def test_L2_excluded_elements_report_the_sentinel(layout):
    """L2: anything not carried returns NO_INDEX, never a stray index."""
    si = layout.si
    carried = {(s.ket, s.bra, s.charge) for s in layout.states()}
    for charge in range(si.ncharge):
        for ket, bra in itertools.product(si.statesdm[charge], repeat=2):
            state = LiouvilleState(ket, bra, charge)
            if (ket, bra, charge) not in carried:
                assert layout.index(state) == NO_INDEX


def test_L3_conjugate_partners_share_one_index(layout):
    """L3: |b><bp| and |bp><b| map to the same reduced index."""
    for state in layout.states():
        partner = state.conjugate()
        assert layout.is_included(partner)
        assert layout.index(partner) == layout.index(state)


def test_L3_orientation_is_antisymmetric_off_the_diagonal(layout):
    """L3: exactly one orientation of an off-diagonal pair is the stored one."""
    for state in layout.states():
        if state.is_population:
            assert layout.is_conj(state)
        else:
            assert layout.is_conj(state) != layout.is_conj(state.conjugate())


def test_L3_each_element_has_exactly_one_representative(layout):
    """L3: is_unique picks each reduced index out exactly once."""
    counts = {}
    for state in layout.unique_states():
        index = layout.index(state)
        counts[index] = counts.get(index, 0) + 1
    assert sorted(counts) == list(range(layout.ndm0))
    assert set(counts.values()) == {1}


def test_L3_unique_and_conj_agree_for_charge_indexing(layout):
    """L3: the two predicates coincide under 'charge' and may not under 'ssq'."""
    if layout.si.indexing != 'charge':
        pytest.skip('rule stated for charge indexing')
    for state in layout.states():
        assert layout.is_unique(state) == layout.is_conj(state)


def test_L3_symmetry_modes_are_surjective_not_bijective():
    """L3: under 'ssq' several elements share one stored index."""
    layout = DensityMatrixLayout(make_si(4, 'ssq'))
    carried = list(layout.states())
    distinct = {layout.index(s) for s in carried}
    assert len(carried) > len(distinct)
    charge_layout = DensityMatrixLayout(make_si(4, 'charge'))
    carried = list(charge_layout.states())
    distinct = {charge_layout.index(s) for s in carried}
    assert len(carried) == 2*len(distinct) - charge_layout.npauli


def test_L4_packed_size_follows_the_rule(layout):
    """L4: ndm0r = npauli + 2*(ndm0 - npauli)."""
    assert layout.ndm0r == layout.npauli + 2*(layout.ndm0 - layout.npauli)
    assert layout.imag_offset == layout.ndm0 - layout.npauli


def test_L4_imaginary_partner_exists_exactly_for_coherences(layout):
    """L4: 'i >= npauli' and the older 'i + ndm0 - npauli >= ndm0' agree."""
    for index in range(layout.ndm0):
        old_spelling = index + layout.ndm0 - layout.npauli >= layout.ndm0
        assert layout.has_imag(index) == old_spelling
        if layout.has_imag(index):
            assert layout.imag_index(index) == index + layout.imag_offset
            assert layout.imag_index(index) < layout.ndm0r
        else:
            assert layout.imag_index(index) == NO_INDEX


def test_L4_excluded_index_never_claims_an_imaginary_partner(layout):
    """L4: the NO_INDEX sentinel must not survive the offset arithmetic.

    Guards the accident that makes the old spelling safe: for bbp == -1 the
    value ndm0 - 1 - npauli is always below ndm0, so the partner test fails.
    """
    assert not layout.has_imag(NO_INDEX)
    assert layout.imag_index(NO_INDEX) == NO_INDEX
    assert NO_INDEX + layout.ndm0 - layout.npauli < layout.ndm0


# --- Tier B: packing round trips ------------------------------------------

def test_L5_round_trip_preserves_a_hermitian_matrix(layout):
    """L5: unpack(pack(rho)) == rho for a supported Hermitian rho."""
    if layout.si.indexing != 'charge':
        pytest.skip('symmetry modes are surjective; see the ssq test')
    rho = random_hermitian(layout, seed=11)
    assert np.allclose(layout.unpack(layout.pack(rho)), rho, atol=1e-14)


def test_L5_round_trip_is_idempotent_in_every_mode(layout):
    """L5: packing is a projection, so a second round trip changes nothing."""
    rho = random_hermitian(layout, seed=12)
    once = layout.unpack(layout.pack(rho))
    twice = layout.unpack(layout.pack(once))
    assert np.allclose(once, twice, atol=1e-14)


def test_L5_unpacked_matrix_is_hermitian_by_construction(layout):
    """L5: Hermiticity is structural, not enforced."""
    rng = np.random.default_rng(13)
    vec = rng.normal(size=layout.ndm0r)
    rho = layout.unpack(vec)
    assert np.allclose(rho, rho.conj().T, atol=1e-14)


def test_L5_populations_are_real(layout):
    """L5: a diagonal element has no imaginary partner, so it cannot be complex."""
    rng = np.random.default_rng(14)
    rho = layout.unpack(rng.normal(size=layout.ndm0r))
    for state in layout.states():
        if state.is_population:
            assert rho[state.ket, state.bra].imag == 0.0


def test_L8_trace_is_the_multiplicity_weighted_population_sum(layout):
    """L8: trace(rho) is the multiplicity-weighted sum of the populations."""
    rho = random_hermitian(layout, seed=15)
    vec = layout.pack(rho)
    assert layout.trace(vec) == pytest.approx(np.trace(layout.unpack(vec)).real)


def test_L8_multiplicity_is_unity_except_under_ssq(layout):
    """L8: only 'ssq' makes a stored index stand for a whole multiplet."""
    weights = layout.multiplicity()
    assert np.all(weights[layout.npauli:] == 0.0)
    assert weights[:layout.npauli].sum() == len(
        [s for s in layout.states() if s.is_population])
    if layout.si.indexing in ('charge', 'sz'):
        assert np.all(weights[:layout.npauli] == 1.0)


def test_L8_ssq_multiplicities_are_greater_than_one():
    """L8: the weighted form is not decoration; 'ssq' really needs it."""
    layout = DensityMatrixLayout(make_si(4, 'ssq'))
    weights = layout.multiplicity()
    assert weights[:layout.npauli].max() > 1.0
    assert weights.sum() == layout.si.nmany


# --- Tier C: the insertion convention -------------------------------------

def random_superoperator(layout, seed):
    """A dense complex superoperator over all carried endpoints."""
    rng = np.random.default_rng(seed)
    states = list(layout.states())
    return {(t, u): rng.normal() + 1j*rng.normal()
            for t in states if layout.is_unique(t) for u in states}


def apply_complex(layout, W, rho):
    """Explicit complex action, used as the oracle for L6."""
    n = layout.si.nmany
    out = np.zeros((n, n), dtype=complex)
    for (t, u), fct in W.items():
        out[t.ket, t.bra] += fct*rho[u.ket, u.bra]
    return out


def assemble(layout, W, handler_cls=KernelHandler):
    kern = np.zeros((layout.ndm0r, layout.ndm0r))
    kh = handler_cls(layout.si)
    kh.set_kern(kern)
    for (t, u), fct in W.items():
        kh.set_matrix_element(fct, t.ket, t.bra, t.charge, u.ket, u.bra, u.charge)
    return kern


def test_L6_packed_kernel_implements_minus_i_times_the_complex_action(layout):
    """L6: the assembled kernel acts as rho -> -i (W rho).

    This is the whole insertion contract in one assertion. It is what makes
    Lindblad's ``1j*fct`` a convention artifact rather than a physics
    difference, and it needs no reference data to check.

    Stated for a bijective layout. Under 'ssq' the element-to-index map is
    many-to-one (L3), so a superoperator indexed by physical elements has no
    unambiguous complex matrix to compare against; the surjection itself is
    pinned separately below.
    """
    if layout.si.indexing != 'charge':
        pytest.skip('L6 oracle needs a bijective layout; see the ssq test')
    if layout.ndm0 == 0:
        pytest.skip('empty layout')
    W = random_superoperator(layout, seed=21)
    kern = assemble(layout, W)
    rho = random_hermitian(layout, seed=22)
    expected = layout.pack(-1j*apply_complex(layout, W, rho))
    assert np.allclose(kern @ layout.pack(rho), expected, atol=1e-12)


def test_L3_symmetry_partners_insert_into_the_same_kernel_entries():
    """L3: under 'ssq', inserting at a symmetry partner hits the same entries.

    The counterpart of the skipped L6 oracle: what makes the complex comparison
    ambiguous is exactly this identification, so pin the identification itself.
    """
    layout = DensityMatrixLayout(make_si(4, 'ssq'))
    groups = {}
    for state in layout.states():
        groups.setdefault((layout.index(state), layout.is_conj(state)), []).append(state)
    shared = [g for g in groups.values() if len(g) > 1]
    assert shared, 'expected ssq to identify several elements'

    for group in shared:
        target = group[0]
        kerns = []
        for source in group:
            kern = np.zeros((layout.ndm0r, layout.ndm0r))
            kh = KernelHandler(layout.si)
            kh.set_kern(kern)
            kh.set_matrix_element(1.0 + 0.5j, target.ket, target.bra, target.charge,
                                  source.ket, source.bra, source.charge)
            kerns.append(kern)
        for kern in kerns[1:]:
            assert np.array_equal(kern, kerns[0])


def test_L6_kernel_is_linear_in_the_inserted_value(layout):
    """L6: inserting fct twice is inserting 2*fct once."""
    W = random_superoperator(layout, seed=23)
    once = assemble(layout, {k: 2*v for k, v in W.items()})
    twice = assemble(layout, W) + assemble(layout, W)
    assert np.allclose(once, twice, atol=1e-12)


def test_L7_set_energy_is_set_matrix_element_with_a_real_value(layout):
    """L7: the bare Liouvillian is not a separate convention."""
    rng = np.random.default_rng(24)
    energies = {}
    for state in layout.unique_states():
        energies[state] = rng.normal()

    kern_energy = np.zeros((layout.ndm0r, layout.ndm0r))
    kh = KernelHandler(layout.si)
    kh.set_kern(kern_energy)
    for state, energy in energies.items():
        kh.set_energy(energy, state.ket, state.bra, state.charge)

    kern_element = assemble(
        layout, {(state, state): complex(energy, 0.0)
                 for state, energy in energies.items()})

    assert np.array_equal(kern_energy, kern_element)


def test_L6_matrix_free_negates_the_imaginary_rows(layout):
    """L6: the matrix-free path writes imaginary rows with the opposite sign.

    A per-row sign leaves the stationary null space untouched, which is why
    ``mfreeq=True`` and ``mfreeq=False`` agree on solutions. It does mean
    ``dphi0_dt`` is not literally the packed time derivative. Pinned here so
    that correcting one path without the other cannot pass silently.
    """
    if layout.ndm0 == layout.npauli:
        pytest.skip('no coherences, so no imaginary rows')
    W = random_superoperator(layout, seed=25)
    rho = random_hermitian(layout, seed=26)
    phi0 = layout.pack(rho)

    kern = assemble(layout, W)

    dphi0_dt = np.zeros(layout.ndm0r)
    khm = KernelHandlerMatrixFree(layout.si)
    khm.set_phi0(phi0)
    khm.set_dphi0_dt(dphi0_dt)
    for (t, u), fct in W.items():
        khm.set_matrix_element(fct, t.ket, t.bra, t.charge, u.ket, u.bra, u.charge)

    row_sign = np.ones(layout.ndm0r)
    row_sign[layout.ndm0:] = -1.0
    assert np.allclose(row_sign*(kern @ phi0), dphi0_dt, atol=1e-12)
    assert not np.allclose(kern @ phi0, dphi0_dt, atol=1e-12)


# --- Tier D: agreement with the shipped machinery -------------------------

def test_L8_norm_vec_selects_exactly_the_population_entries():
    """L8: generate_norm_vec builds the trace functional of L8."""
    import qmeq

    system = qmeq.Builder(2, {(0, 0): 0.0, (1, 1): 0.1, (0, 1): 0.3}, {},
                          2, {(0, 0): 0.3, (0, 1): 0.3, (1, 0): 0.25, (1, 1): 0.25},
                          [0.5, -0.5], [1.0, 1.0], [1e3, 1e3], kerntype='1vN', itype=1)
    system.solve()
    appr = system.appr
    layout = DensityMatrixLayout(appr.si)
    assert np.array_equal(appr.norm_vec, layout.multiplicity())


@pytest.mark.parametrize('indexing', ['charge', 'sz', 'ssq'])
def test_L8_norm_vec_is_the_multiplicity_vector(indexing):
    """L8: generate_norm_vec accumulates multiplicities, including under ssq."""
    import qmeq

    nsingle = 4
    hsingle = {(i, i): 0.1*i for i in range(nsingle)}
    tleads = {(l, i): 0.3 for l in range(2) for i in range(nsingle)}
    system = qmeq.Builder(nsingle, hsingle, {}, 2, tleads,
                          [0.5, -0.5], [1.0, 1.0], [1e3, 1e3],
                          kerntype='1vN', itype=1, indexing=indexing)
    system.solve()
    layout = DensityMatrixLayout(system.appr.si)
    assert np.array_equal(system.appr.norm_vec, layout.multiplicity())


def test_handler_offset_matches_the_specification(layout):
    """L4: the handler's precomputed offset is the layout's offset."""
    kh = KernelHandler(layout.si)
    assert kh.imag_offset == layout.imag_offset
    assert kh.npauli == layout.npauli
    assert kh.ndm0 == layout.ndm0
    assert kh.ndm0r == layout.ndm0r


def test_handler_predicates_match_the_specification(layout):
    """L2, L3: is_included and is_unique agree with the reference layout."""
    kh = KernelHandler(layout.si)
    si = layout.si
    for charge in range(si.ncharge):
        for ket, bra in itertools.product(si.statesdm[charge], repeat=2):
            state = LiouvilleState(ket, bra, charge)
            assert kh.is_included(ket, bra, charge) == layout.is_included(state)
            assert bool(kh.is_unique(ket, bra, charge)) == layout.is_unique(state)


def test_get_phi0_element_matches_the_reference_unpack(layout):
    """L5: the handler's reconstruction is the layout's unpack."""
    rng = np.random.default_rng(31)
    vec = rng.normal(size=layout.ndm0r)
    kh = KernelHandler(layout.si)
    kh.set_phi0(vec)
    rho = layout.unpack(vec)
    for state in layout.states():
        assert kh.get_phi0_element(state.ket, state.bra, state.charge) == pytest.approx(
            rho[state.ket, state.bra])


def test_named_accessors_match_the_maptype_integers(layout):
    """L3: the named accessors are exactly the integer maptypes they replace."""
    si = layout.si
    for state in layout.states():
        b, bp, c = state.ket, state.bra, state.charge
        assert si.get_ind_dm0_bool(b, bp, c) == si.get_ind_dm0(b, bp, c, maptype=2)
        assert si.get_ind_dm0_conj(b, bp, c) == si.get_ind_dm0(b, bp, c, maptype=3)


@pytest.mark.parametrize('cls_name', ['StateIndexingDM', 'StateIndexingDMc',
                                     'StateIndexingPauli'])
@pytest.mark.parametrize('indexing', ['charge', 'sz', 'ssq'])
def test_named_bool_accessor_exists_on_every_class(cls_name, indexing):
    """L3: maptype=2 is supported by all three indexing classes, so the named
    form must be too.

    The electron-phonon approaches hold two indexing objects at once: ``si`` is
    a StateIndexingDM while ``si_elph`` is a StateIndexingDMc. Code reached
    through ``si_elph`` therefore needs the accessor on that class as well.
    """
    import qmeq.indexing as indexing_module

    si = getattr(indexing_module, cls_name)(4, indexing=indexing)
    for charge in range(si.ncharge):
        for b, bp in itertools.product(si.statesdm[charge], repeat=2):
            assert si.get_ind_dm0_bool(b, bp, charge) == si.get_ind_dm0(
                b, bp, charge, maptype=2)


def test_liouville_state_is_immutable():
    state = LiouvilleState(1, 2, 1)
    with pytest.raises(AttributeError):
        state.ket = 3
    with pytest.raises(AttributeError):
        del state.bra
    assert state.conjugate() == LiouvilleState(2, 1, 1)
    assert state.conjugate().conjugate() == state
    assert not state.is_population
    assert LiouvilleState(1, 1, 1).is_population


def test_L2_no_index_endpoint_is_a_no_op_not_a_stray_write(layout, monkeypatch):
    """L2: inserting at an endpoint with no index must not touch the kernel.

    Every shipped caller guards with ``is_included``, and a probe run of the
    whole suite with a hard assertion never fired. The guard exists so that a
    future caller that forgets loses its contribution loudly-by-testing rather
    than silently corrupting the last row or column through the -1 sentinel.
    """
    from qmeq.approach import dm_layout

    monkeypatch.setattr(dm_layout, 'STRICT_INDEX', False)
    si = layout.si
    charges = [c for c in range(si.ncharge) if len(si.statesdm[c]) > 0]
    if not charges:
        pytest.skip('no states')
    charge = charges[0]
    b = si.statesdm[charge][0]

    missing = None
    for other in range(si.ncharge):
        for candidate in si.statesdm[other]:
            if si.get_ind_dm0(b, candidate, charge) == NO_INDEX:
                missing = candidate
                break
        if missing is not None:
            break
    if missing is None:
        pytest.skip('this layout carries every pair')

    kern = np.zeros((layout.ndm0r, layout.ndm0r))
    kh = KernelHandler(si)
    kh.set_kern(kern)
    kh.set_matrix_element(1.0 + 1.0j, b, missing, charge, b, b, charge)
    kh.set_matrix_element(1.0 + 1.0j, b, b, charge, b, missing, charge)
    assert not kern.any()

    dphi0_dt = np.zeros(layout.ndm0r)
    khm = KernelHandlerMatrixFree(si)
    khm.set_phi0(np.zeros(layout.ndm0r))
    khm.set_dphi0_dt(dphi0_dt)
    khm.set_matrix_element(1.0 + 1.0j, b, missing, charge, b, b, charge)
    assert not dphi0_dt.any()


@pytest.mark.parametrize('cls_name,args', [
    ('StateIndexingDM', (1, 2, 1)),
    ('StateIndexingDMc', (1, 2, 1)),
    ('StateIndexingPauli', (1, 1, 1)),
])
def test_unsupported_maptype_raises(cls_name, args):
    """Unsupported maptype values used to return None, which NumPy accepts as
    np.newaxis and silently reshapes. They now raise."""
    import qmeq.indexing as indexing

    si = getattr(indexing, cls_name)(2, indexing='charge')
    assert si.get_ind_dm0(*args, maptype=1) != NO_INDEX
    with pytest.raises(ValueError, match='maptype'):
        si.get_ind_dm0(*args, maptype=99)
    if cls_name != 'StateIndexingDM':
        with pytest.raises(ValueError, match='maptype'):
            si.get_ind_dm0(*args, maptype=3)


def test_strict_index_mode_turns_a_skipped_insertion_into_an_error(monkeypatch):
    """L2: STRICT_INDEX converts the silent skip into a raise.

    The default no-op is the right runtime behaviour, but it makes a missing
    ``is_included`` guard invisible. Strict mode makes the whole suite a probe
    for that mistake, which is how the two live defects in this area were found.
    Pure-Python only: the compiled insertion methods are ``noexcept nogil``.
    """
    from qmeq.approach import dm_layout

    layout = DensityMatrixLayout(make_si(4, 'sz'))
    si = layout.si
    charge = next(c for c in range(si.ncharge) if len(si.statesdm[c]) > 0)
    b = si.statesdm[charge][0]
    missing = next(
        (cand for other in range(si.ncharge) for cand in si.statesdm[other]
         if si.get_ind_dm0(b, cand, charge) == NO_INDEX), None)
    assert missing is not None, 'sz indexing should exclude some pair'

    kern = np.zeros((layout.ndm0r, layout.ndm0r))
    kh = KernelHandler(si)
    kh.set_kern(kern)

    monkeypatch.setattr(dm_layout, 'STRICT_INDEX', False)
    kh.set_matrix_element(1.0 + 1.0j, b, missing, charge, b, b, charge)
    assert not kern.any()

    monkeypatch.setattr(dm_layout, 'STRICT_INDEX', True)
    with pytest.raises(IndexError, match='no index'):
        kh.set_matrix_element(1.0 + 1.0j, b, missing, charge, b, b, charge)
    assert not kern.any()


def test_strict_index_is_off_by_default_and_env_driven():
    """The default must stay silent so normal runs are unaffected."""
    from qmeq.approach import dm_layout

    assert dm_layout.STRICT_INDEX_ENV == 'QMEQ_STRICT_INDEX'
    assert dm_layout.STRICT_INDEX is (
        os.environ.get('QMEQ_STRICT_INDEX', '').strip().lower()
        in ('1', 'true', 'on', 'yes'))


def test_rtd_matrix_selector_values_are_the_historical_integers():
    """The RtdMatrix values index set_matrix_list and must not be reordered."""
    from qmeq.approach.kernel_handler import RtdMatrix

    assert [(m.name, m.value) for m in RtdMatrix] == [
        ('Wdd', 0), ('WE1', 1), ('WE2', 2), ('ReWdn', 3),
        ('ImWdn', 4), ('ReWnd', 5), ('ImWnd', 6), ('Lnn_inv', 7)]


def test_rtd_matrix_selects_the_array_its_name_denotes():
    """set_matrix_list must order its arrays to match the enum.

    The contract that makes the selector readable is that ``mats[RtdMatrix.X]``
    is the array called ``X``. Pinned by handing the handler distinguishable
    sentinels named after the members and checking where each lands.
    """
    from qmeq.approach.kernel_handler import KernelHandlerRTD, RtdMatrix

    kh = KernelHandlerRTD(make_si(2, 'charge'))
    for member in RtdMatrix:
        setattr(kh, member.name, 'array-%s' % member.name)
    kh.set_matrix_list()

    for member in RtdMatrix:
        assert kh.mats[member] == 'array-%s' % member.name, member


def test_compiled_rtd_matrix_enum_mirrors_the_python_one():
    """The cdef enum in c_kernel_handler.pxd must agree member-for-member.

    Declared ``cpdef`` precisely so this check is possible: a plain ``cdef``
    enum is invisible to Python and the two copies could drift silently.
    """
    import qmeq

    if qmeq.get_backend_status()['active'] != 'cython':
        pytest.skip('compiled backend not active')

    from qmeq.approach.c_kernel_handler import RtdMatrixC
    from qmeq.approach.kernel_handler import RtdMatrix

    compiled = {m.name: int(m) for m in RtdMatrixC}
    expected = {'MAT_WDD': 'Wdd', 'MAT_WE1': 'WE1', 'MAT_WE2': 'WE2',
                'MAT_RE_WDN': 'ReWdn', 'MAT_IM_WDN': 'ImWdn',
                'MAT_RE_WND': 'ReWnd', 'MAT_IM_WND': 'ImWnd',
                'MAT_LNN_INV': 'Lnn_inv'}
    assert set(compiled) == set(expected)
    for c_name, py_name in expected.items():
        assert compiled[c_name] == RtdMatrix[py_name].value, c_name
