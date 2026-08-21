"""Canonical specification of QmeQ's packed real density-matrix layout.

Lindblad, Redfield, 1vN, their electron-phonon variants and the RTD approaches
all solve for the stationary density matrix in one real vector of length
``si.ndm0r``.  Until this module existed that layout was defined only by its
uses: the offset arithmetic, the inclusion test and the conjugation sign were
open-coded at every insertion site in :mod:`qmeq.approach.kernel_handler` and
its Cython twin.  This module is the specification those sites now refer to.

The rules below are numbered.  Every test in ``qmeq/tests/test_dm_layout.py``
names the rule it pins, so a reviewer can check a diagram generator or a new
kernel handler against a written contract rather than against nested loops.

Rules
-----
**L1 — sector partition.**  Only same-charge elements exist.  ``si.statesdm[c]``
lists the many-body states of charge ``c``.  The flat, unreduced index of
:math:`|b\\rangle\\langle b'|` is ``maptype=0``::

    lenlst[c] * dictdm[b] + dictdm[bp] + shiftlst0[c]

**L2 — reduced index.**  ``mapdm0`` maps the flat index either to
:data:`NO_INDEX` (the element is not carried) or to a contiguous reduced index.
Populations occupy ``[0, npauli)`` and coherences ``[npauli, ndm0)``.  The
reduced indices of the included elements are exactly ``range(si.ndm0)``.

**L3 — pairs and orientation.**  :math:`|b\\rangle\\langle b'|` and
:math:`|b'\\rangle\\langle b|` share one reduced index.  Two boolean maps
distinguish their roles, and they are *not* the same predicate:

* ``booldm0`` (``maptype=2``, :meth:`StateIndexingDM.get_ind_dm0_bool`) marks
  the representative used to enumerate the element exactly once.
* ``conjdm0`` (``maptype=3``, :meth:`StateIndexingDM.get_ind_dm0_conj`) marks
  the stored orientation.  It selects the sign in L5.

Under ``indexing='charge'`` the two coincide and ``conjdm0`` is true precisely
when ``b < bp`` in ``statesdm`` order.  Under ``'ssq'`` they differ, and the map
from physical elements to reduced indices is a surjection rather than a
bijection: several elements related by the symmetry share one stored index.
Round-tripping is therefore only faithful for states already lying in the
symmetric subspace.

**L4 — real packing.**  The packed vector has length::

    ndm0r = npauli + 2 * (ndm0 - npauli)

Entry ``i`` holds a real part for every included element.  A coherence, and
only a coherence, additionally has an imaginary partner at ``i + IMAG_OFFSET``
with ``IMAG_OFFSET = ndm0 - npauli``.  Equivalently, ``i`` has an imaginary
partner iff ``i >= npauli``; the older spelling ``i + ndm0 - npauli >= ndm0``
is the same test.

**L5 — reconstruction.**  With ``i`` the reduced index of
:math:`|b\\rangle\\langle b'|` and ``s = +1`` if ``conjdm0`` else ``-1``::

    rho[b, bp] = phi0[i] + 1j * s * phi0[i + IMAG_OFFSET]

Hermiticity is structural: the sign flip between an element and its partner is
what makes :math:`\\rho^\\dagger = \\rho` hold by construction.  It is not
enforced anywhere, so a packed vector cannot represent a non-Hermitian matrix.

**L6 — insertion convention.**  This is the rule with the least local evidence
and the widest reach.  The packed kernel assembled by
``KernelHandler.set_matrix_element`` implements

.. math:: \\rho \\mapsto -i\\,(W\\rho)

That is, a value ``fct`` inserted at ``(b, bp) <- (a, ap)`` contributes
``-1j * fct * rho[a, ap]`` to element ``(b, bp)`` of the result.  This is why
1vN and Redfield pass their kernel entries directly while Lindblad passes
``1j * fct``: the Lindblad dissipator carries no ``i`` of its own and must
cancel the one built in here.  That difference is a convention artifact, not a
difference of physics.

**L7 — the bare Liouvillian is not a special case.**  As a consequence of L6,

    ``set_energy(E, b, bp, c) == set_matrix_element(complex(E, 0), b, bp, c, b, bp, c)``

holds exactly, because :math:`-i(E_b - E_{b'})\\rho_{bb'}` is the commutator
term.  ``set_energy`` is a fast path, not a separate convention.  This identity
is the reason the ``-i`` of L6 is load-bearing and must not be normalized away:
it lets the commutator and the tunnelling kernel share one insertion path.

**L8 — trace and normalization.**  The trace is a *multiplicity-weighted* sum
over the population entries::

    trace(phi0) = sum(multiplicity[i] * phi0[i] for i in range(npauli))

where ``multiplicity[i]`` counts the physical diagonal elements that share the
stored index ``i`` under L3.  Under ``'charge'`` and ``'sz'`` every multiplicity
is one and this degenerates to ``sum(phi0[0:npauli])``; under ``'ssq'`` it does
not, because a stored index then represents a whole symmetry multiplet.  This
weighted vector is exactly what ``Approach.generate_norm_vec`` accumulates with
its ``norm_vec[bb] += 1`` per many-body state, and what ``solve_kern``
substitutes into the row given by ``funcp.norm_row``.

**L9 — the RTD coherence packing is a different layout.**  ``KernelHandlerRTD``
inserts into ``Wdn``, ``Wnd`` and ``Lnn_inv``, whose coherence axis is *not* the
``ndm0r`` layout above.  There a coherence with reduced index ``i`` sits at
``i - npauli``, and its imaginary part at ``i - npauli + IMAG_OFFSET``, giving
an axis of length ``2 * (ndm0 - npauli)`` with no population entries at all.
That layout also selects orientation by comparing ``b > bp`` directly rather
than by consulting ``conjdm0``; under ``indexing='charge'``, the only mode RTD
supports, the two agree.  Kept as-is and documented: legacy RTD depends on it.

Notes
-----
This module deliberately has no NumPy-free fast path and is not on the hot
insertion path.  The handlers use :data:`IMAG_OFFSET`-style precomputed
constants derived from the same rules; this module is the reference
implementation the tests compare them against.
"""

from __future__ import annotations

import itertools
import os
from typing import Iterator

import numpy as np

from ..indexing import StateIndexingDM
from ..wrappers.mytypes import complexnp
from ..wrappers.mytypes import doublenp

NO_INDEX = -1
"""Returned by ``get_ind_dm0`` in place of an index when the density-matrix
element is not part of the reduced representation, so no index exists for it
(rule L2).  The complement of ``KernelHandler.is_included``.

Never index an array with this value. NumPy/Cython do not catch an unguarded
use automatically:

* in pure Python, NumPy reads ``-1`` as the *last* row or column and silently
  corrupts an unrelated entry;
* in the compiled path it is worse.  ``c_kernel_handler.pyx`` is built with
  ``wraparound=False`` and ``boundscheck=False``, so ``-1`` is an out-of-bounds
  access with undefined behaviour rather than a wrap.

The shared handlers now guard this value before indexing. Compare against this
name in new code as well, and see :data:`STRICT_INDEX` for a mode that turns a
missed caller-side inclusion check into an exception.
"""

STRICT_INDEX_ENV = 'QMEQ_STRICT_INDEX'
"""Environment variable enabling :data:`STRICT_INDEX`."""

STRICT_INDEX = os.environ.get(STRICT_INDEX_ENV, '').strip().lower() in (
    '1', 'true', 'on', 'yes')
"""When true, inserting a matrix element at an endpoint with no index raises
instead of being silently skipped.  Off by default and read once from
``QMEQ_STRICT_INDEX``; tests may set it directly.

This is a pure-Python diagnostic.  The compiled handler's insertion methods are
``noexcept nogil`` and cannot raise, so run the strict leg with
``QMEQ_BACKEND=python``.  Because the check lives inside a branch that a correct
caller never takes, enabling it costs nothing in the normal path.
"""


def no_index_error(method, endpoints):
    """Build the exception raised under :data:`STRICT_INDEX`.

    Parameters
    ----------
    method : str
        Name of the insertion method that was reached.
    endpoints : dict
        Endpoint labels mapped to their looked-up index, so the offending one is
        visible as ``NO_INDEX`` in the message.

    Returns
    -------
    IndexError
        Describing which endpoint has no index.
    """
    shown = ', '.join('%s=%s' % (name, 'NO_INDEX' if value == NO_INDEX else value)
                      for name, value in endpoints.items())
    return IndexError(
        '%s reached an endpoint with no index (%s). The element is not part of '
        'the reduced representation, so the contribution has nowhere to go; the '
        'caller is missing an is_included guard. Raised because %s is set.'
        % (method, shown, STRICT_INDEX_ENV))


class LiouvilleState(object):
    """Immutable ``(ket, bra, charge)`` view of one density-matrix element.

    Attributes
    ----------
    ket : int
        Many-body index :math:`b` of :math:`|b\\rangle\\langle b'|`.
    bra : int
        Many-body index :math:`b'`.
    charge : int
        Charge of both states; they are equal by L1.
    """

    __slots__ = ('ket', 'bra', 'charge')

    ket: int
    bra: int
    charge: int

    def __init__(self, ket: int, bra: int, charge: int) -> None:
        object.__setattr__(self, 'ket', int(ket))
        object.__setattr__(self, 'bra', int(bra))
        object.__setattr__(self, 'charge', int(charge))

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError('LiouvilleState is immutable')

    def __delattr__(self, name: str) -> None:
        raise AttributeError('LiouvilleState is immutable')

    @property
    def is_population(self) -> bool:
        return self.ket == self.bra

    def conjugate(self) -> LiouvilleState:
        """Return :math:`|b'\\rangle\\langle b|`, the Hermitian partner (L3)."""
        return LiouvilleState(self.bra, self.ket, self.charge)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LiouvilleState):
            return NotImplemented
        return (self.ket, self.bra, self.charge) == (other.ket, other.bra, other.charge)

    def __hash__(self) -> int:
        return hash((self.ket, self.bra, self.charge))

    def __repr__(self) -> str:
        return 'LiouvilleState(ket=%d, bra=%d, charge=%d)' % (self.ket, self.bra, self.charge)


class DensityMatrixLayout(object):
    """Reference implementation of the packed real ``dm0`` layout.

    Wraps a :class:`qmeq.indexing.StateIndexingDM` and exposes the rules of
    this module as named operations.  Used by tests and by callers that need
    the layout without the performance constraints of an insertion loop.

    Parameters
    ----------
    si : StateIndexingDM
        State indexing for the system, from :mod:`qmeq.indexing`.  Must carry
        the ``dm0`` conjugation map, so ``StateIndexingDMc`` is not accepted.
    """

    si: StateIndexingDM
    npauli: int
    ndm0: int
    ndm0r: int
    imag_offset: int

    def __init__(self, si: StateIndexingDM) -> None:
        self.si = si
        self.npauli = si.npauli
        self.ndm0 = si.ndm0
        self.ndm0r = si.ndm0r
        self.imag_offset = si.ndm0 - si.npauli
        """Distance from a reduced index to its imaginary partner (L4)."""

    # -- indexing (L1-L4) --------------------------------------------------

    def index(self, state: LiouvilleState) -> int:
        """Reduced index of ``state``, or :data:`NO_INDEX` (L2)."""
        return self.si.get_ind_dm0(state.ket, state.bra, state.charge)

    def is_included(self, state: LiouvilleState) -> bool:
        """True if the element is carried at all (L2)."""
        return self.index(state) != NO_INDEX

    def is_unique(self, state: LiouvilleState) -> bool:
        """True for the representative used to enumerate the element once (L3)."""
        return bool(self.si.get_ind_dm0_bool(state.ket, state.bra, state.charge))

    def is_conj(self, state: LiouvilleState) -> bool:
        """True if ``state`` is the stored orientation (L3)."""
        return bool(self.si.get_ind_dm0_conj(state.ket, state.bra, state.charge))

    def conj_sign(self, state: LiouvilleState) -> int:
        """``+1`` for the stored orientation, ``-1`` for its partner (L3, L5)."""
        return +1 if self.is_conj(state) else -1

    def has_imag(self, index: int) -> bool:
        """True if the reduced ``index`` carries an imaginary part (L4)."""
        return index >= self.npauli

    def imag_index(self, index: int) -> int:
        """Packed position of the imaginary part, or :data:`NO_INDEX` (L4)."""
        if not self.has_imag(index):
            return NO_INDEX
        return index + self.imag_offset

    def states(self) -> Iterator[LiouvilleState]:
        """Iterate every included element, both orientations (L1, L2)."""
        si = self.si
        for charge in range(si.ncharge):
            for ket, bra in itertools.product(si.statesdm[charge], repeat=2):
                state = LiouvilleState(ket, bra, charge)
                if self.is_included(state):
                    yield state

    def unique_states(self) -> Iterator[LiouvilleState]:
        """Iterate each element once, in its representative orientation (L3)."""
        for state in self.states():
            if self.is_unique(state):
                yield state

    # -- packing (L5, L8) --------------------------------------------------

    def pack(self, rho: np.ndarray) -> np.ndarray:
        """Pack a complex density matrix into the real ``ndm0r`` vector (L5)."""
        vec = np.zeros(self.ndm0r, dtype=doublenp)
        for state in self.unique_states():
            index = self.index(state)
            vec[index] = rho[state.ket, state.bra].real
            imag = self.imag_index(index)
            if imag != NO_INDEX:
                vec[imag] = self.conj_sign(state) * rho[state.ket, state.bra].imag
        return vec

    def unpack(self, vec: np.ndarray) -> np.ndarray:
        """Expand a packed real vector into a complex matrix (L5)."""
        rho = np.zeros((self.si.nmany, self.si.nmany), dtype=complexnp)
        for state in self.states():
            index = self.index(state)
            imag = self.imag_index(index)
            value = vec[index]
            if imag != NO_INDEX:
                value = value + 1j * self.conj_sign(state) * vec[imag]
            rho[state.ket, state.bra] = value
        return rho

    def multiplicity(self) -> np.ndarray:
        """Number of physical diagonal elements behind each stored index (L8).

        All ones except under ``indexing='ssq'``, where a stored index stands
        for a symmetry multiplet. This is the vector
        ``Approach.generate_norm_vec`` accumulates.
        """
        weights = np.zeros(self.ndm0r, dtype=doublenp)
        for state in self.states():
            if state.is_population:
                weights[self.index(state)] += 1.0
        return weights

    def trace(self, vec: float | np.ndarray) -> float:
        """Trace of a packed vector: its multiplicity-weighted populations (L8)."""
        return float(np.dot(self.multiplicity(), vec))
