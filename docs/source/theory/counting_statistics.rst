Zero-frequency current counting statistics
==========================================

QmeQ calculates the first two zero-frequency cumulants of the particle current:
the mean current :math:`I` and noise :math:`S`. The implementation follows the
counting-field formulation of `Emary, Phys. Rev. B 80, 235306 (2009)
<https://arxiv.org/abs/0902.3544>`_. It was developed by Simon Wozny in his
`QmeQ fork <https://github.com/si8881wo/qmeq>`_; his
`example notebook <https://github.com/si8881wo/qmeq-noise-example>`_ provides a
longer worked calculation.

First-order formula
-------------------

For Pauli, Lindblad, Redfield, and 1vN, QmeQ decomposes the Markovian kernel as

.. math::

   K(\chi) = K + (e^{i\chi}-1)J_+ + (e^{-i\chi}-1)J_- .

All leads in ``countingleads`` share the same field :math:`\chi`; the result is
therefore the cumulant of their aggregate particle transfer. It is not a matrix
of lead-to-lead cross correlations.

Let :math:`|0\rangle\!\rangle` be the normalized stationary state,
:math:`\langle\!\langle\tilde 0|` the approach's trace vector, and

.. math::

   Q = 1-|0\rangle\!\rangle\langle\!\langle\tilde 0|,
   \qquad R = Q K^+ Q,

where :math:`K^+` is the Moore--Penrose pseudoinverse. With
:math:`K^{(1)}=i(J_+-J_-)` and :math:`K^{(2)}=-(J_++J_-)`, QmeQ evaluates

.. math::

   I = -i\langle\!\langle\tilde 0|K^{(1)}|0\rangle\!\rangle,

.. math::

   S = -\langle\!\langle\tilde 0|
       \left[K^{(2)}-2K^{(1)}RK^{(1)}\right]
       |0\rangle\!\rangle.

The physical kernel is checked for exactly one stationary state. A disconnected
model with a non-unique stationary state raises ``LinAlgError`` instead of
returning a basis-dependent cumulant. The same Python counting-kernel and
pseudoinverse code runs after either the pure-Python or compiled first-order
solver.

API and conventions
-------------------

Counting is opt-in. ``None`` is the default and performs no counting-kernel
allocation. A nonempty iterable must contain unique integer lead indices:

.. code-block:: python

   import qmeq

   system = qmeq.Builder(
       nsingle=1,
       hsingle={(0, 0): 0.0},
       nleads=2,
       tleads={(0, 0): 0.1, (1, 0): 0.1},
       mulst={0: 1.0, 1: -1.0},
       tlst={0: 0.2, 1: 0.2},
       dband={0: 100.0, 1: 100.0},
       kerntype="Pauli",
       countingleads=[0],
   )
   system.solve()
   current, noise = system.current_noise

``current_noise`` is ``[I, S]``. The current uses QmeQ's convention: positive
means particle flow from the counted lead into the dot. Consequently its first
entry equals the sum of the corresponding entries in ``system.current``.
Changing ``system.countingleads`` before another solve changes the counted
aggregate; assigning ``None`` disables it again.

QmeQ uses natural units :math:`\hbar=k_\mathrm{B}=|e|=1`. Thus particle current
and zero-frequency particle-number noise have units of inverse time (energy in
these units). To obtain charge cumulants, multiply :math:`I` by the signed
electron charge according to the desired electrical-current convention and
:math:`S` by :math:`e^2`.

RTD results and approximation order
-----------------------------------

Use ``kerntype='pyRTDnoise'`` for Real Time Diagrammatic counting statistics.
``kerntype='RTDnoise'`` is a documented alias for that same pure-Python
implementation:

.. code-block:: python

   system = qmeq.Builder(
       # model parameters as above
       kerntype="RTDnoise",
       countingleads=[0],
       off_diag_corrections=False,
   )
   system.solve()
   full = system.current_noise
   sequential = system.current_noise_first
   truncated = system.current_noise_o4trunc

The arrays mean:

* ``current_noise`` is ``[I, S]`` from the full fourth-order RTD kernel and its
  stationary state.
* ``current_noise_first`` is the sequential ``[I, S]`` result.
* ``current_noise_o4trunc`` is
  ``[I_sequential, I_fourth_order, S_sequential, S_fourth_order]``, where the
  latter member of each pair is evaluated with the consistent fourth-order
  truncation.

Here *fourth order* always means fourth order in the tunnelling Hamiltonian
:math:`H_T`, equivalently second order in the tunnel rate :math:`\Gamma`. It
does not mean fourth order in :math:`\Gamma`. RTD noise includes the
non-Markovian energy-derivative terms of Emary's formulation.

Limitations
-----------

Only the first two zero-frequency particle-current cumulants are implemented.
There are no arbitrary higher cumulants, separate lead-to-lead correlation
matrices, or energy-current noise. Counting is not implemented for 2vN,
electron-phonon approaches, or matrix-free solvers. RTD counting does not
implement off-diagonal corrections and requires
``off_diag_corrections=False``. These combinations raise
``NotImplementedError`` rather than returning uncontrolled values.
