Transport integration options
=============================

QmeQ exposes two descriptive options for the lead integrals used by its
transport approaches:

``bandwidth``
    Selects whether the finite lead bandwidths in ``dband`` are enforced.
    Use ``'finite'`` to drop transitions outside the bands or ``'infinite'``
    for the wide-band limit.

``principal_part``
    Selects how principal-value contributions are evaluated.  Use ``'quad'``
    for numerical quadrature, ``'digamma'`` for the wide-band analytic
    approximation, or ``'omit'`` to neglect them.

The names have the same meaning for every approach, while the supported
combinations depend on the approximation:

==========  ==========================================================
Approach    Supported ``(bandwidth, principal_part)`` pairs
==========  ==========================================================
Pauli       ``(finite, omit)``, ``(infinite, omit)``
1vN         ``(finite, quad)``, ``(infinite, digamma)``,
            ``(finite, omit)``, ``(infinite, omit)``
Redfield    ``(finite, quad)``, ``(infinite, digamma)``,
            ``(finite, omit)``, ``(infinite, omit)``
Lindblad    ``(finite, digamma)``, ``(infinite, digamma)``,
            ``(finite, omit)``, ``(infinite, omit)``
RTD         ``(infinite, digamma)``
2vN         neither option is used
==========  ==========================================================

For Pauli, ``principal_part='omit'`` records that the approximation has no
principal-value contribution; it does not change the Pauli kernel.

For Lindblad, the principal-value contribution is the
:doc:`Lamb shift <lambshift>`.  Numerical quadrature is not implemented for
this approach, so ``principal_part='quad'`` raises ``ValueError`` instead of
silently selecting a different calculation.

For example, an infinite-band Lindblad calculation including the Lamb shift
uses:

.. code-block:: python

    system = qmeq.Builder(
        nsingle, hsingle, coulomb, nleads, tleads, mulst, tlst, dband,
        kerntype="Lindblad",
        bandwidth="infinite",
        principal_part="digamma",
    )

Legacy ``itype``
----------------

The integer ``itype`` remains available for backwards compatibility.  In the
1vN and Redfield approaches its values map to:

===========  ==================  ====================
``itype``    ``bandwidth``       ``principal_part``
===========  ==================  ====================
0            ``finite``          ``quad``
1            ``infinite``        ``digamma``
2            ``finite``          ``omit``
3            ``infinite``        ``omit``
===========  ==================  ====================

For Lindblad, legacy ``itype`` controls only the bandwidth: values 0 and 2
select ``'finite'``, while 1 and 3 select ``'infinite'``.  It always preserves
the historical omission of the Lamb shift.  Use ``principal_part='digamma'``
explicitly to include it.

New code should prefer the descriptive options.  Supplying ``itype`` together
with a conflicting descriptive value raises ``ValueError``.
