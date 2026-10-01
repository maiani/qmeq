QmeQ: Quantum master equation for Quantum dot transport calculations
====================================================================

QmeQ is an open-source Python package for calculations of transport through
quantum dot devices. The device is an Anderson-type model, a set of
interacting orbitals coupled to non-interacting leads by tunnelling. QmeQ
calculates the stationary state and the stationary **particle**, **energy** and
**heat currents** with several approximate density-matrix approaches:

* first order in the tunnel coupling, which describes Coulomb blockade:
  * the Pauli (classical) master equation,
  * the Lindblad approach, optionally with the lead-induced Lamb shift,
  * the Redfield approach, and
  * the first-order von Neumann (1vN) approach;
* second order in the tunnel coupling, which adds cotunnelling and pair
  tunnelling:
  * the second-order von Neumann (2vN) approach, which also describes the
    broadening of the dot states, and
  * the Real Time Diagrammatic (RTD) approach, which needs much less memory
    and computation time.

QmeQ also provides:

* electron-phonon coupling inside the dot, for the first-order approaches;
* zero-frequency counting statistics, the mean current and its noise resolved
  by lead, for Pauli, Lindblad, Redfield, 1vN and RTD (`RTDnoise`). They follow
  [Emary's formulation](https://arxiv.org/abs/0902.3544) and were developed by
  Simon Wozny in his [QmeQ fork](https://github.com/si8881wo/qmeq);
* diagnostics that warn when a stationary state is unphysical or an approach
  is used outside its regime; and
* compiled (Cython) and pure-Python implementations of the same approaches.

Physics disclaimer
------------------

All the methods in QmeQ are approximate so depending on parameter regime they
**can fail**, and a good knowledge of the method is required whether to trust
the result or not. For example, the Redfield, 1vN, 2vN and RTD approaches can
**violate positivity** of the reduced density matrix and lead to **currents
flowing against the bias**. We still think it is important to have a package
where a user can duplicate existing calculations, check applicability of
different methods, or simply discover new kind of physics using different
approximate master equations. What each approach assumes, where it fails, and
how to check convergence is collected in
[the approaches guide](docs/docs/guide/approaches.md).

Installation
------------

See [INSTALL.md](INSTALL.md).

Documentation, tutorials and examples
-------------------------------------

The [documentation](docs/docs/index.md) contains a user guide, tutorials,
theory notes, the API reference and the internal conventions; see
[docs/README.md](docs/README.md) to build it. The tutorial notebooks and
runnable example scripts are in [`examples/`](examples/README.md).

Contributing
------------

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development setup, the rules
for changing the code, the test gates and the release procedure.

Contributors and provenance
---------------------------

QmeQ combines the original project with later RTD, maintenance,
counting-statistics, and tutorial histories. See [AUTHORS.md](AUTHORS.md) for
the authors, source forks, and their specific contributions. Git authorship is
preserved for the complete imported histories.

License
-------

QmeQ has [The BSD 2-Clause License][license] and it can be found
in [LICENSE.md](LICENSE.md).

Citing QmeQ
-----------

Please consider citing QmeQ if the use of this project gives results which lead
to scientific publication:

G. Kiršanskas, J. N. Pedersen, O. Karlström, M. Leijnse, and A. Wacker,
*QmeQ 1.0: An open-source Python package for calculations of transport through
quantum dot devices*, [Comput. Phys. Commun. 221, 317 (2017)][qmeqdoi].

The preprint version of the paper can be found on the
[arXiv.org][qmeqarxiv] server.

[license]: https://opensource.org/licenses/BSD-2-Clause
[qmeqdoi]: https://dx.doi.org/10.1016/j.cpc.2017.07.024
[qmeqarxiv]: https://arxiv.org/abs/1706.10104
