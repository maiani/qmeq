QmeQ: Quantum master equation for Quantum dot transport calculations
====================================================================

QmeQ supports the first two zero-frequency particle-current cumulants through
counting fields for Pauli, Lindblad, Redfield, 1vN, and RTD. Pass a nonempty
iterable such as `countingleads=[0, 2]`; after `solve()`,
`system.current_noise` contains `[current, noise]` for the aggregate counted
leads, while `system.current_noise_matrix` contains their lead-resolved noise
covariance matrix in the order given by `countingleads`. Omitting
`countingleads` leaves counting disabled. See the
[counting-statistics documentation](docs/docs/theory/counting-statistics.md)
for formulas, RTD outputs, conventions, and limitations.

This implementation was developed by Simon Wozny and integrated from his
[QmeQ fork](https://github.com/si8881wo/qmeq). See his
[example notebook](https://github.com/si8881wo/qmeq-noise-example) and
[Emary's counting-statistics formulation](https://arxiv.org/abs/0902.3544).

QmeQ is an open-source Python package for calculations of transport through
quantum  dot devices. The so-called Anderson-type models are used to describe
the quantum dot device, where quantum dots are coupled to the leads by
tunneling. QmeQ can calculate the stationary state **particle** and
**energy currents** using various approximate density matrix approaches. As for
now we have implemented the following first-order methods

* Pauli (classical) master equation
* Lindblad approach
* Redfield approach
* First order von Neumann (1vN) approach

which can describe the effect of Coulomb blockade. Additionally, there is a
possibility to model electron-phonon interaction inside a quantum dot using
the first-order approaches. QmeQ also has two second-order methods

* Second order von Neumann (2vN) approach
* Real Time Diagrammatic (RTD) approach

2vN and RTD approaches can address the effects of cotunneling and pair tunneling.
Additionally, the 2vN approach can describe broadening of quantum dot states.
The advantage of RTD approach is that it requires a lot less memory and
computation time resources.

Physics disclaimer
------------------

All the methods in QmeQ are approximate so depending on parameter regime they
**can fail**, and a good knowledge of the method is required whether to trust
the result or not. For example, Redfield, 1vN, and 2vN approaches can **violate
positivity** of the reduced density matrix and lead to **currents flowing against
the bias**. We still think it is important to have a package where a user can
duplicate existing calculations, check applicability of different methods, or
simply discover new kind of physics using different approximate master equations.

Installation
------------

For installation instructions see [INSTALL.md](INSTALL.md).

Development checks
------------------

Install the development tools with `pip install -e '.[dev]'`. Run the
zero-debt correctness lint before submitting changes:

```bash
ruff check .
```

Repository-wide automatic formatting is not enabled yet.

Tutorial & Examples
-------------------

For an introduction to QmeQ see the [tutorials][tutorial] and the runnable
[example scripts][examples] in the [`examples/`][examples] directory. The
notebooks are also rendered in the [documentation][qmeqdocs].

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

[tutorial]: examples/tutorials
[examples]: examples
[qmeqdocs]: docs/
[license]: https://opensource.org/licenses/BSD-2-Clause
[qmeqdoi]: https://dx.doi.org/10.1016/j.cpc.2017.07.024
[qmeqarxiv]: https://arxiv.org/abs/1706.10104
