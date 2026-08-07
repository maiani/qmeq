# Authors, contributors, and repository provenance

QmeQ is the product of several scientific and software-development histories.
This file records the major contributions that can be verified from the
publication, repository history, and merged forks.

## Scientific foundation and original project

- **Gediminas Kiršanskas** founded and led the original QmeQ implementation,
  including the package architecture and core quantum-master-equation
  approaches. Original repository: [gedaskir/qmeq](https://github.com/gedaskir/qmeq).
- **Johannes N. Pedersen, Oskar Karlström, Martin Leijnse, and Andreas Wacker**
  co-authored the QmeQ 1.0 methods and software paper with Gediminas Kiršanskas:
  *QmeQ 1.0: An open-source Python package for calculations of transport
  through quantum dot devices*, Computer Physics Communications 221, 317
  (2017).

## Major implementation histories

- **Martin Josefsson** implemented the Real Time Diagrammatic (`RTD` and
  `pyRTD`) approaches, their Python/OpenMP-Cython implementations, and later
  support for complex tunnelling amplitudes. This work entered through the
  `real_time_diagrammatics` history merged into
  [gedaskir/qmeq](https://github.com/gedaskir/qmeq).
- **Stephanie Matern** developed the initial pure-Python implementation of the
  Lamb-shift Hamiltonian for the Lindblad approach in
  [materns/qmeq](https://github.com/materns/qmeq). The implementation in this
  repository retains that contribution while adding an explicit transport
  option, compiled-backend support, documentation, and regression tests.
- **Viktor Svensson** maintained
  [cvsvensson/qmeq](https://github.com/cvsvensson/qmeq), contributing modern
  Python compatibility, build and wheel work, CI maintenance, and numerical
  stability fixes. That maintenance history is an ancestor of this repository.
- **Simon Wozny** developed zero-frequency counting statistics for Pauli,
  Lindblad, Redfield, 1vN, and RTD, including the `RTDnoise` implementation,
  consistently fourth-order-truncated RTD cumulants, and later pseudoinverse
  corrections. Source repositories:
  [si8881wo/qmeq](https://github.com/si8881wo/qmeq) and
  [qmeq-noise-example](https://github.com/si8881wo/qmeq-noise-example).
  His complete 60-commit history is preserved in this repository.
- **Athanasios Tsintzis** contributed a quantum-dot indexing fix through
  [atsintzis/qmeq](https://github.com/atsintzis/qmeq).
- **Florido Paganelli** contributed an installation-command correction to the
  original project history.

## Current integration and maintenance

- **Andrea Maiani** maintains
  [maiani/qmeq](https://github.com/maiani/qmeq) as an integration and
  modernization fork.

## Examples and documentation

- The learning material under `examples/` incorporates the historical QmeQ
  example repository and later maintenance and tutorial work.
