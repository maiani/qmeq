QmeQ examples
=============

Learning material for the [QmeQ][QmeQ] package, organized as:

* [`scripts/`][scripts] — short, self-contained `.py` examples, from a minimal
  single-dot calculation to spinful multi-dot models.
* [`tutorials/`][tutorials] — the current step-by-step learning path for
  students new to quantum transport and master equations.
* [`appendix/`][appendix] — reference notebooks on state types and symmetries.
* [`legacy_tutorials/`][legacy] — the original introductory and Real Time
  Diagrammatics notebooks, kept for reference; their material is covered by the
  numbered tutorials.

The current tutorials are:

1. **A first transport calculation** — reservoirs, occupations, current signs,
   and the Pauli rate equation.
2. **Coulomb blockade and many-body states** — the interacting Anderson model
   and charge-state probabilities.
3. **Bias and gate sweeps** — current maps, differential conductance, and
   numerical checks for stability diagrams.
4. **Coherence in a double dot and choosing an approximation** — populations
   versus coherences, Pauli against Lindblad, Redfield, and 1vN, and an
   analytical large-bias check.
5. **Energy and heat transport** — energy and heat currents, conservation laws,
   thermovoltage, tight coupling, and unit conversions.
6. **Cotunnelling and second-order methods** — RTD and 2vN, convergence and
   scaling tests, a quantum-dot heat engine, and many-body input.

The scripts can be run directly with Python, e.g.

```bash
$ python scripts/example0_minimal.py
```

The notebooks require [Jupyter][Jupyter], which (together with Matplotlib) can
be installed with

```bash
$ pip install matplotlib jupyter
```

Then start it from this directory and open `tutorials/`:

```bash
$ jupyter notebook
```

[QmeQ]: https://github.com/gedaskir/qmeq
[Jupyter]: https://jupyter.org
[scripts]: scripts
[appendix]: appendix
[tutorials]: tutorials
[legacy]: legacy_tutorials
