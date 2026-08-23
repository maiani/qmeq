# Before you begin

The tutorials are written for master's students who know elementary quantum
mechanics but may be new to quantum transport, open quantum systems, and master
equations. Each notebook introduces the physics it needs before using QmeQ.

## Prerequisites

You should be comfortable with:

* complex numbers, matrices, and eigenstates;
* creation and annihilation operators at an introductory level;
* Python variables, functions, loops, and NumPy arrays.

No prior knowledge of density-matrix approximations is assumed. The tutorials
explain why an approximation is introduced, which effects it retains, and
which conclusions it cannot support.

## Install the learning environment

For a source checkout, an editable pure-Python installation is the simplest
starting point:

```console
QMEQ_BACKEND=python python -m pip install -e '.[test]'
python -m pip install matplotlib jupyter
jupyter notebook examples/tutorials
```

The compiled backend is optional. It produces the same physical
approximations more quickly, but it is not necessary for the foundational
tutorials.

## Conventions used throughout

The tutorials use natural units

$$
\hbar = k_\mathrm{B} = |e| = 1.
$$

Consequently, energies, temperatures, chemical potentials, voltages, and
tunnelling rates are expressed relative to one chosen energy scale. QmeQ
reports **particle current**, positive when particles flow from a lead into the
dot. Electrical current requires restoring the carrier charge. The tutorials
state the convention again where its sign matters.

QmeQ computes stationary transport through interacting quantum-dot models
using approximate master equations. Different approximations may disagree
without either implementation being defective: they retain different physical
processes and have different regimes of validity. Never select an approach
only because it gives the most convenient result.

## How to use a tutorial

For each notebook:

1. Read the qualitative prediction before running the calculation.
2. Execute the cells in order.
3. Compare the result with the prediction.
4. Keep the normalization, conservation, equilibrium, and convergence checks.
5. Read the validity section before adapting the model to research parameters.

The notebooks use small grids so they execute quickly. Increase resolution
only after reproducing the supplied checks.
