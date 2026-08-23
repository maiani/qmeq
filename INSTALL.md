Installation of QmeQ
====================

QmeQ can be installed through [pip][pip] or by building it from source.
To be able to use and build QmeQ you need to have:

* [Python][Python] 3.11 or newer,
* [NumPy][NumPy] package,
* [SciPy][SciPy] package.

Building the compiled backend from source additionally requires Cython 3
(`>=3.0,<4`), [setuptools][setuptools], and a compatible C compiler. The Python
build dependencies are declared in `pyproject.toml` and are installed
automatically by `pip`.

OpenMP is optional. By default the build detects whether the compiler can use
it and falls back to a serial build (with a warning) when it cannot; set
`QMEQ_OPENMP=on` to turn that fallback into an error, or `QMEQ_OPENMP=off` to
skip OpenMP outright. A serial build gives the same results, but the RTD and
2vN kernels do not run in parallel. Apple's clang has no OpenMP runtime of its
own, so on macOS you need `brew install libomp` (point `QMEQ_OPENMP_PREFIX` at
it if it is installed somewhere unusual) to build with OpenMP.

Note that the published macOS **wheels** are built serially, so that they stay
installable on older macOS releases; the Conda packages are built with OpenMP.
For threaded kernels on macOS, use the Conda package or build from source with
`QMEQ_OPENMP=on`.

The tutorial and [examples][examples] are included in the `examples/` directory
and are also rendered in the documentation. Running the notebooks requires
[Matplotlib][Matplotlib] and [Jupyter][Jupyter], which can be installed with

```bash
$ pip install matplotlib jupyter
```

To install QmeQ through pip run

```bash
$ pip install qmeq
```

or by going into the [downloaded source][qmeqsrc] directory and running

```bash
$ pip install .
```

To work on QmeQ itself, install it in editable mode instead

```bash
$ pip install -e .
```

Optional feature sets are available as extras and can be requested in brackets;
they combine with any of the commands above (e.g. `pip install -e .[dev]`):

```bash
$ pip install qmeq[test]         # pytest for running the test suite
$ pip install qmeq[docs]         # mkdocs + material, for the docs/ site
$ pip install qmeq[docs-sphinx]  # sphinx + theme, for the legacy legacy_docs/ manual
$ pip install qmeq[dev]          # everything above plus cython, build, and twine
```

We note that the binaries **pip** and **python** have to be in the system path.

Backend selection
-----------------

QmeQ supports pure-Python and compiled Cython implementations. Set the
`QMEQ_BACKEND` environment variable before installing or importing QmeQ:

* `auto` (default) uses Cython when the complete extension set is available and
  otherwise uses Python.
* `python` forces the pure-Python implementation and skips extension builds.
* `cython` requires compiled extensions and fails clearly when they cannot be
  built or imported.

For example, on Linux or macOS:

```bash
$ QMEQ_BACKEND=python pip install .
$ QMEQ_BACKEND=python python calculation.py
$ QMEQ_BACKEND=cython python calculation.py
```

In PowerShell, set the variable with
`$env:QMEQ_BACKEND = "python"` before running the corresponding command.
The value is read when QmeQ is first imported and cannot be changed for an
already imported process.

The selected backend and its component groups can be included in bug reports:

```python
import qmeq
print(qmeq.get_backend_status())
```

In `auto` mode QmeQ falls back only when extensions are absent. A broken or
partially installed extension set is reported as an error rather than hidden
by the fallback.

C compiler
----------

For **Linux** and **Mac** we recommend to use the C compiler in the conventional
[gcc][gcc] suite, which will be recognized by Cython. For **Windows** the
**Visual Studio** or **Windows SDK C/C++** compiler can be used and more
instructions how to setup these compilers to work with Cython are available
[here][cext].

NumPy and OpenBLAS/MKL
----------------------

For a good performance of the calculations NumPy needs to be linked to an
optimized BLAS/LAPACK library such as OpenBLAS or MKL. The NumPy and SciPy
wheels installed by `pip` already bundle OpenBLAS on all major platforms, so
in most cases no extra setup is needed. To inspect the linked library open a
Python interpreter and write

```python
import numpy
numpy.show_config()
```

and check the reported **blas** / **lapack** backend.

Tests
-----

To run the [tests][qmeqtest] included with QmeQ we use

* [pytest][pytest] testing framework.

To install it, use the `test` extra

```bash
$ pip install qmeq[test]
```

From the source directory the tests can be performed by calling

```bash
$ pytest
```

The tests are also shipped inside the installed package, so a compiled,
installed build can be validated from any directory with

```bash
$ pytest --pyargs qmeq.tests
```

Documentation
-------------

QmeQ ships two documentation trees while it migrates from one to the other:

* `docs/` — the current documentation, built with [MkDocs][MkDocs] and the
  [Material][mkdocs-material] theme. It is **incomplete**: the user manual,
  API reference, theory pages, and examples have not all been migrated here
  yet. Install its dependencies with the `docs` extra and build with

  ```bash
  $ pip install qmeq[docs]
  $ mkdocs build --strict -f 'path to qmeq source'/docs/mkdocs.yml
  ```

  The generated site lands in *'path to qmeq source'/docs/site/index.html*.

* `legacy_docs/` — the legacy [Sphinx][Sphinx] manual (using the
  [sphinx-rtd-theme][srtdt]), still the **complete** user manual, API
  reference, theory notes, and worked examples until the migration above
  finishes. Install its dependencies with the `docs-sphinx` extra and build
  with

  ```bash
  $ pip install qmeq[docs-sphinx]
  $ cd 'path to qmeq source'/legacy_docs
  $ make html
  ```

  The generated documentation lands in
  *'path to qmeq source'/legacy_docs/build/html/index.html*.

If neither build is at hand, the docstrings in the source code are the ground
truth either way.

[Python]: https://www.python.org
[Cython]: https://cython.org
[NumPy]: https://numpy.org
[SciPy]: https://scipy.org
[Matplotlib]: https://matplotlib.org
[Jupyter]: https://jupyter.org
[Sphinx]: https://www.sphinx-doc.org
[MkDocs]: https://www.mkdocs.org
[mkdocs-material]: https://squidfunk.github.io/mkdocs-material/
[pytest]: https://docs.pytest.org

[setuptools]: https://setuptools.pypa.io
[pip]: https://pip.pypa.io
[gcc]: https://gcc.gnu.org
[cext]: https://github.com/cython/cython/wiki/CythonExtensionsOnWindows
[srtdt]: https://github.com/readthedocs/sphinx_rtd_theme
[examples]: examples

[qmeqsrc]: http://github.com/gedaskir/qmeq/archive/master.zip
[qmeqtest]: http://github.com/gedaskir/qmeq/tree/master/qmeq/tests
