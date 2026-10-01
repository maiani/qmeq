Installation of QmeQ
====================

QmeQ 1.2 is not on PyPI yet. **`pip install qmeq` installs the upstream QmeQ
1.1 from 2021**, which has none of the features described in this
documentation and whose results differ in the cases listed at the top of
[CHANGELOG.md](CHANGELOG.md). Install it in one of these ways instead:

* **Conda or pixi**, from the [prefix.dev][prefix] channel, with compiled,
  OpenMP-enabled packages for Linux (x86-64, aarch64) and Apple Silicon macOS:

  ```bash
  $ pixi add --channel https://prefix.dev/andmai/science qmeq
  $ conda install -c https://prefix.dev/andmai/science qmeq
  ```

* **pip, from a release.** Every [release][releases] carries compiled wheels
  for Python 3.12-3.14 on Linux, macOS and Windows, and the source
  distribution. Download the wheel for your platform and `pip install` it, or
  build a tag from source (this needs a C compiler; see below):

  ```bash
  $ pip install "qmeq @ git+https://github.com/qmeq/qmeq.git@v1.2.0.dev12"
  ```

  A project that must reproduce its results should pin a tag like this rather
  than a branch.

* **From source**, as described below.

To be able to use and build QmeQ you need to have:

* [Python][Python] 3.12 or newer,
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

To install QmeQ from source, go into the [downloaded source][qmeqsrc]
directory and run

```bash
$ pip install .
```

The binaries **pip** and **python** have to be in the system path. To work on
QmeQ itself, follow [CONTRIBUTING.md](CONTRIBUTING.md) instead.

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

Validating an installation
--------------------------

The [tests][qmeqtest] ship inside the installed package and use the
[pytest][pytest] framework. Install the `test` extra from the source directory
(`qmeq[test]` without a path would fetch the upstream 1.1 from PyPI), then
validate an installed build from any directory:

```bash
$ pip install ".[test]"
$ pytest --pyargs qmeq.tests
```

Documentation
-------------

The documentation lives in `docs/`; [docs/README.md](docs/README.md) describes
how to build it.

[Python]: https://www.python.org
[NumPy]: https://numpy.org
[SciPy]: https://scipy.org
[Matplotlib]: https://matplotlib.org
[Jupyter]: https://jupyter.org
[pytest]: https://docs.pytest.org

[setuptools]: https://setuptools.pypa.io
[gcc]: https://gcc.gnu.org
[cext]: https://github.com/cython/cython/wiki/CythonExtensionsOnWindows
[examples]: examples

[prefix]: https://prefix.dev
[releases]: https://github.com/qmeq/qmeq/releases
[qmeqsrc]: https://github.com/qmeq/qmeq/archive/refs/heads/master.zip
[qmeqtest]: https://github.com/qmeq/qmeq/tree/master/qmeq/tests
