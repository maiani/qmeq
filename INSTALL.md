Installation of QmeQ
====================

QmeQ can be installed through [pip][pip] or by building it from source.
To be able to use and build QmeQ you need to have:

* [Python][Python] 3.10 or newer,
* [Cython][Cython] and a C compiler compatible with it,
* [NumPy][NumPy] package,
* [SciPy][SciPy] package.

Building from source additionally requires [setuptools][setuptools]; these
build dependencies are declared in `pyproject.toml` and are installed
automatically by `pip`.

The [tutorial and examples][examples] are distributed separately in the
[qmeq-examples][examples] repository. Running them requires
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
$ pip install qmeq[test]   # pytest for running the test suite
$ pip install qmeq[docs]   # sphinx and the theme for building the docs
$ pip install qmeq[dev]    # everything above plus cython, build, and twine
```

We note that the binaries **pip** and **python** have to be in the system path.

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

QmeQ contains the [documentation][qmeqdocs] generated from docstrings in the
source code. This documentation can be generated in
**html**, **latex**, and other formats using

* [Sphinx][Sphinx] package,
* [sphinx-rtd-theme][srtdt] Read the Docs Sphinx theme.

To install the above packages, use the `docs` extra

```bash
$ pip install qmeq[docs]
```

For example, to generate the documentation in **html** format run

```bash
$ cd 'path to qmeq source'/docs
$ make html
```

The generated documentation should be in
*'path to qmeq source'/docs/build/html/index.html*

[Python]: https://www.python.org
[Cython]: https://cython.org
[NumPy]: https://numpy.org
[SciPy]: https://scipy.org
[Matplotlib]: https://matplotlib.org
[Jupyter]: https://jupyter.org
[Sphinx]: https://www.sphinx-doc.org
[pytest]: https://docs.pytest.org

[setuptools]: https://setuptools.pypa.io
[pip]: https://pip.pypa.io
[gcc]: https://gcc.gnu.org
[cext]: https://github.com/cython/cython/wiki/CythonExtensionsOnWindows
[srtdt]: https://github.com/readthedocs/sphinx_rtd_theme
[examples]: https://github.com/gedaskir/qmeq-examples

[qmeqdocs]: http://github.com/gedaskir/qmeq/tree/master/docs
[qmeqsrc]: http://github.com/gedaskir/qmeq/archive/master.zip
[qmeqtest]: http://github.com/gedaskir/qmeq/tree/master/qmeq/tests
