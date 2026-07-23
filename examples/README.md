QmeQ examples
=============

Learning material for the [QmeQ][QmeQ] package, organized as:

* [`scripts/`][scripts] — short, self-contained `.py` examples, from a minimal
  single-dot calculation to spinful multi-dot models.
* [`tutorial/`][tutorial] — a guided [Jupyter][Jupyter] walkthrough
  ([`tutorial.ipynb`][tutorial]) and a Real Time Diagrammatics tutorial
  ([`RTD_tutorial.ipynb`][rtd]).
* [`appendix/`][appendix] — reference notebooks on state types and symmetries.

The scripts can be run directly with Python, e.g.

```bash
$ python scripts/example0_minimal.py
```

The notebooks require [Jupyter][Jupyter], which (together with Matplotlib) can
be installed with

```bash
$ pip install matplotlib jupyter
```

Then start it from this directory and open a notebook:

```bash
$ jupyter notebook
```

[QmeQ]: https://github.com/gedaskir/qmeq
[Jupyter]: https://jupyter.org
[scripts]: scripts
[appendix]: appendix
[tutorial]: tutorial/tutorial.ipynb
[rtd]: tutorial/RTD_tutorial.ipynb
