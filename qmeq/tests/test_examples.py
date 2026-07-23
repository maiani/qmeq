"""Smoke tests that the vendored examples still run against the current API.

The quick examples run as part of the normal suite; the long-running ones
(second-order 2vN / RTD sweeps) are marked ``slow`` and only run with
``pytest --runslow``.  All example tests need Matplotlib (and, for notebooks, a
Jupyter ``python3`` kernel) and are skipped when those are unavailable.
"""
import os
import sys
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / 'examples' / 'scripts'
EXAMPLES_DIR = ROOT / 'examples'

# Examples fast enough to run in the default suite; everything else is `slow`.
FAST_SCRIPTS = {'example0_minimal.py'}
FAST_NOTEBOOKS = {'appendix/00_types.ipynb', 'appendix/01_symmetries.ipynb'}

# Generous ceiling so the heavy sweeps do not fail spuriously under --runslow.
TIMEOUT = 1800


def _params(paths, fast):
    out = []
    for p in paths:
        marks = () if p in fast else (pytest.mark.slow,)
        out.append(pytest.param(p, marks=marks, id=p))
    return out


def _script_ids():
    if not SCRIPTS_DIR.is_dir():
        return []
    return sorted(p.name for p in SCRIPTS_DIR.glob('example*.py'))


def _notebook_ids():
    if not EXAMPLES_DIR.is_dir():
        return []
    return sorted(
        p.relative_to(EXAMPLES_DIR).as_posix()
        for p in EXAMPLES_DIR.glob('**/*.ipynb')
    )


@pytest.mark.parametrize('name', _params(_script_ids(), FAST_SCRIPTS))
def test_example_script(name, tmp_path):
    pytest.importorskip('matplotlib')
    env = dict(os.environ, MPLBACKEND='Agg')
    # Run in a temp cwd so generated figures/data land there, not in the repo.
    result = subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / name)],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=TIMEOUT,
    )
    assert result.returncode == 0, result.stderr[-3000:]


@pytest.mark.parametrize('name', _params(_notebook_ids(), FAST_NOTEBOOKS))
def test_example_notebook(name):
    pytest.importorskip('matplotlib')
    nbformat = pytest.importorskip('nbformat')
    pytest.importorskip('nbclient')
    from nbclient import NotebookClient
    from jupyter_client.kernelspec import KernelSpecManager

    if 'python3' not in KernelSpecManager().find_kernel_specs():
        pytest.skip("no 'python3' Jupyter kernel available")

    path = EXAMPLES_DIR / name
    nb = nbformat.read(str(path), as_version=4)
    client = NotebookClient(
        nb, timeout=TIMEOUT, kernel_name='python3',
        resources={'metadata': {'path': str(path.parent)}},
    )
    client.execute()
