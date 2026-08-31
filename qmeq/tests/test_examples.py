"""Smoke tests that the vendored examples still run against the current API.

Python scripts and notebooks form separate marker groups.  The quick Python
script runs in the normal suite.  Notebooks are excluded by the project-level
pytest configuration because their Jupyter kernels require local sockets; run
them explicitly with ``pytest -m notebook`` in a suitable environment.  The
long-running examples (second-order 2vN / RTD sweeps) are additionally marked
``slow`` and only run with ``pytest --runslow``.
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
# Tutorials 4 and 6 stay `slow`: the first sweeps two dense first-order
# conductance maps, the second runs the 2vN and RTD sweeps and the heat engine.
FAST_NOTEBOOKS = {
    'appendix/00_types.ipynb',
    'appendix/01_symmetries.ipynb',
    'tutorials/01_first_transport_calculation.ipynb',
    'tutorials/02_coulomb_blockade.ipynb',
    'tutorials/03_bias_and_gate_sweeps.ipynb',
    'tutorials/05_energy_and_heat_transport.ipynb',
}

# Not executed, and skipped with this reason rather than dropped, so that the
# report says why. These are publication-figure sweeps, not tests: their cost
# is set by a dense parameter grid, so no timeout makes them pass.
NOT_EXECUTED = {
    'example1c_spinful_single_orbital.py':
        'publication-figure sweep: a 201x201 stability diagram runs about '
        '81000 2vN solves, each iterating seven times over kpnt=2**12 energy '
        'points. It exceeded a 1800 s cap on the compiled backend without '
        'finishing. example1b covers the same 2vN code path on a 101-point '
        'bias trace.',
}

# A backstop against a hang, not a performance budget: with the sweep above out
# of the executed set, the whole compiled run is about fifteen minutes for
# every script and notebook together.
TIMEOUT = 900


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


@pytest.mark.example
@pytest.mark.parametrize('name', _params(_script_ids(), FAST_SCRIPTS))
def test_example_script(name, tmp_path):
    if name in NOT_EXECUTED:
        pytest.skip(NOT_EXECUTED[name])
    pytest.importorskip('matplotlib')
    env = dict(os.environ, MPLBACKEND='Agg')
    # Run in a temp cwd so generated figures/data land there, not in the repo.
    result = subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / name)],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=TIMEOUT,
    )
    assert result.returncode == 0, result.stderr[-3000:]


@pytest.mark.example
@pytest.mark.notebook
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
