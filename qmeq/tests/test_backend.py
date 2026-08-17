"""Tests for explicit pure-Python and Cython backend selection."""

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

import qmeq

ROOT = Path(__file__).resolve().parents[2]


def _run_python(code, backend):
    env = dict(os.environ, QMEQ_BACKEND=backend)
    return subprocess.run(
        [sys.executable, '-c', code],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )


def test_backend_status_matches_selected_approach():
    status = qmeq.get_backend_status()
    approach_module = qmeq.Builder(nsingle=0).Approach.__module__

    assert status['requested'] in {'auto', 'python', 'cython'}
    assert status['active'] in {'python', 'cython'}
    assert set(status['groups'].values()) == {status['active']}
    if status['active'] == 'cython':
        assert '.c_' in approach_module
    else:
        assert '.c_' not in approach_module


def test_python_backend_is_forced_and_quiet():
    code = """
import json
import qmeq

status = qmeq.get_backend_status()
assert qmeq.Builder(nsingle=0).Approach.__module__ == 'qmeq.approach.base.pauli'
print(json.dumps(status, sort_keys=True))
"""
    result = _run_python(code, 'python')

    assert result.returncode == 0, result.stderr
    assert result.stderr == ''
    status = json.loads(result.stdout)
    assert status['requested'] == 'python'
    assert status['active'] == 'python'
    assert set(status['groups'].values()) == {'python'}


def test_cython_backend_is_required_when_requested():
    code = """
import json
import qmeq

status = qmeq.get_backend_status()
assert qmeq.Builder(nsingle=0).Approach.__module__ == (
    'qmeq.approach.base.c_pauli'
)
print(json.dumps(status, sort_keys=True))
"""
    result = _run_python(code, 'cython')
    extension_available = (
        importlib.util.find_spec('qmeq.approach.base.c_pauli') is not None
    )

    if extension_available:
        assert result.returncode == 0, result.stderr
        status = json.loads(result.stdout)
        assert status['requested'] == 'cython'
        assert status['active'] == 'cython'
        assert set(status['groups'].values()) == {'cython'}
    else:
        assert result.returncode != 0
        assert 'BackendUnavailableError' in result.stderr
        assert 'QMEQ_BACKEND=python' in result.stderr


def test_invalid_backend_fails_before_qmeq_import():
    result = _run_python('import qmeq', 'invalid')

    assert result.returncode != 0
    assert 'BackendConfigurationError' in result.stderr
    assert "expected one of: auto, python, cython" in result.stderr


def test_python_backend_disables_extensions_in_setup():
    if not (ROOT / 'setup.py').is_file():
        pytest.skip('requires the source-tree setup.py')

    code = """
import runpy
import setuptools

setuptools.setup = lambda **kwargs: print(len(kwargs['ext_modules']))
runpy.run_path('setup.py', run_name='__main__')
"""
    result = _run_python(code, 'python')

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == '0'


def _run_setup_python(code, **extra_env):
    env = dict(os.environ, QMEQ_BACKEND='cython', **extra_env)
    return subprocess.run(
        [sys.executable, '-c', code],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )


_REPORT_OPENMP_ARGS = """
import runpy
import sys

module = runpy.run_path('setup.py', run_name='not_main')
print(module['get_openmp_args']())
"""


def test_openmp_off_builds_without_openmp_flags():
    """``QMEQ_OPENMP=off`` must not put any OpenMP flag on the extensions.

    The compiled kernels stay correct without OpenMP -- Cython lowers ``prange``
    to an ordinary loop and ``c_RTD.pyx`` shims the two OpenMP API calls it
    makes -- and the macOS wheels are built this way so they stay installable on
    older macOS releases.
    """
    if not (ROOT / 'setup.py').is_file():
        pytest.skip('requires the source-tree setup.py')

    result = _run_setup_python(_REPORT_OPENMP_ARGS, QMEQ_OPENMP='off')

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == '([], [])'


def test_invalid_openmp_mode_is_rejected():
    if not (ROOT / 'setup.py').is_file():
        pytest.skip('requires the source-tree setup.py')

    result = _run_setup_python(_REPORT_OPENMP_ARGS, QMEQ_OPENMP='sometimes')

    assert result.returncode != 0
    assert 'expected one of: auto, off, on' in result.stderr


@pytest.mark.parametrize('backend', ['auto', 'cython'])
def test_partial_extension_set_never_imports(backend, tmp_path):
    """A half-built extension set must fail, not silently degrade.

    ``build_ext`` aborts on the first extension that fails to compile, but an
    incremental rebuild can still leave freshly built modules next to stale ones
    from an earlier build. Importing such a tree must raise rather than mix
    compiled and pure-Python implementations, in ``auto`` just as much as in
    ``cython``: the fallback is only for a *cleanly* absent extension set.
    """
    victim = importlib.util.find_spec('qmeq.approach.base.c_pauli')
    if victim is None or victim.origin is None:
        pytest.skip('requires the compiled extensions')

    origin = Path(victim.origin)
    hidden = tmp_path / origin.name
    shutil.move(origin, hidden)
    try:
        result = _run_python('import qmeq', backend)
    finally:
        shutil.move(hidden, origin)

    assert result.returncode != 0, result.stdout
    assert 'BackendUnavailableError' in result.stderr
    # The message has to name the module that could not be loaded, otherwise a
    # broken install is undiagnosable from the traceback alone.
    assert 'c_pauli' in result.stderr
