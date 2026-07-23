"""Tests for explicit pure-Python and Cython backend selection."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

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
    code = """
import runpy
import setuptools

setuptools.setup = lambda **kwargs: print(len(kwargs['ext_modules']))
runpy.run_path('setup.py', run_name='__main__')
"""
    result = _run_python(code, 'python')

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == '0'
