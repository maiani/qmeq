import os
import shutil
import subprocess
import sys
import tempfile
import textwrap

import numpy as np

from setuptools import setup, Extension
from setuptools.command.build import build as _build

BACKEND_ENV = 'QMEQ_BACKEND'
VALID_BACKENDS = {'auto', 'python', 'cython'}
OPENMP_ENV = 'QMEQ_OPENMP'
VALID_OPENMP = {'auto', 'on', 'off'}
OPENMP_PREFIX_ENV = 'QMEQ_OPENMP_PREFIX'
CYTHON_COMPILER_DIRECTIVES = {
    'binding': True,
    'language_level': 3,
    'legacy_implicit_noexcept': False,
}

# Only compiling and linking is checked, not running: the failure this guards
# against is an unresolved omp_* symbol, and a probe that has to run would be
# wrong under cross-compilation.
_OPENMP_PROBE_SOURCE = textwrap.dedent("""
    #include <omp.h>
    int main(void) {
        int used = 0;
        #pragma omp parallel reduction(+:used)
        used += 1;
        return (used > 0 && omp_get_max_threads() > 0) ? 0 : 1;
    }
    """)


def get_requested_backend():
    """Return and validate the build backend requested through the environment."""

    value = os.environ.get(BACKEND_ENV, 'auto').strip().lower()
    if value not in VALID_BACKENDS:
        choices = ', '.join(sorted(VALID_BACKENDS))
        raise RuntimeError(
            f"Invalid {BACKEND_ENV}={value!r}; expected one of: {choices}."
        )
    return value


def get_requested_openmp():
    """Return and validate the OpenMP mode requested through the environment."""

    value = os.environ.get(OPENMP_ENV, 'auto').strip().lower()
    if value not in VALID_OPENMP:
        choices = ', '.join(sorted(VALID_OPENMP))
        raise RuntimeError(
            f"Invalid {OPENMP_ENV}={value!r}; expected one of: {choices}."
        )
    return value


def _openmp_prefixes():
    """Return prefixes that may hold an OpenMP runtime's headers and library."""

    prefixes = []
    explicit = os.environ.get(OPENMP_PREFIX_ENV, '').strip()
    if explicit:
        prefixes.append(explicit)
    # Conda environments (and the Conda recipe's llvm-openmp) install omp.h and
    # libomp into the environment root.
    prefixes.append(sys.prefix)
    if sys.platform == 'darwin':
        for formula in ('libomp', 'llvm'):
            try:
                found = subprocess.run(
                    ['brew', '--prefix', formula],
                    capture_output=True, text=True, timeout=30, check=True,
                ).stdout.strip()
            except (OSError, subprocess.SubprocessError):
                continue
            if found:
                prefixes.append(found)
    return [p for p in prefixes if p and os.path.isdir(p)]


def _openmp_candidates():
    """Return ``(compile_args, link_args)`` pairs to try, most standard first.

    The correct flags depend on the compiler, not just the platform: MSVC wants
    ``/openmp``, GCC wants ``-fopenmp`` for both compiling and linking, and
    Apple's clang needs the preprocessor flag passed through plus an explicit
    link against a separately installed ``libomp``.
    """

    if os.name == 'nt':
        # MSVC. Nothing extra is needed at link time.
        return [(['/openmp'], [])]

    candidates = [(['-fopenmp'], ['-fopenmp'])]
    if sys.platform == 'darwin':
        candidates.append((['-Xpreprocessor', '-fopenmp'], ['-lomp']))
        for prefix in _openmp_prefixes():
            include = os.path.join(prefix, 'include')
            lib = os.path.join(prefix, 'lib')
            candidates.append((
                ['-Xpreprocessor', '-fopenmp', f'-I{include}'],
                [f'-L{lib}', f'-Wl,-rpath,{lib}', '-lomp'],
            ))
    return candidates


def _openmp_works(compile_args, link_args):
    """Return whether a probe program compiles and links with these flags."""

    from setuptools._distutils.ccompiler import new_compiler
    from setuptools._distutils.sysconfig import customize_compiler

    tmpdir = tempfile.mkdtemp(prefix='qmeq-openmp-probe-')
    try:
        source = os.path.join(tmpdir, 'probe.c')
        with open(source, 'w') as handle:
            handle.write(_OPENMP_PROBE_SOURCE)

        compiler = new_compiler()
        customize_compiler(compiler)
        # A probe failure is reported by an exception whose type varies with the
        # compiler and the distutils version, so anything raised here is taken
        # to mean "these flags are not usable".
        try:
            # MSVC resolves its toolchain lazily; compile() would do this too,
            # but doing it up front keeps a setup failure from being reported as
            # "OpenMP unavailable". Only MSVCCompiler defines this.
            initialize = getattr(compiler, 'initialize', None)
            if initialize is not None:
                initialize()
            objects = compiler.compile(
                [source], output_dir=tmpdir, extra_postargs=list(compile_args)
            )
            compiler.link_executable(
                objects, 'probe', output_dir=tmpdir,
                extra_postargs=list(link_args),
            )
        except Exception:
            return False
        return True
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def get_openmp_args():
    """Return the OpenMP ``(compile_args, link_args)`` to build with.

    ``QMEQ_OPENMP=off`` builds the extensions without OpenMP; ``on`` requires it
    and fails if no usable flags are found; ``auto`` (the default) probes and
    falls back to a serial build with a warning. A serial build is fully
    functional: Cython lowers ``prange`` to an ordinary loop, and the two
    OpenMP API calls in ``c_RTD.pyx`` are shimmed to their single-thread values.
    """

    requested = get_requested_openmp()
    if requested == 'off':
        print(f'qmeq: {OPENMP_ENV}=off, building the extensions without OpenMP')
        return [], []

    for compile_args, link_args in _openmp_candidates():
        if _openmp_works(compile_args, link_args):
            print(f'qmeq: building with OpenMP ({" ".join(compile_args)})')
            return list(compile_args), list(link_args)

    if requested == 'on':
        raise RuntimeError(
            f'{OPENMP_ENV}=on was requested but no usable OpenMP compiler and '
            'linker flags were found. Install an OpenMP-capable toolchain (on '
            'macOS, Apple clang needs a separate libomp, e.g. '
            f'"brew install libomp"; set {OPENMP_PREFIX_ENV} if it lives '
            f'outside the usual prefixes), or build with {OPENMP_ENV}=off to '
            'get a serial build.'
        )

    print(
        'qmeq: WARNING no usable OpenMP flags found; building the extensions '
        'without OpenMP. The compiled kernels stay correct but run serially. '
        f'Set {OPENMP_ENV}=on to make this a hard error instead.'
    )
    return [], []


class BackendBuild(_build):
    """Keep build products from different backend modes isolated."""

    def finalize_options(self):
        self.build_base = os.path.join(
            self.build_base, get_requested_backend()
        )
        super().finalize_options()


def get_ext_modules():
    """Generate the optional C extensions.

    ``QMEQ_BACKEND=python`` produces a pure-Python installation. Otherwise,
    the ``.pyx`` sources are the canonical extension sources and are always
    cythonized; generated ``*.c`` files are build artifacts and are not
    tracked in the repository.
    """

    if get_requested_backend() == 'python':
        return []

    from Cython.Build import cythonize

    file_list = ['qmeq/approach/c_aprclass.pyx',
                 'qmeq/approach/c_kernel_handler.pyx',
                 # base
                 'qmeq/approach/base/c_pauli.pyx',
                 'qmeq/approach/base/c_lindblad.pyx',
                 'qmeq/approach/base/c_redfield.pyx',
                 'qmeq/approach/base/c_neumann1.pyx',
                 'qmeq/approach/base/c_neumann2.pyx',
                 'qmeq/approach/base/c_RTD.pyx',
                 'qmeq/specfunc/c_specfunc.pyx',
                 # elph
                 'qmeq/approach/elph/c_pauli.pyx',
                 'qmeq/approach/elph/c_lindblad.pyx',
                 'qmeq/approach/elph/c_redfield.pyx',
                 'qmeq/approach/elph/c_neumann1.pyx',
                 'qmeq/specfunc/c_specfunc_elph.pyx',
                 # wrappers
                 'qmeq/wrappers/c_lapack.pyx',
                 'qmeq/wrappers/c_mytypes.pyx',]

    ext = []
    openmp_compile_args, openmp_link_args = get_openmp_args()
    for file_name in file_list:
        module_name = file_name[:-len('.pyx')].replace('/', '.')
        ext.append(
            Extension(
                module_name,
                [file_name],
                include_dirs=[np.get_include()],
                extra_compile_args=list(openmp_compile_args),
                extra_link_args=list(openmp_link_args),
            )
        )

    return cythonize(
        ext,
        build_dir='build/cython',
        compiler_directives=CYTHON_COMPILER_DIRECTIVES,
    )


# Static project metadata lives in pyproject.toml; setup.py only builds the
# Cython/C extension modules. Guarded so that the helpers above can be
# introspected (by the tests, or by hand) without triggering a build; build
# frontends run this file as __main__, so the guard does not affect them.
if __name__ == '__main__':
    setup(
        cmdclass={'build': BackendBuild},
        ext_modules=get_ext_modules(),
    )
