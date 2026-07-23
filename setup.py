import os
import sys
import numpy as np

from setuptools import setup, Extension
from setuptools.command.build import build as _build

BACKEND_ENV = 'QMEQ_BACKEND'
VALID_BACKENDS = {'auto', 'python', 'cython'}


def get_requested_backend():
    """Return and validate the build backend requested through the environment."""

    value = os.environ.get(BACKEND_ENV, 'auto').strip().lower()
    if value not in VALID_BACKENDS:
        choices = ', '.join(sorted(VALID_BACKENDS))
        raise RuntimeError(
            f"Invalid {BACKEND_ENV}={value!r}; expected one of: {choices}."
        )
    return value


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
    already generated ``*.c`` files are reused when present; Cython generates
    them when absent or when the custom ``--cython`` option is supplied.
    """

    if get_requested_backend() == 'python':
        return []

    # Check if *.c files are already there
    file_list = ['qmeq/approach/c_aprclass.c',
                 'qmeq/approach/c_kernel_handler.c',
                 # base
                 'qmeq/approach/base/c_pauli.c',
                 'qmeq/approach/base/c_lindblad.c',
                 'qmeq/approach/base/c_redfield.c',
                 'qmeq/approach/base/c_neumann1.c',
                 'qmeq/approach/base/c_neumann2.c',
                 'qmeq/approach/base/c_RTD.c',
                 'qmeq/specfunc/c_specfunc.c',
                 # elph
                 'qmeq/approach/elph/c_pauli.c',
                 'qmeq/approach/elph/c_lindblad.c',
                 'qmeq/approach/elph/c_redfield.c',
                 'qmeq/approach/elph/c_neumann1.c',
                 'qmeq/specfunc/c_specfunc_elph.c',
                 # wrappers
                 'qmeq/wrappers/c_lapack.c',
                 'qmeq/wrappers/c_mytypes.c',]
    c_files_exist = all([os.path.isfile(f) for f in file_list])

    # Check if --cython option is specified
    if '--cython' in sys.argv:
        use_cython = True
        sys.argv.remove('--cython')
    else:
        use_cython = False

    if c_files_exist and not use_cython:
        cythonize = None
        file_ext = '.c'
        # print('using already Cython generated C files')
    else:
        from Cython.Build import cythonize
        file_ext = '.pyx'
        # print('using cythonize to generate C files')

    ext = []
    openmp_flag = '-fopenmp' if os.name == 'posix' else '/openmp'
    for file_no_ext in file_list:
        file_base = file_no_ext[:-2]
        file_name = file_base + file_ext
        module_name = file_base.replace('/', '.')
        ext.append(
            Extension(
                module_name,
                [file_name],
                include_dirs=[np.get_include()],
                extra_compile_args=[openmp_flag],
                extra_link_args=[openmp_flag],
            )
        )

    cext = ext if cythonize is None else cythonize(ext)
    return cext


# Static project metadata lives in pyproject.toml; setup.py only builds the
# Cython/C extension modules.
setup(
    cmdclass={'build': BackendBuild},
    ext_modules=get_ext_modules(),
)
