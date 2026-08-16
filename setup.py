import os
import numpy as np

from setuptools import setup, Extension
from setuptools.command.build import build as _build

BACKEND_ENV = 'QMEQ_BACKEND'
VALID_BACKENDS = {'auto', 'python', 'cython'}
CYTHON_COMPILER_DIRECTIVES = {
    'binding': True,
    'language_level': 3,
    'legacy_implicit_noexcept': False,
}


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
    openmp_flag = '-fopenmp' if os.name == 'posix' else '/openmp'
    for file_name in file_list:
        module_name = file_name[:-len('.pyx')].replace('/', '.')
        ext.append(
            Extension(
                module_name,
                [file_name],
                include_dirs=[np.get_include()],
                extra_compile_args=[openmp_flag],
                extra_link_args=[openmp_flag],
            )
        )

    return cythonize(ext, compiler_directives=CYTHON_COMPILER_DIRECTIVES)


# Static project metadata lives in pyproject.toml; setup.py only builds the
# Cython/C extension modules.
setup(
    cmdclass={'build': BackendBuild},
    ext_modules=get_ext_modules(),
)
