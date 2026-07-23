"""
This QmeQ package contains modules, where different approximate master equations
are implemented. Modules starting with `c_` are implemented using Cython.
"""

# For backwards compatibility with 1.0
import sys

from .._backend import load_compiled_modules
from .base import pauli
from .base import lindblad
from .base import redfield
from .base import neumann1
from .base import neumann2
from .base import RTD
sys.modules['qmeq.approach.pauli'] = pauli
sys.modules['qmeq.approach.lindblad'] = lindblad
sys.modules['qmeq.approach.redfield'] = redfield
sys.modules['qmeq.approach.neumann1'] = neumann1
sys.modules['qmeq.approach.neumann2'] = neumann2
sys.modules['qmeq.approach.RTD'] = RTD

_compiled_names = (
    'qmeq.approach.base.c_pauli',
    'qmeq.approach.base.c_lindblad',
    'qmeq.approach.base.c_redfield',
    'qmeq.approach.base.c_neumann1',
    'qmeq.approach.base.c_neumann2',
    'qmeq.approach.base.c_RTD',
)
_compiled = load_compiled_modules('legacy-approach-aliases', _compiled_names)
if _compiled is not None:
    for _name in _compiled_names:
        _alias = _name.replace('.base', '')
        sys.modules[_alias] = _compiled[_name]
