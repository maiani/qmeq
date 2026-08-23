"""
QmeQ: Quantum master equation for Quantum dot transport calculations
====================================================================

QmeQ is an open-source Python package for transport calculations through
quantum  dot devices. The so-called Anderson-type models are used to describe
the quantum dot device, where quantum dots are coupled to the leads by
tunneling. QmeQ can calculate the stationary state particle and energy currents
using various approximate density matrix approaches. As for now we have
implemented the following first-order methods

* Pauli (classical) master equation
* Lindblad approach
* Redfield approach
* First order von Neumann (1vN) approach

which can describe the effect of Coulomb blockade. QmeQ also has one
second-order method

* Second order von Neumann (2vN) approach

which can additionally address cotunneling, pair tunneling, and
broadening effects.

Physics disclaimer
------------------

All the methods in QmeQ are approximate so depending on parameter regime they
can fail, and a good knowledge of the method is required whether to trust the
result or not. For example, Redfield, 1vN, and 2vN approaches can violate
positivity of the reduced density matrix and lead to currents flowing against
the bias. We still think it is important to have a package where a user can
duplicate existing calculations, check applicability of different methods, or
simply discover new kind of physics using different approximate master equations.
"""

from ._backend import BACKEND_ENV as BACKEND_ENV
from ._backend import BackendConfigurationError as BackendConfigurationError
from ._backend import BackendUnavailableError as BackendUnavailableError
from ._backend import get_backend as get_backend
from ._backend import get_backend_status as get_backend_status
from ._backend import get_requested_backend as get_requested_backend
from ._warnings import QmeqRuntimeWarning as QmeqRuntimeWarning
from ._warnings import QmeqWarning as QmeqWarning
from .approach.base.RTD import RTDCoherenceWarning as RTDCoherenceWarning
from .approach.base.RTD import RTDNoBroadeningWarning as RTDNoBroadeningWarning
from .approach.aprclass import Approach as Approach
from .approach.aprclass import ApproachElPh as ApproachElPh
from .approach.aprclass import ApproachBase2vN as ApproachBase2vN
from .builder.builder import Builder as Builder
from .builder.builder_base import BuilderBase as BuilderBase
from .builder.builder_base import BuilderManyBody as BuilderManyBody
from .builder.builder_base import ModelParameters as ModelParameters
from .builder.builder_elph import BuilderElPh as BuilderElPh
from .builder.builder_elph import BuilderManyBodyElPh as BuilderManyBodyElPh
from .builder.funcprop import FunctionProperties as FunctionProperties
from .indexing import StateIndexing as StateIndexing
from .indexing import StateIndexingPauli as StateIndexingPauli
from .indexing import StateIndexingDM as StateIndexingDM
from .indexing import StateIndexingDMc as StateIndexingDMc
from .leadstun import LeadsTunneling as LeadsTunneling
from .baths import PhononBaths as PhononBaths
from .qdot import QuantumDot as QuantumDot

__version__ = '1.2.0.dev1'
