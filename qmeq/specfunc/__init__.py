"""
Package that contains modules for various special functions.
"""

from .._backend import load_compiled_modules
from .specfunc import fermi_func
from .specfunc import diff_fermi
from .specfunc import phi
from .specfunc import delta_phi
from .specfunc import diff_phi
from .specfunc import diff2_phi
from .specfunc import bose
from .specfunc import polygamma
from .specfunc import digamma
from .specfunc import integralD
from .specfunc import integralX
from .specfunc import BW_Ozaki
from .specfunc import func_pauli
from .specfunc import func_1vN
from .specfunc import kernel_fredriksen
from .specfunc import hilbert_fredriksen
from .specfunc_elph import Func as pyFunc

_compiled = load_compiled_modules(
    'special-functions',
    ('qmeq.specfunc.c_specfunc', 'qmeq.specfunc.c_specfunc_elph'),
)
if _compiled is None:
    c_fermi_func = fermi_func
    c_diff_fermi = diff_fermi
    c_phi = phi
    c_delta_phi = delta_phi
    c_diff_phi = diff_phi
    c_diff2_phi = diff2_phi
    c_bose = bose
    c_polygamma = polygamma
    c_digamma = digamma
    c_integralD = integralD
    c_integralX = integralX
    c_BW_Ozaki = BW_Ozaki
    c_func_pauli = func_pauli
    c_func_1vN = func_1vN
    Func = pyFunc
else:
    _c_specfunc = _compiled['qmeq.specfunc.c_specfunc']
    c_fermi_func = _c_specfunc.c_fermi_func
    c_diff_fermi = _c_specfunc.c_diff_fermi
    c_phi = _c_specfunc.c_phi
    c_delta_phi = _c_specfunc.c_delta_phi
    c_diff_phi = _c_specfunc.c_diff_phi
    c_diff2_phi = _c_specfunc.c_diff2_phi
    c_bose = _c_specfunc.c_bose
    c_polygamma = _c_specfunc.c_polygamma
    c_digamma = _c_specfunc.c_digamma
    c_integralD = _c_specfunc.c_integralD
    c_integralX = _c_specfunc.c_integralX
    c_BW_Ozaki = _c_specfunc.c_BW_Ozaki
    c_func_pauli = _c_specfunc.c_func_pauli
    c_func_1vN = _c_specfunc.c_func_1vN
    Func = _compiled['qmeq.specfunc.c_specfunc_elph'].Func
