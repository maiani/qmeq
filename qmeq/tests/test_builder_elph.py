from numpy import sqrt, exp, pi
from numpy.linalg import norm
import itertools

from qmeq import BuilderElPh
from qmeq import ModelParameters
from qmeq.specfunc import Func
from qmeq.tests.reference_data import load_reference_bundle
from qmeq.tests.test_builder import Calcs

EPS = 1e-10
CHECK_PY = False
PRNTQ = False

_LEGACY_BUNDLE = load_reference_bundle("legacy")
LEGACY_BUILDER_ELPH_REFERENCE = _LEGACY_BUNDLE.resolve(
    _LEGACY_BUNDLE.manifest["snapshots"]["builder_elph"]
)


class JFunc(Func):
    def eval(self, E):
        return 3.8804e-4*E


class SpinfulDoubleDotWithElPh(BuilderElPh):

    def __init__(self,
                 # Dot parameters
                 eL=0.05,
                 eR=-0.05,
                 omega=0.05,
                 U=12,
                 Un=2.5,
                 # Tunneling and lead parameters
                 gamL=9.0e-5,
                 gamR=9.0e-5,
                 tempL=0.005,
                 tempR=0.005,
                 vbias=0.0,
                 dband=50.0,
                 # Phonon baths and electron-phonon coupling parameters
                 tempPh=0.025,
                 d=120,
                 a=5.8,
                 alpha=pi/3*1j,
                 bath_func=[JFunc()],
                 dband_ph_min=1e-8,
                 dband_ph_max=100,
                 # Master equation parameters
                 kerntype='Pauli',
                 itype=0,
                 itype_ph=0,
                 principal_part=None,
                 indexing='ssq',
                 symq=True,
                 mfreeq=False):
        """Initialization of the Model class."""

        self.p = ModelParameters(locals())
        self.p.delta = eL-eR

        nsingle = 4
        hsingle = {(0,0): eL, (1,1): eR, (0,1): omega}
        coulomb = {(0,0,0,0): U,
                   (1,1,1,1): U,
                   (0,1,1,0): Un}

        nleads = 4
        tL, tR = sqrt(gamL/(2*pi)), sqrt(gamR/(2*pi))
        tleads = {(0,0): tL, (1,1): tR}
        mulst = {0: vbias/2, 1: -vbias/2}
        tlst =  {0: tempL,   1: tempR}

        nbaths = 1
        dband_ph = {0: [dband_ph_min, dband_ph_max]}
        tlst_ph = {0: tempPh}

        yelph = {(0,0,0): 1.0,
                 (0,1,1): exp(alpha),
                 (0,0,1): exp(alpha/2)*exp(-(d**2)/(4*a**2)),
                 (0,1,0): exp(alpha/2)*exp(-(d**2)/(4*a**2))}


        # Initialise the system
        BuilderElPh.__init__(self,
            nsingle, hsingle, coulomb,
            nleads, tleads, mulst, tlst, dband,
            nbaths, yelph, tlst_ph, dband_ph,
            bath_func=bath_func,
            kerntype=kerntype,
            itype=itype,
            itype_ph=itype_ph,
            principal_part=principal_part,
            symq=symq,
            mfreeq=mfreeq,
            indexing=indexing,
            symmetry='spin')

    # ------------------------------------------------

    # Detuning
    def get_delta(self):
        return self.p.delta

    def set_delta(self, value):
        self.eL, self.eR = +value/2, -value/2
        self.change(hsingle={(0,0): self.eL,
                             (1,1): self.eR})
        self.p.delta = value
    delta = property(get_delta, set_delta)

    # ------------------------------------------------


def test_Builder_elph_double_dot_spinful():
    data = LEGACY_BUILDER_ELPH_REFERENCE
    calcs = Calcs()

    # Check if the results agree with previously calculated data
    kerns = ['Pauli', 'Redfield', '1vN', 'Lindblad']
    kerns += ['pyPauli', 'pyRedfield', 'py1vN', 'pyLindblad'] if CHECK_PY else []
    itypes, itypes_ph = [0, 1, 2], [0, 2]
    repetitions = 3
    for kerntype, itype, itype_ph in itertools.product(kerns, itypes, itypes_ph):
        if kerntype in {'Pauli', 'pyPauli', 'Lindblad', 'pyLindblad'} and (itype in [0, 1] or itype_ph in [0]):
            continue

        system = SpinfulDoubleDotWithElPh(kerntype=kerntype, itype=itype, itype_ph=itype_ph)

        for i in range(repetitions):
            system.solve()

            attr = kerntype+str(itype)+str(itype_ph)
            setattr(calcs, attr, system)

            if PRNTQ:
                print('kerntype - ', kerntype, 'itype - ', itype, 'repetition - ', i)
                print('current')
                print(system.current)
                print( data[attr+'current'] )
                print('energy_current')
                print(system.energy_current)
                print( data[attr+'energy_current'] )
                print('differences:')
                print( norm(system.current - data[attr+'current']) )
                print( norm(system.energy_current - data[attr+'energy_current']) )

            for param in ['current', 'energy_current']:
                assert norm(getattr(system, param) - data[attr+param]) < EPS

        # Check least-squares solution with non-square matrix, i.e., symq=False
        system = SpinfulDoubleDotWithElPh(kerntype=kerntype, itype=itype, itype_ph=itype_ph, symq=False)

        for i in range(repetitions):
            system.solve()
            for param in ['current', 'energy_current']:
                assert norm(getattr(system, param) - data[attr+param]) < EPS

    # Check matrix-free methods
    for kerntype in kerns:
        itype, itype_ph = 2, 2
        system = SpinfulDoubleDotWithElPh(kerntype=kerntype, itype=itype, itype_ph=itype_ph, mfreeq=True)

        for i in range(repetitions):
            system.solve()
            attr = kerntype+str(itype)+str(itype_ph)
            for param in ['current', 'energy_current']:
                assert norm(getattr(system, param) - data[attr+param]) < 1e-4

    # Check results with different indexing
    indexings = ['Lin', 'charge', 'sz', 'ssq']
    for kerntype, indexing in itertools.product(kerns, indexings):
        itype, itype_ph = 2, 2
        system = SpinfulDoubleDotWithElPh(kerntype=kerntype, itype=itype, itype_ph=itype_ph, indexing=indexing)

        for i in range(repetitions):
            system.solve()
            attr = kerntype+str(itype)+str(itype_ph)
            for param in ['current', 'energy_current']:
                assert norm(getattr(system, param) - data[attr+param]) < EPS


def test_every_builder_runs_the_pre_approach_hook():
    """``_init_before_appr`` must fire for every builder.

    ``BuilderElPh.__init__`` repeats ``BuilderBase.__init__``'s sequence rather
    than delegating to it, so the hook call is easy to drop there. When that
    happens the many-body state indexing lands after the ``Approach`` object is
    built, and a compiled approach that sizes buffers from ``si`` at
    construction time (RTD) gets the wrong size -- see
    ``test_RTD_many_body_construction_matches_Builder``.
    """
    import numpy as np

    from qmeq import Builder
    from qmeq import BuilderElPh
    from qmeq import BuilderManyBody
    from qmeq import BuilderManyBodyElPh

    hook_ran = []

    common = dict(mulst=[0.0, 0.0], tlst=[1.0, 1.0], dband=100.0,
                  kerntype='pyPauli')
    elph = dict(tlst_ph=[1.0], dband_ph=[100.0])
    many_body = dict(Ea=np.array([0.0, 1.0, 1.2, 2.5]), Na=[0, 1, 1, 2],
                     Tba=np.zeros((2, 4, 4), dtype=complex))
    Vbbp = np.zeros((1, 4, 4), dtype=complex)

    cases = [
        (Builder, dict(nsingle=1, nleads=2, tleads={(0, 0): 0.1}, **common)),
        (BuilderElPh, dict(nsingle=1, nleads=2, tleads={(0, 0): 0.1},
                           nbaths=1, velph={(0, 0, 0): 0.1},
                           **common, **elph)),
        (BuilderManyBody, dict(**many_body, **common)),
        (BuilderManyBodyElPh, dict(**many_body, Vbbp=Vbbp, **common, **elph)),
    ]

    for cls, kwargs in cases:
        # Restore by putting the original back when the class defined its own,
        # and only delete when the attribute was inherited. Deleting
        # unconditionally would strip the real override off, say,
        # BuilderManyBody and silently disable it for every later test.
        defines_its_own = '_init_before_appr' in cls.__dict__
        original = cls._init_before_appr

        def spy(self, _cls=cls, _original=original):
            hook_ran.append(_cls.__name__)
            return _original(self)

        cls._init_before_appr = spy
        try:
            cls(**kwargs)
        finally:
            if defines_its_own:
                cls._init_before_appr = original
            else:
                del cls._init_before_appr

    assert hook_ran == ['Builder', 'BuilderElPh',
                        'BuilderManyBody', 'BuilderManyBodyElPh']
