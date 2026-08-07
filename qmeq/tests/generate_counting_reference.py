"""Print counting references from Simon Wozny's pristine source checkout.

This is a provenance tool, not part of the test suite.  Run it with the
working directory set to an untouched checkout of
``si8881wo/qmeq@aa1af46dd687c271505d28dbfb7ccce03a8a1739``.  The committed
values in :mod:`qmeq.tests.data_counting` were generated under Python 3.13
with NumPy 2.2.6 and SciPy 1.15.3.
"""

import subprocess

import numpy as np

import qmeq


SOURCE_COMMIT = "aa1af46dd687c271505d28dbfb7ccce03a8a1739"


def _require_source_commit():
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != SOURCE_COMMIT:
        raise RuntimeError(
            f"Run from the pristine Simon checkout at {SOURCE_COMMIT}; "
            f"found {commit}."
        )


def _first_order(kerntype, gate, bias):
    return qmeq.Builder(
        nsingle=2,
        hsingle={(0, 0): -10 + gate, (1, 1): -12 + gate, (0, 1): 20},
        coulomb={(0, 1, 1, 0): 30},
        nleads=2,
        tleads={(0, 0): 2.0, (1, 1): 1.0, (0, 1): 0.6, (1, 0): 0.1},
        mulst={0: bias / 2, 1: -bias / 2},
        tlst={0: 25.0, 1: 25.0},
        dband={0: 1000.0, 1: 1000.0},
        kerntype=f"py{kerntype}",
        itype=2,
        countingleads=[0],
    )


def _rtd(gate, bias):
    system = qmeq.Builder(
        nsingle=1,
        hsingle={(0, 0): gate},
        nleads=2,
        tleads={
            (0, 0): np.sqrt(0.08 / (2 * np.pi)),
            (1, 0): np.sqrt(0.12 / (2 * np.pi)),
        },
        mulst={0: bias / 2, 1: -bias / 2},
        tlst={0: 1.0, 1: 1.0},
        dband={0: 100.0, 1: 100.0},
        kerntype="pyRTDnoise",
        countingleads=[0],
    )
    system.off_diag_corrections = False
    return system


def main():
    _require_source_commit()
    np.set_printoptions(precision=17)
    for kerntype in ("Pauli", "Lindblad", "Redfield", "1vN"):
        print(kerntype)
        for gate, bias in ((-15.0, 4.0), (0.0, 5.0), (18.0, 12.0)):
            system = _first_order(kerntype, gate, bias)
            system.solve()
            print(repr(system.current_noise))
    for gate, bias in ((-1.5, 4.0), (0.0, 6.0), (1.25, 8.0)):
        system = _rtd(gate, bias)
        system.solve()
        print(repr(system.current_noise))
        print(repr(system.appr.current_noise_first))
        print(repr(system.appr.current_noise_o4trunc))


if __name__ == "__main__":
    main()
