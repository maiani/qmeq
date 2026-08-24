"""Fail if an installed wheel did not get a working compiled backend.

``CIBW_TEST_COMMAND`` runs with ``QMEQ_BACKEND`` unset, which means ``auto``.
Auto falls back to pure Python whenever the extension set is cleanly absent,
so a wheel whose extensions failed to build, or were never included, passes an
ordinary test run without complaint. This forces ``cython`` before the first
import, so a missing or broken extension raises instead of degrading, and then
asserts the backend actually in use.

Run against the *installed* package, never the source tree.
"""

import os
import sys

os.environ["QMEQ_BACKEND"] = "cython"

import qmeq  # noqa: E402  (must follow the environment assignment)

status = qmeq.get_backend_status()
active = status.get("active")

print(f"qmeq {qmeq.__version__} from {os.path.dirname(qmeq.__file__)}")
print(f"backend status: {status}")

if active != "cython":
    sys.exit(
        f"wheel did not provide a working compiled backend: active={active!r}. "
        "The extensions are missing or failed to import."
    )

print("compiled backend confirmed")
