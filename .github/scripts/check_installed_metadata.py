"""Assert an installed qmeq's distribution metadata agrees with the package.

The version is declared once, as ``qmeq.__version__``, and read from there by
``pyproject.toml``'s dynamic version. If the two ever disagree in an installed
artifact, ``build_wheels.yml``'s tag check would compare a tag against one of
them while users see the other. Cheap to assert, and it runs against the
installed copy where the skew would actually appear.
"""

import importlib.metadata as metadata
import sys

import qmeq

failures = []

declared = metadata.version("qmeq")
if declared != qmeq.__version__:
    failures.append(
        f"distribution version {declared!r} != qmeq.__version__ "
        f"{qmeq.__version__!r}"
    )

fields = metadata.metadata("qmeq")
if fields["Name"].lower().replace("_", "-") != "qmeq":
    failures.append(f"unexpected distribution name {fields['Name']!r}")

requires = fields["Requires-Python"]
if not requires or "3.11" not in requires:
    failures.append(f"unexpected Requires-Python {requires!r}")

print(f"qmeq {declared} | Requires-Python {requires}")

if failures:
    print("installed metadata failures:")
    for failure in failures:
        print(f"  - {failure}")
    sys.exit(1)
print("installed metadata OK")
