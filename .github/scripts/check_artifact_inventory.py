"""Assert that a built sdist and wheel contain what they should, and nothing else.

`twine check` validates metadata rendering, not contents, and installing an
artifact only proves that what *is* there imports. Neither notices a missing
notebook, a stale `qmeq.egg-info/SOURCES.txt` silently overriding `MANIFEST.in`,
or a repository-workflow document leaking into a distribution.

Usage:

    python .github/scripts/check_artifact_inventory.py dist/
"""

from __future__ import annotations

import argparse
import fnmatch
import pathlib
import sys
import tarfile
import zipfile

# (description, pattern) pairs. Patterns match the archive path with the
# leading distribution directory stripped, so they read like repository paths.
SDIST_REQUIRED = [
    ("readme", "README.md"),
    ("install guide", "INSTALL.md"),
    ("licence", "LICENSE.md"),
    ("authors", "AUTHORS.md"),
    ("changelog", "CHANGELOG.md"),
    ("bibliography cited by docstrings", "REFERENCES.md"),
    ("packaging metadata", "pyproject.toml"),
    ("Cython sources", "qmeq/*/*.pyx"),
    ("Cython headers", "qmeq/*/*.pxd"),
    ("test suite", "qmeq/tests/test_*.py"),
    ("reference bundles", "qmeq/tests/data/*/*.npz"),
    ("reference manifests", "qmeq/tests/data/*/*.json"),
    ("tutorial notebooks", "examples/tutorials/*.ipynb"),
    ("example scripts", "examples/scripts/*.py"),
    ("MkDocs tree", "docs/docs/*.md"),
    ("MkDocs configuration", "docs/mkdocs.yml"),
    ("legacy Sphinx tree", "legacy_docs/source/*.rst"),
]

SDIST_FORBIDDEN = [
    ("agent guidance", "AGENTS.md"),
    ("agent guidance alias", "CLAUDE.md"),
    ("issue tracker", "TODO.md"),
    ("development plans", "*_devplan.md"),
    ("MkDocs build output", "docs/site/*"),
    ("Sphinx build output", "legacy_docs/build/*"),
    ("notebook symlink duplicate", "docs/docs/notebooks/*"),
    ("generated example figures", "examples/*.png"),
    ("generated example data", "examples/*.dat"),
    ("compiled extensions", "*.so"),
    ("generated C", "qmeq/*/*.c"),
]

WHEEL_REQUIRED = [
    ("package", "qmeq/__init__.py"),
    ("Cython sources for rebuilds", "qmeq/*/*.pyx"),
    ("reference bundles", "qmeq/tests/data/*/*.npz"),
]

WHEEL_FORBIDDEN = [
    ("vendored examples", "examples/*"),
    ("notebooks", "*.ipynb"),
    ("documentation trees", "docs/*"),
    ("legacy documentation", "legacy_docs/*"),
    ("root documents", "*.md"),
]


def sdist_names(path):
    with tarfile.open(path, "r:gz") as archive:
        names = archive.getnames()
    # strip the leading "qmeq-<version>/" component
    return [n.split("/", 1)[1] for n in names if "/" in n]


def wheel_names(path):
    """Wheel contents, excluding the build backend's own .dist-info metadata.

    `.dist-info/` is generated from `pyproject.toml`, not from `MANIFEST.in`,
    and legitimately contains a copy of the licence under PEP 639. These rules
    are about what the package ships, so that directory is out of scope.
    """
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
    return [n for n in names if ".dist-info/" not in n]


def check(kind, names, required, forbidden):
    failures = []
    for description, pattern in required:
        if not any(fnmatch.fnmatch(n, pattern) for n in names):
            failures.append(f"{kind}: missing {description} ({pattern})")
    for description, pattern in forbidden:
        hits = [n for n in names if fnmatch.fnmatch(n, pattern)]
        if hits:
            shown = ", ".join(sorted(hits)[:4])
            failures.append(f"{kind}: must not ship {description}: {shown}")
    print(f"{kind}: {len(names)} entries, "
          f"{len(required)} required and {len(forbidden)} forbidden rules checked")
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dist", type=pathlib.Path,
                        help="directory holding the built sdist and wheel")
    args = parser.parse_args()

    sdists = sorted(args.dist.glob("*.tar.gz"))
    wheels = sorted(args.dist.glob("*.whl"))
    if len(sdists) != 1 or len(wheels) != 1:
        sys.exit(
            f"expected exactly one sdist and one wheel in {args.dist}, "
            f"found {len(sdists)} and {len(wheels)}"
        )

    failures = check("sdist", sdist_names(sdists[0]), SDIST_REQUIRED, SDIST_FORBIDDEN)
    failures += check("wheel", wheel_names(wheels[0]), WHEEL_REQUIRED, WHEEL_FORBIDDEN)

    if failures:
        print("\nartifact inventory failures:")
        for failure in failures:
            print(f"  - {failure}")
        sys.exit(1)
    print("artifact inventory OK")


if __name__ == "__main__":
    main()
