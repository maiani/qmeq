"""Remove build artifacts and generated files from the repository.

Two kinds of artifact are removed, and they have very different costs:

* **caches** -- ``build/``, ``dist/``, ``*.egg-info``, ``__pycache__``, and the
  test caches. Cheap to regenerate, and stale ones cause real confusion: a
  leftover ``qmeq.egg-info/SOURCES.txt`` silently overrides ``MANIFEST.in``, so
  an sdist keeps shipping files the manifest no longer lists.
* **compiled extensions** -- the in-place ``.so``/``.c``/``.o`` files produced
  by ``setup.py build_ext --inplace``. Removing these costs a rebuild, and
  until you do, ``QMEQ_BACKEND=cython`` fails and ``auto`` quietly falls back
  to pure Python.

The default removes both, as it always has. Use ``--caches-only`` to clear the
caches while keeping a working compiled backend. ``--dry-run`` lists targets
without deleting. Paths resolve relative to this script, not the working
directory.
"""

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Top-level build/cache directories.
TOP_LEVEL_DIRS = [
    ROOT / '.cache',
    ROOT / '.pytest_cache',
    ROOT / 'build',
    ROOT / 'dist',
    ROOT / 'legacy_docs' / 'build',
    ROOT / 'docs' / 'site',
    ROOT / 'qmeq.egg-info',
    ROOT / 'qmeq' / 'build',
]

# Generated Cython/build artifacts, wherever they occur under qmeq/.
GENERATED_FILE_PATTERNS = [
    '*.o', '*.so', '*.pyd', '*.dll', '*.pyc', '*.c', '*.html',
]

# Of those, the ones whose removal costs a full extension rebuild.
EXTENSION_SUFFIXES = {'.o', '.so', '.pyd', '.dll', '.c'}


def find_targets():
    """Return the directories and files this script would remove."""

    dirs = [d for d in TOP_LEVEL_DIRS if d.is_dir()]
    dirs += [p for p in (ROOT / 'qmeq').rglob('__pycache__') if p.is_dir()]

    files = []
    for pattern in GENERATED_FILE_PATTERNS:
        files += (ROOT / 'qmeq').rglob(pattern)
    # Drop files that live inside a directory already slated for removal
    # (e.g. *.pyc under a __pycache__/ that rmtree will take with it).
    files = [f for f in files if not any(d in f.parents for d in dirs)]

    return dirs, files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--dry-run', action='store_true',
        help='List what would be removed without deleting anything.',
    )
    parser.add_argument(
        '--caches-only', action='store_true',
        help=('Remove build and test caches but keep the in-place compiled '
              'extensions, so the Cython backend keeps working.'),
    )
    args = parser.parse_args()

    dirs, files = find_targets()
    if args.caches_only:
        files = [f for f in files if f.suffix not in EXTENSION_SUFFIXES]

    for d in dirs:
        print(('would remove' if args.dry_run else 'removing') + f' directory {d}')
        if not args.dry_run:
            shutil.rmtree(d)

    for f in files:
        print(('would remove' if args.dry_run else 'removing') + f' file {f}')
        if not args.dry_run:
            f.unlink()

    removed = sum(1 for f in files if f.suffix in EXTENSION_SUFFIXES)
    if removed and not args.dry_run:
        print(
            f'\nRemoved {removed} compiled artifacts. Rebuild with '
            '"python setup.py build_ext --inplace" before using '
            'QMEQ_BACKEND=cython; until then "auto" falls back to pure Python.'
        )


if __name__ == '__main__':
    main()
