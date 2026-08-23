"""Remove build artifacts and generated files from the repository.

Run with ``--dry-run`` to list what would be removed without deleting
anything. All paths are resolved relative to this script's location, not the
current working directory, so the cleanup targets stay the same regardless of
where the script is invoked from.
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
    args = parser.parse_args()

    dirs, files = find_targets()

    for d in dirs:
        print(('would remove' if args.dry_run else 'removing') + f' directory {d}')
        if not args.dry_run:
            shutil.rmtree(d)

    for f in files:
        print(('would remove' if args.dry_run else 'removing') + f' file {f}')
        if not args.dry_run:
            f.unlink()


if __name__ == '__main__':
    main()
