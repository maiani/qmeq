#!/bin/bash
set -e -x

# The compiled extensions use -fopenmp, which Apple's clang doesn't support.
# Symlink in the newest Homebrew GCC as plain "gcc" so it's picked up instead.
# Detected dynamically (rather than pinned to e.g. gcc-10) since the set of
# GCC versions preinstalled on GitHub-hosted macOS runners changes over time,
# and the Homebrew prefix differs between Intel (/usr/local) and Apple
# Silicon (/opt/homebrew) runners.
BREW_PREFIX=$(brew --prefix)
GCC=$(ls "$BREW_PREFIX"/bin/gcc-[0-9]* 2>/dev/null | sort -V | tail -n1)

if [ -z "$GCC" ]; then
    brew install gcc
    GCC=$(ls "$BREW_PREFIX"/bin/gcc-[0-9]* | sort -V | tail -n1)
fi

ln -sf "$GCC" "$BREW_PREFIX/bin/gcc"
gcc --version

# Homebrew's GCC bundles an OpenMP runtime (libgomp) built for the host's own
# macOS version, so the wheel's deployment target must be raised to match it
# or delocate-wheel refuses to repair the wheel later. cibuildwheel's
# CIBW_ENVIRONMENT_MACOS is evaluated by a restricted shell-subset parser that
# does not support pipes inside $(...), so the major-version extraction has
# to happen here instead, with the result handed off through a plain file.
sw_vers -productVersion | cut -d. -f1 > /tmp/qmeq_macosx_deployment_target_major
