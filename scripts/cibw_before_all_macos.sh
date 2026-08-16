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
