# Maintenance TODO

## High priority

- [ ] Modernize continuous integration.
  - Run tests for pull requests and branch pushes, not only tagged releases.
  - Test the actively supported Python versions on Linux, macOS, and Windows.
  - Update the GitHub Actions and cibuildwheel versions after checking their
    current supported releases.
  - Build the documentation with warnings treated as errors.

- [ ] Build and test the Cython extensions in CI.
  - Exercise both the compiled implementations and the pure-Python fallbacks.
  - Verify that source builds and wheels work on every supported Python version.
  - Ensure tests do not silently cover only the fallback implementations.

- [x] Modernize packaging metadata.
  - Consolidate project metadata in `pyproject.toml` where practical.
  - Declare a realistic supported-Python range.
  - Review build requirements and remove unnecessary dependencies.
  - Replace manual package lists with reliable package discovery.
  - Validate wheels and an extracted source distribution before release.

## Documentation and distribution

- [x] Choose one canonical README and use it for the package description.
  - Removed the outdated `README.rst`; `README.md` is now canonical and is used
    as the package long description.
  - Keep installation, feature, citation, and physics-disclaimer text aligned.

- [x] Refresh `INSTALL.md`.
  - Replace the deprecated `python setup.py install` instructions.
  - Correct the generated HTML path to `docs/build/html/index.html`.
  - Update stale links and supported-toolchain guidance.

- [ ] Tighten `MANIFEST.in`.
  - Include documentation sources without including generated `docs/build`
    artifacts.
  - Inspect the contents and size of an extracted source distribution.

- [ ] Simplify the Sphinx navigation hierarchy.
  - Remove duplicate toctree references while preserving access to all API
    pages.

## Runtime maintenance

- [x] Make the electron-phonon Bose function numerically stable for large
  positive arguments and add regression tests for its limiting behavior.

- [ ] Replace mutable dictionary defaults in builder constructors with `None`
  and initialize dictionaries inside the functions.

- [ ] Replace printed optional-extension messages with appropriate Python
  warnings so applications can filter or capture them.

- [ ] Review remaining deprecated Python, NumPy, SciPy, and Cython APIs under
  the supported dependency versions.

## Release checks

- [ ] Run the complete pure-Python and compiled test suites.
- [ ] Build the documentation with `-W --keep-going`.
- [ ] Build wheels and a source distribution in clean environments.
- [ ] Install and test the produced artifacts rather than only the working tree.
- [ ] Update `CHANGELOG.md` and verify package version consistency.
