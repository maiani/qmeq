# QmeQ documentation

This is QmeQ's documentation, built with MkDocs. It is the successor to the
Sphinx documentation, which now lives at `legacy_docs/` (`legacy_docs/`) and
remains the complete reference — user manual, API reference, theory pages, and
worked examples — until this site has equivalent coverage. It still builds
warning-clean in CI.

!!! warning "This site is incomplete"
    Two sections exist so far:

    - [Guide](guide/index.md) — user-and-API-facing material: what QmeQ
      computes, the seven approaches and their validity domains, and the
      `Builder` API. Started, not comprehensive.
    - [Conventions](conventions/index.md) — internal conventions the code
      relies on but never wrote down: index layouts, sign conventions,
      sentinel values. For people changing QmeQ's internals.

    Neither section has migrated the full Sphinx manual yet. **If you want API
    reference material today**, go to `legacy_docs/` (`legacy_docs/`) (build it
    with Sphinx — see its `README` — or read it online if a build is hosted),
    or read the docstrings directly in `qmeq/`, which are the ground truth
    either way.

## Contents

- [Conventions](conventions/index.md) — index layouts, sign conventions, and
  the machinery around them.
