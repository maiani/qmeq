# AGENTS.md

Operational guidance for AI agents working in QmeQ.

Read [CONTRIBUTING.md](CONTRIBUTING.md) before changing anything. It owns the
development setup, the rules for changing the code (preserving the physics,
keeping the Python and Cython backends together, immutable reference data,
what a test asserts), the test gates, the documentation routing and the
release procedure, and all of it applies to you. This file adds only what is
specific to working as an agent. For the project overview, installation,
authorship and citation, use [README.md](README.md),
[INSTALL.md](INSTALL.md) and [AUTHORS.md](AUTHORS.md); do not duplicate them
here.

## Sources of truth

- [DEVPLAN.md](DEVPLAN.md) owns scope, priorities and open work. QmeQ takes no
  new features: its section 1 sets what belongs here and its section 2 lists
  what was dropped. It is a coordination document, so production code, tests,
  fixtures and permanent documentation state durable conventions and
  provenance directly rather than link to it or copy its item labels.
- [CHANGELOG.md](CHANGELOG.md) owns user-visible changes under `[Unreleased]`.
- [REFERENCES.md](REFERENCES.md) owns stable keys for external literature used
  by the implementation. The README remains the source for citing QmeQ itself.
- [docs/README.md](docs/README.md) describes the documentation tree and how to
  build it.

Read the relevant source and tests before editing. Treat line numbers and
status claims in plans as snapshots, and verify them against the working tree.

## Writing for this repository

Comments, docstrings, documentation and test descriptions state what the code
does, in the present tense. Do not narrate history ("used to", "previously",
"now", "no longer") or record which pass found something; that belongs in the
commit message. A finding that is live guidance, such as a trap worth naming,
may stay, stripped of the narrative.

## Worktree and finish discipline

Assume uncommitted changes belong to the user. Before editing or staging:

- confirm the repository top level;
- inspect staged and unstaged changes separately;
- preserve unrelated edits, generated artifacts and intentional deletions;
- stage explicit relevant paths only; and
- do not commit, tag, push or publish without the corresponding user approval.

Run the two forced-backend suites one after the other, never in parallel in
the same checkout, and do not edit Python sources while a suite runs: tests
that start subprocesses import the files as they are on disk.

Before handing off a change, complete the checklist in CONTRIBUTING.md
("Before a change is merged"), inspect the final staged diff, and list
anything intentionally left unstaged.
