# Typing

QmeQ is adding type hints **opportunistically**: when a file is touched for
another reason and its signatures are unclear, they get annotated. There is no
big-bang conversion.

## Policy

**Hints are for readers, not for tools.** There is no type checker configured,
no `py.typed` marker, and no CI gate. Annotations exist to answer "what is this
argument" without reading three call sites. They are documentation that happens
to be machine-readable.

Consequences worth being explicit about:

- **An unchecked hint can be wrong.** Nothing verifies these. Treat a hint as a
  strong comment, not a guarantee — and if you find one that disagrees with the
  code, the code is right and the hint is a bug to fix.
- **Coverage will be uneven for a long time.** An unannotated signature means
  nobody has needed it yet, not that it is dynamically typed on purpose.
- **Do not annotate speculatively.** Annotate what you are already changing.

## Conventions

**Use `from __future__ import annotations`.** Annotations stay strings, cost
nothing at import, and forward references need no quoting.

**Prefer a truthful union over a convenient lie.** The parameter that motivated
this policy is `si`, threaded through the approaches and kernel handlers:

```python
def __init__(self, si: StateIndexingDM | StateIndexingDMc) -> None:
```

`StateIndexingDM` alone would read better and would be **false** — the 2vN
approaches pass a `StateIndexingDMc`, which is why `c_kernel_handler.pyx`
branches on `isinstance(si, StateIndexingDMc)`.

The union cannot be collapsed to a common ancestor either. `StateIndexingDMc`
is a *sibling* of `StateIndexingDM`, not a subclass, and their shared base
`StateIndexing` does not define `get_ind_dm0` at all — see
[class hierarchy](state-indexing.md#class-hierarchy). Naming the base would
admit types that cannot satisfy the method and would say less than the union
does.

Where a class genuinely requires the narrower type, say so:
`DensityMatrixLayout` takes `StateIndexingDM` because it needs the conjugation
map.

**Keep array types simple.** `np.ndarray` rather than parameterised
`NDArray[...]`. Element dtypes are governed by `qmeq.wrappers.mytypes`
(`doublenp`, `complexnp`) and belong in the docstring, where the reason for the
choice can be stated.

**Annotate `None` returns.** Most of the kernel-handler methods mutate an array
in place and return nothing. `-> None` makes that visible at the signature
instead of at the end of the body.

## Relationship to the Cython twins

The compiled `.pyx` implementations already carry real static types in their
`.pxd` headers, and those types are enforced by the compiler. Python
annotations on the pure-Python twin do **not** enforce anything and are not
kept in sync automatically. When the two disagree, the `.pxd` is authoritative
for the compiled path.

## Annotated so far

- `qmeq.approach.dm_layout` — fully annotated (new module).
- `qmeq.approach.kernel_handler` — constructors, predicates, and the insertion
  and accessor methods.
- `qmeq.indexing` — the three `get_ind_dm0` overloads and the two named
  accessors. Note the return type `int | bool | None`, which is the honest
  signature of the `maptype` overload described in
  [State indexing](state-indexing.md).
