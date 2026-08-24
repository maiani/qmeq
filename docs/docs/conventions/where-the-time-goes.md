# Where the time goes

Measured, because the shape of this calculation is not what the name "master
equation solver" suggests. It is **not** a sparse linear-algebra problem.

## The kernels are dense, and get denser

Fraction of non-zero entries in the assembled kernel, compiled backend, a chain
of `nsingle` orbitals with one interaction term and two leads:

| kerntype | N = 70 | N = 924 | N = 12870 |
|---|---|---|---|
| 1vN | 0.950 | 0.978 | **0.992** |
| Lindblad | 0.682 | 0.756 | 0.771 |

Smaller systems, all approaches (pure Python, `nsingle = 5`):

| kerntype | N | nnz/N² |
|---|---|---|
| Pauli | 32 | 0.441 |
| Lindblad | 252 | 0.304 |
| 1vN / Redfield | 252 | 0.594 |
| RTD | 32 | 0.891 |

Density *rises* with system size. Sparse storage generally needs well under 10%
occupancy to pay for its indirection, so a sparse format here would be a
pessimization, not an optimization. The reason is structural: within a charge
sector the tunnelling terms couple essentially every retained element to every
other, and those blocks dominate the matrix.

## The linear solve is not the bottleneck

Profile of five `1vN` solves, pure Python, `nsingle = 5`, N = 252:

```
1.268 s  total
1.213 s  generate_kern                     (95.7 %)
1.210 s    generate_coupling_terms
0.299 s      set_matrix_element     (96 670 calls)
0.125 s      get_ind_dm0           (291 430 calls)
0.118 s      get_ind_dm1           (350 950 calls)
0.049 s  generate_fct
```

`np.linalg.solve` does not appear in the top nine entries. **Assembly is 96 % of
the runtime**, and roughly 40 % of assembly is index lookup and insertion — a
scalar-dispatch cost, not arithmetic.

## RTD is dominated by scalar special functions

Five `RTD` solves, pure Python, `nsingle = 5`:

```
101.4 s  total          (120 065 651 function calls)
 99.1 s  generate_col_diag_kern_2nd_order   (97.7 %)
 44.6 s    phi                    (3 601 095 calls)
 37.7 s    integralD              (1 840 920 calls)
 32.7 s    integralX              (1 840 920 calls)
 31.4 s    delta_phi              (3 153 825 calls)
```

Three and a half million scalar `phi` evaluations for a five-orbital dot. This is
why the Cython twins exist: it is a scalar-loop problem, and no linear-algebra
library can help with it.

## Why loops rather than NumPy

The assembly code is nested `for` loops over many-body states, which invites the
question. Five reasons, in decreasing order of force:

1. **The iteration space is ragged and filtered.** Loops run over
   `statesdm[charge]`, whose sizes are binomial coefficients, nested and then
   pruned by `is_included` / `is_unique` from the symmetry reduction. It is a
   filtered multi-level product, not a rectangular array.
2. **The scalar kernel branches on its data.** `_D_integral_equal_T` selects
   between *different formulas* on `abs(E1 - E2) < 1e-10` and
   `abs(deltaE) < 1e-10` — degenerate-energy limits. Vectorising means either
   evaluating every branch and selecting, which is both wasteful and wrong in
   the degenerate limit, or segmenting the inputs by branch first.
3. **Destinations are scatter-adds through a lookup table with a sentinel.**
   `mapdm0` may return `NO_INDEX`, and many contributions accumulate into the
   same entry. NumPy needs `np.add.at`, which is slow, or precomputed index
   arrays.
4. **The inner reductions are tiny.** `for l in range(nleads)` with two leads.
   NumPy's per-call overhead exceeds the work.
5. **The project's answer for hot paths was Cython, not NumPy** — a `c_` twin per
   hot module. For irregular, branchy scalar work that is arguably the better
   tool: no temporaries, no branch segmentation, and the loops keep the shape of
   the published equations, which is how the physics gets checked.

### The loops are not currently the binding constraint

`MAX_CACHE` in `qmeq/specfunc/specfunc.py` bounds the `lru_cache` on every
memoised special function. It was 100; it is now 10000. Measured by editing the
constant, RTD, pure Python, best of three:

| bound | nsingle=4 | nsingle=5 | speed-up |
|---|---|---|---|
| 100 (was) | 0.776 s / 80.9 % | 11.81 s / 74.8 % | 1.00× |
| 1 000 | 0.396 s / 94.4 % | 6.20 s / 90.9 % | ~1.9× |
| **10 000** (now) | **0.384 s / 97.1 %** | **5.06 s / 95.7 %** | **2.0–2.3×** |
| 50 000 | 0.368 s / 99.1 % | 5.00 s / 96.8 % | 2.1–2.4× |
| unbounded | 0.344 s / 99.1 % | 4.52 s / 99.2 % | 2.3–2.6× |

**A 2.2× speed-up from one integer.** The knee is near 1000; 10000 captures
almost all of the rest; 50000 adds about one per cent.

Memory: **207 bytes per entry**, measured with `tracemalloc`. Across eight
memoised functions the bound costs at most ~17 MB, against a kernel that reaches
1.3 GB on its own at large N.

!!! warning "The bound must stay finite"
    Distinct keys per solve grow roughly elevenfold per added orbital — 221,
    1 735, 16 122, 187 646 for `nsingle` 2 to 5 — and parameter sweeps generate
    fresh float keys indefinitely with nothing to evict them. Unbounded is a
    leak, not an optimisation. `test_specfunc.py` asserts finiteness.

### The gain is not shared evenly

Raising the bound is not a general speed-up. Best of three, pure Python,
`nsingle = 5` (`nsingle = 4` for RTD):

| kerntype | bound 100 | bound 10000 | gain |
|---|---|---|---|
| RTD | 0.849 s | 0.427 s | **2.0×** |
| Pauli | 0.0046 s | 0.0017 s | 2.7× (negligible in absolute terms) |
| Lindblad | 0.183 s | 0.168 s | 1.09× |
| Redfield | 0.189 s | 0.188 s | — |
| 1vN | 0.195 s | 0.194 s | — |

Redfield and 1vN go through `func_1vN`, which is **not** memoised and does its own
integration; their cost is assembly, not special functions, exactly as the
profile above shows. So the cache bound is effectively an RTD optimisation with a
small Lindblad bonus.

### Cacheability falls off with arity

Which is why only some functions are memoised. Measured hit rates at the old
bound, against the number of *independent continuous* arguments:

| function | independent float args | hit rate |
|---|---|---|
| `bose(x, sign)` | 1 | 98.5 % |
| `fermi_func(x)` | 1 | 93.0 % |
| `diff_phi(x, sign)` | 1 | 87.8 % |
| `phi(x, Dp, Dm, sign)` | 1 plus near-constants | 69.4 % |
| `delta_phi(x1, x2, Dp, Dm, sign)` | 2 | 58.5 % |
| `integralD(p1, eta1, E1, E2, E3, T1, T2, mu1, mu2, D, b_and_R, ImGamma)` | 3 of 12 args | **0.3 %** |

So the memoised set is exactly the low-arity tail, and that is not an accident.
**Caching `integralD`/`integralX` was tried and rejected**: at a 0.3 % hit rate
the twelve-argument key costs more to hash than the call saves, and the solve got
*slower* — 0.406 s to 0.449 s at `nsingle = 4`.

Nothing else is a candidate. `func_pauli`, `func_1vN`, `func_lambshift` and
`fermi_lpm` return arrays; `hilbert_fredriksen` takes arrays and is unhashable;
`kernel_fredriksen`, `Ozaki` and `BW_Ozaki` are called once per setup rather than
per element.

Two properties of the change worth keeping in mind: it is **pure-Python only**,
since the compiled path uses its own `c_specfunc`; and it is **bit-identical by
construction**, because memoisation is exact — verified by solving the whole RTD
reference matrix at both bounds and comparing all 21 arrays with
`np.array_equal`.

## What this means for optimisation

The levers, in the order the profile suggests:

1. **The special functions** — memoisation or vectorisation of `phi`,
   `delta_phi`, `integralD`, `integralX`. Dominant for RTD by a wide margin.
2. **Diagram enumeration** — the second-order loop nest, which is what generates
   those millions of calls in the first place.
3. **Index lookup and insertion** — 640 000 `get_ind_dm*` calls per five 1vN
   solves. This is the cost the packed-real layout imposes; see
   [Density-matrix layout](density-matrix-layout.md).
4. **The linear solve** — last, and only at large N.

!!! warning "The scaling wall is dense memory, not solve time"
    At N = 12870 the kernel alone is $12870^2 \times 8\,\mathrm{B} \approx
    1.3\,\mathrm{GB}$, and RTD additionally allocates `Wdd2` with a
    per-thread axis. Memory is reached before flops become the limit, and
    because the matrix is 99 % dense there is no sparse escape.

For future coherence-retaining RTD work, record diagram counts, generation
time, integral time, assembly time, memory, and solve time against this
baseline. The data already show that the dominant cost will be integrals and
enumeration, not linear algebra.
