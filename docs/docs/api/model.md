# Model construction

The modules used to build the physical model before handing it to a
`Builder`.

## `qmeq.indexing`

::: qmeq.indexing

## `qmeq.qdot`

::: qmeq.qdot

## `qmeq.leadstun`

::: qmeq.leadstun

## `qmeq.baths`

::: qmeq.baths

## `qmeq.wrappers.mytypes`

The NumPy dtypes shared by the model and the approaches.

::: qmeq.wrappers.mytypes

`qmeq.wrappers.c_mytypes` holds the compiled dtype declarations in its `.pxd`,
mirroring `qmeq.wrappers.mytypes`. `qmeq.wrappers.c_lapack` has no pure-Python
twin: it wraps `scipy.linalg.cython_lapack` for the compiled solver, where the
pure-Python path calls `numpy.linalg`.
