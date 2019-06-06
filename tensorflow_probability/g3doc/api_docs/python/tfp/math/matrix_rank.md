<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.math.matrix_rank" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.math.matrix_rank

Compute the matrix rank; the number of non-zero SVD singular values.

``` python
tfp.math.matrix_rank(
    a,
    tol=None,
    validate_args=False,
    name=None
)
```



Defined in [`python/math/linalg.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/math/linalg.py).

<!-- Placeholder for "Used in" -->


#### Arguments:


* <b>`a`</b>: (Batch of) `float`-like matrix-shaped `Tensor`(s) which are to be
  pseudo-inverted.
* <b>`tol`</b>: Threshold below which the singular value is counted as "zero".
  Default value: `None` (i.e., `eps * max(rows, cols) * max(singular_val)`).
* <b>`validate_args`</b>: When `True`, additional assertions might be embedded in the
  graph.
  Default value: `False` (i.e., no graph assertions are added).
* <b>`name`</b>: Python `str` prefixed to ops created by this function.
  Default value: "matrix_rank".


#### Returns:


* <b>`matrix_rank`</b>: (Batch of) `int32` scalars representing the number of non-zero
  singular values.