<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.math.cholesky_concat" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.math.cholesky_concat

Concatenates `chol @ chol.T` with additional rows and columns.

``` python
tfp.math.cholesky_concat(
    chol,
    cols,
    name=None
)
```



Defined in [`python/math/linalg.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/math/linalg.py).

<!-- Placeholder for "Used in" -->

This operation is conceptually identical to:
```python
def cholesky_concat_slow(chol, cols):  # cols shaped (n + m) x m = z x m
  mat = tf.matmul(chol, chol, adjoint_b=True)  # batch of n x n
  # Concat columns.
  mat = tf.concat([mat, cols[..., :tf.shape(mat)[-2], :]], axis=-1)  # n x z
  # Concat rows.
  mat = tf.concat([mat, tf.linalg.matrix_transpose(cols)], axis=-2)  # z x z
  return tf.linalg.cholesky(mat)
```
but whereas `cholesky_concat_slow` would cost `O(z**3)` work,
`cholesky_concat` only costs `O(z**2 + m**3)` work.

The resulting (implicit) matrix must be symmetric and positive definite.
Thus, the bottom right `m x m` must be self-adjoint, and we do not require a
separate `rows` argument (which can be inferred from `conj(cols.T)`).

#### Args:


* <b>`chol`</b>: Cholesky decomposition of `mat = chol @ chol.T`.
* <b>`cols`</b>: The new columns whose first `n` rows we would like concatenated to the
  right of `mat = chol @ chol.T`, and whose conjugate transpose we would
  like concatenated to the bottom of `concat(mat, cols[:n,:])`. A `Tensor`
  with final dims `(n+m, m)`. The first `n` rows are the top right rectangle
  (their conjugate transpose forms the bottom left), and the bottom `m x m`
  is self-adjoint.
* <b>`name`</b>: Optional name for this op.


#### Returns:


* <b>`chol_concat`</b>: The Cholesky decomposition of:
  ```
  [ [ mat  cols[:n, :] ]
    [   conj(cols.T)   ] ]
  ```