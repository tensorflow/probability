<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.distributions.tridiag" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.distributions.tridiag

``` python
tfp.distributions.tridiag(
    below=None,
    diag=None,
    above=None,
    name=None
)
```

Creates a matrix with values set above, below, and on the diagonal.

Example:

```python
tridiag(below=[1., 2., 3.],
        diag=[4., 5., 6., 7.],
        above=[8., 9., 10.])
# ==> array([[  4.,   8.,   0.,   0.],
#            [  1.,   5.,   9.,   0.],
#            [  0.,   2.,   6.,  10.],
#            [  0.,   0.,   3.,   7.]], dtype=float32)
```

Warning: This Op is intended for convenience, not efficiency.

#### Args:

* <b>`below`</b>: `Tensor` of shape `[B1, ..., Bb, d-1]` corresponding to the below
    diagonal part. `None` is logically equivalent to `below = 0`.
* <b>`diag`</b>: `Tensor` of shape `[B1, ..., Bb, d]` corresponding to the diagonal
    part.  `None` is logically equivalent to `diag = 0`.
* <b>`above`</b>: `Tensor` of shape `[B1, ..., Bb, d-1]` corresponding to the above
    diagonal part.  `None` is logically equivalent to `above = 0`.
* <b>`name`</b>: Python `str`. The name to give this op.


#### Returns:

* <b>`tridiag`</b>: `Tensor` with values set above, below and on the diagonal.


#### Raises:

* <b>`ValueError`</b>: if all inputs are `None`.