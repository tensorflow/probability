<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.edward2.MultivariateNormalDiag" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.edward2.MultivariateNormalDiag

Create a random variable for MultivariateNormalDiag.

``` python
tfp.edward2.MultivariateNormalDiag(
    *args,
    **kwargs
)
```



Defined in [`python/edward2/interceptor.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/edward2/interceptor.py).

<!-- Placeholder for "Used in" -->

See MultivariateNormalDiag for more details.

#### Returns:

  RandomVariable.

#### Original Docstring for Distribution

Construct Multivariate Normal distribution on `R^k`.

The `batch_shape` is the broadcast shape between `loc` and `scale`
arguments.

The `event_shape` is given by last dimension of the matrix implied by
`scale`. The last dimension of `loc` (if provided) must broadcast with this.

Recall that `covariance = scale @ scale.T`. A (non-batch) `scale` matrix is:

```none
scale = diag(scale_diag + scale_identity_multiplier * ones(k))
```

* <b>`where`</b>: 
* `scale_diag.shape = [k]`, and,
* `scale_identity_multiplier.shape = []`.

Additional leading dimensions (if any) will index batches.

If both `scale_diag` and `scale_identity_multiplier` are `None`, then
`scale` is the Identity matrix.


#### Args:

* <b>`loc`</b>: Floating-point `Tensor`. If this is set to `None`, `loc` is
  implicitly `0`. When specified, may have shape `[B1, ..., Bb, k]` where
  `b >= 0` and `k` is the event size.
* <b>`scale_diag`</b>: Non-zero, floating-point `Tensor` representing a diagonal
  matrix added to `scale`. May have shape `[B1, ..., Bb, k]`, `b >= 0`,
  and characterizes `b`-batches of `k x k` diagonal matrices added to
  `scale`. When both `scale_identity_multiplier` and `scale_diag` are
  `None` then `scale` is the `Identity`.
* <b>`scale_identity_multiplier`</b>: Non-zero, floating-point `Tensor` representing
  a scaled-identity-matrix added to `scale`. May have shape
  `[B1, ..., Bb]`, `b >= 0`, and characterizes `b`-batches of scaled
  `k x k` identity matrices added to `scale`. When both
  `scale_identity_multiplier` and `scale_diag` are `None` then `scale` is
  the `Identity`.
* <b>`validate_args`</b>: Python `bool`, default `False`. When `True` distribution
  parameters are checked for validity despite possibly degrading runtime
  performance. When `False` invalid inputs may silently render incorrect
  outputs.
* <b>`allow_nan_stats`</b>: Python `bool`, default `True`. When `True`,
  statistics (e.g., mean, mode, variance) use the value "`NaN`" to
  indicate the result is undefined. When `False`, an exception is raised
  if one or more of the statistic's batch members are undefined.
* <b>`name`</b>: Python `str` name prefixed to Ops created by this class.


#### Raises:

* <b>`ValueError`</b>: if at most `scale_identity_multiplier` is specified.