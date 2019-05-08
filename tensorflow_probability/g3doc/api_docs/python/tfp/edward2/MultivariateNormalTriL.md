<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.edward2.MultivariateNormalTriL" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.edward2.MultivariateNormalTriL

Create a random variable for MultivariateNormalTriL.

``` python
tfp.edward2.MultivariateNormalTriL(
    *args,
    **kwargs
)
```



Defined in [`python/edward2/interceptor.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/edward2/interceptor.py).

<!-- Placeholder for "Used in" -->

See MultivariateNormalTriL for more details.

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
scale = scale_tril
```

where `scale_tril` is lower-triangular `k x k` matrix with non-zero
diagonal, i.e., `tf.diag_part(scale_tril) != 0`.

Additional leading dimensions (if any) will index batches.


#### Args:

* <b>`loc`</b>: Floating-point `Tensor`. If this is set to `None`, `loc` is
  implicitly `0`. When specified, may have shape `[B1, ..., Bb, k]` where
  `b >= 0` and `k` is the event size.
* <b>`scale_tril`</b>: Floating-point, lower-triangular `Tensor` with non-zero
  diagonal elements. `scale_tril` has shape `[B1, ..., Bb, k, k]` where
  `b >= 0` and `k` is the event size.
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

* <b>`ValueError`</b>: if neither `loc` nor `scale_tril` are specified.