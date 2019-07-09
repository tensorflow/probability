<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.edward2.MultivariateNormalFullCovariance" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.edward2.MultivariateNormalFullCovariance

Create a random variable for MultivariateNormalFullCovariance.

``` python
tfp.edward2.MultivariateNormalFullCovariance(
    *args,
    **kwargs
)
```



Defined in [`python/edward2/interceptor.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/edward2/interceptor.py).

<!-- Placeholder for "Used in" -->

See MultivariateNormalFullCovariance for more details.

#### Returns:

RandomVariable.


#### Original Docstring for Distribution

Construct Multivariate Normal distribution on `R^k`.

The `batch_shape` is the broadcast shape between `loc` and
`covariance_matrix` arguments.

The `event_shape` is given by last dimension of the matrix implied by
`covariance_matrix`. The last dimension of `loc` (if provided) must
broadcast with this.

A non-batch `covariance_matrix` matrix is a `k x k` symmetric positive
definite matrix.  In other words it is (real) symmetric with all eigenvalues
strictly positive.

Additional leading dimensions (if any) will index batches.

#### Args:


* <b>`loc`</b>: Floating-point `Tensor`. If this is set to `None`, `loc` is
  implicitly `0`. When specified, may have shape `[B1, ..., Bb, k]` where
  `b >= 0` and `k` is the event size.
* <b>`covariance_matrix`</b>: Floating-point, symmetric positive definite `Tensor` of
  same `dtype` as `loc`.  The strict upper triangle of `covariance_matrix`
  is ignored, so if `covariance_matrix` is not symmetric no error will be
  raised (unless `validate_args is True`).  `covariance_matrix` has shape
  `[B1, ..., Bb, k, k]` where `b >= 0` and `k` is the event size.
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


* <b>`ValueError`</b>: if neither `loc` nor `covariance_matrix` are specified.