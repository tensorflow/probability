<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.edward2.MultivariateNormalLinearOperator" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.edward2.MultivariateNormalLinearOperator

Create a random variable for MultivariateNormalLinearOperator.

``` python
tfp.edward2.MultivariateNormalLinearOperator(
    *args,
    **kwargs
)
```



Defined in [`python/edward2/interceptor.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/edward2/interceptor.py).

<!-- Placeholder for "Used in" -->

See MultivariateNormalLinearOperator for more details.

#### Returns:

  RandomVariable.

#### Original Docstring for Distribution

Construct Multivariate Normal distribution on `R^k`.

The `batch_shape` is the broadcast shape between `loc` and `scale`
arguments.

The `event_shape` is given by last dimension of the matrix implied by
`scale`. The last dimension of `loc` (if provided) must broadcast with this.

Recall that `covariance = scale @ scale.T`.

Additional leading dimensions (if any) will index batches.


#### Args:

* <b>`loc`</b>: Floating-point `Tensor`. If this is set to `None`, `loc` is
  implicitly `0`. When specified, may have shape `[B1, ..., Bb, k]` where
  `b >= 0` and `k` is the event size.
* <b>`scale`</b>: Instance of `LinearOperator` with same `dtype` as `loc` and shape
  `[B1, ..., Bb, k, k]`.
* <b>`validate_args`</b>: Python `bool`, default `False`. Whether to validate input
  with asserts. If `validate_args` is `False`, and the inputs are
  invalid, correct behavior is not guaranteed.
* <b>`allow_nan_stats`</b>: Python `bool`, default `True`. If `False`, raise an
  exception if a statistic (e.g. mean/mode/etc...) is undefined for any
  batch member If `True`, batch members with valid parameters leading to
  undefined statistics will return NaN for this statistic.
* <b>`name`</b>: The name to give Ops created by the initializer.


#### Raises:

* <b>`ValueError`</b>: if `scale` is unspecified.
* <b>`TypeError`</b>: if not `scale.dtype.is_floating`