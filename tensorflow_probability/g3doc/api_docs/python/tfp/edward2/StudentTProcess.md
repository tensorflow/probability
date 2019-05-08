<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.edward2.StudentTProcess" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.edward2.StudentTProcess

Create a random variable for StudentTProcess.

``` python
tfp.edward2.StudentTProcess(
    *args,
    **kwargs
)
```



Defined in [`python/edward2/interceptor.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/edward2/interceptor.py).

<!-- Placeholder for "Used in" -->

See StudentTProcess for more details.

#### Returns:

  RandomVariable.

#### Original Docstring for Distribution

Instantiate a StudentTProcess Distribution.


#### Args:

* <b>`df`</b>: Positive Floating-point `Tensor` representing the degrees of freedom.
  Must be greater than 2.
* <b>`kernel`</b>: `PositiveSemidefiniteKernel`-like instance representing the
  TP's covariance function.
* <b>`index_points`</b>: `float` `Tensor` representing finite (batch of) vector(s) of
  points in the index set over which the TP is defined. Shape has the form
  `[b1, ..., bB, e, f1, ..., fF]` where `F` is the number of feature
  dimensions and must equal `kernel.feature_ndims` and `e` is the number
  (size) of index points in each batch. Ultimately this distribution
  corresponds to a `e`-dimensional multivariate Student's T. The batch
  shape must be broadcastable with `kernel.batch_shape` and any batch dims
  yielded by `mean_fn`.
* <b>`mean_fn`</b>: Python `callable` that acts on `index_points` to produce a (batch
  of) vector(s) of mean values at `index_points`. Takes a `Tensor` of
  shape `[b1, ..., bB, f1, ..., fF]` and returns a `Tensor` whose shape is
  broadcastable with `[b1, ..., bB]`. Default value: `None` implies
  constant zero function.
* <b>`jitter`</b>: `float` scalar `Tensor` added to the diagonal of the covariance
  matrix to ensure positive definiteness of the covariance matrix.
  Default value: `1e-6`.
* <b>`validate_args`</b>: Python `bool`, default `False`. When `True` distribution
  parameters are checked for validity despite possibly degrading runtime
  performance. When `False` invalid inputs may silently render incorrect
  outputs.
  Default value: `False`.
* <b>`allow_nan_stats`</b>: Python `bool`, default `True`. When `True`,
  statistics (e.g., mean, mode, variance) use the value "`NaN`" to
  indicate the result is undefined. When `False`, an exception is raised
  if one or more of the statistic's batch members are undefined.
  Default value: `False`.
* <b>`name`</b>: Python `str` name prefixed to Ops created by this class.
  Default value: "StudentTProcess".


#### Raises:

* <b>`ValueError`</b>: if `mean_fn` is not `None` and is not callable.