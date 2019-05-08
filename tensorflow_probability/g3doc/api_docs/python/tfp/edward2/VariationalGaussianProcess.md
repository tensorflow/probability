<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.edward2.VariationalGaussianProcess" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.edward2.VariationalGaussianProcess

Create a random variable for VariationalGaussianProcess.

``` python
tfp.edward2.VariationalGaussianProcess(
    *args,
    **kwargs
)
```



Defined in [`python/edward2/interceptor.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/edward2/interceptor.py).

<!-- Placeholder for "Used in" -->

See VariationalGaussianProcess for more details.

#### Returns:

  RandomVariable.

#### Original Docstring for Distribution

Instantiate a VariationalGaussianProcess Distribution.


#### Args:

* <b>`kernel`</b>: `PositiveSemidefiniteKernel`-like instance representing the
  GP's covariance function.
* <b>`index_points`</b>: `float` `Tensor` representing finite (batch of) vector(s) of
  points in the index set over which the VGP is defined. Shape has the
  form `[b1, ..., bB, e1, f1, ..., fF]` where `F` is the number of feature
  dimensions and must equal `kernel.feature_ndims` and `e1` is the number
  (size) of index points in each batch (we denote it `e1` to distinguish
  it from the numer of inducing index points, denoted `e2` below).
  Ultimately the VariationalGaussianProcess distribution corresponds to an
  `e1`-dimensional multivariate normal. The batch shape must be
  broadcastable with `kernel.batch_shape`, the batch shape of
  `inducing_index_points`, and any batch dims yielded by `mean_fn`.
* <b>`inducing_index_points`</b>: `float` `Tensor` of locations of inducing points in
  the index set. Shape has the form `[b1, ..., bB, e2, f1, ..., fF]`, just
  like `index_points`. The batch shape components needn't be identical to
  those of `index_points`, but must be broadcast compatible with them.
* <b>`variational_inducing_observations_loc`</b>: `float` `Tensor`; the mean of the
  (full-rank Gaussian) variational posterior over function values at the
  inducing points, conditional on observed data. Shape has the form `[b1,
  ..., bB, e2]`, where `b1, ..., bB` is broadcast compatible with other
  parameters' batch shapes, and `e2` is the number of inducing points.
* <b>`variational_inducing_observations_scale`</b>: `float` `Tensor`; the scale
  matrix of the (full-rank Gaussian) variational posterior over function
  values at the inducing points, conditional on observed data. Shape has
  the form `[b1, ..., bB, e2, e2]`, where `b1, ..., bB` is broadcast
  compatible with other parameters and `e2` is the number of inducing
  points.
* <b>`mean_fn`</b>: Python `callable` that acts on index points to produce a (batch
  of) vector(s) of mean values at those index points. Takes a `Tensor` of
  shape `[b1, ..., bB, f1, ..., fF]` and returns a `Tensor` whose shape is
  (broadcastable with) `[b1, ..., bB]`. Default value: `None` implies
  constant zero function.
* <b>`observation_noise_variance`</b>: `float` `Tensor` representing the variance
  of the noise in the Normal likelihood distribution of the model. May be
  batched, in which case the batch shape must be broadcastable with the
  shapes of all other batched parameters (`kernel.batch_shape`,
  `index_points`, etc.).
  Default value: `0.`
* <b>`predictive_noise_variance`</b>: `float` `Tensor` representing additional
  variance in the posterior predictive model. If `None`, we simply re-use
  `observation_noise_variance` for the posterior predictive noise. If set
  explicitly, however, we use the given value. This allows us, for
  example, to omit predictive noise variance (by setting this to zero) to
  obtain noiseless posterior predictions of function values, conditioned
  on noisy observations.
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
  Default value: "VariationalGaussianProcess".


#### Raises:

* <b>`ValueError`</b>: if `mean_fn` is not `None` and is not callable.