<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.edward2.GaussianProcessRegressionModel" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.edward2.GaussianProcessRegressionModel

Create a random variable for GaussianProcessRegressionModel.

``` python
tfp.edward2.GaussianProcessRegressionModel(
    *args,
    **kwargs
)
```



Defined in [`python/edward2/interceptor.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/edward2/interceptor.py).

<!-- Placeholder for "Used in" -->

See GaussianProcessRegressionModel for more details.

#### Returns:

  RandomVariable.

#### Original Docstring for Distribution

Construct a GaussianProcessRegressionModel instance.


#### Args:

* <b>`kernel`</b>: `PositiveSemidefiniteKernel`-like instance representing the
  GP's covariance function.
* <b>`index_points`</b>: `float` `Tensor` representing finite collection, or batch of
  collections, of points in the index set over which the GP is defined.
  Shape has the form `[b1, ..., bB, e, f1, ..., fF]` where `F` is the
  number of feature dimensions and must equal `kernel.feature_ndims` and
  `e` is the number (size) of index points in each batch. Ultimately this
  distribution corresponds to an `e`-dimensional multivariate normal. The
  batch shape must be broadcastable with `kernel.batch_shape` and any
  batch dims yielded by `mean_fn`.
* <b>`observation_index_points`</b>: `float` `Tensor` representing finite collection,
  or batch of collections, of points in the index set for which some data
  has been observed. Shape has the form `[b1, ..., bB, e, f1, ..., fF]`
  where `F` is the number of feature dimensions and must equal
  `kernel.feature_ndims`, and `e` is the number (size) of index points in
  each batch. `[b1, ..., bB, e]` must be broadcastable with the shape of
  `observations`, and `[b1, ..., bB]` must be broadcastable with the
  shapes of all other batched parameters (`kernel.batch_shape`,
  `index_points`, etc). The default value is `None`, which corresponds to
  the empty set of observations, and simply results in the prior
  predictive model (a GP with noise of variance
  `predictive_noise_variance`).
* <b>`observations`</b>: `float` `Tensor` representing collection, or batch of
  collections, of observations corresponding to
  `observation_index_points`. Shape has the form `[b1, ..., bB, e]`, which
  must be brodcastable with the batch and example shapes of
  `observation_index_points`. The batch shape `[b1, ..., bB]` must be
  broadcastable with the shapes of all other batched parameters
  (`kernel.batch_shape`, `index_points`, etc.). The default value is
  `None`, which corresponds to the empty set of observations, and simply
  results in the prior predictive model (a GP with noise of variance
  `predictive_noise_variance`).
* <b>`observation_noise_variance`</b>: `float` `Tensor` representing the variance
  of the noise in the Normal likelihood distribution of the model. May be
  batched, in which case the batch shape must be broadcastable with the
  shapes of all other batched parameters (`kernel.batch_shape`,
  `index_points`, etc.).
  Default value: `0.`
* <b>`predictive_noise_variance`</b>: `float` `Tensor` representing the variance in
  the posterior predictive model. If `None`, we simply re-use
  `observation_noise_variance` for the posterior predictive noise. If set
  explicitly, however, we use this value. This allows us, for example, to
  omit predictive noise variance (by setting this to zero) to obtain
  noiseless posterior predictions of function values, conditioned on noisy
  observations.
* <b>`mean_fn`</b>: Python `callable` that acts on `index_points` to produce a
  collection, or batch of collections, of mean values at `index_points`.
  Takes a `Tensor` of shape `[b1, ..., bB, f1, ..., fF]` and returns a
  `Tensor` whose shape is broadcastable with `[b1, ..., bB]`.
  Default value: `None` implies the constant zero function.
* <b>`jitter`</b>: `float` scalar `Tensor` added to the diagonal of the covariance
  matrix to ensure positive definiteness of the covariance matrix.
  Default value: `1e-6`.
* <b>`validate_args`</b>: Python `bool`, default `False`. When `True` distribution
  parameters are checked for validity despite possibly degrading runtime
  performance. When `False` invalid inputs may silently render incorrect
  outputs.
  Default value: `False`.
* <b>`allow_nan_stats`</b>: Python `bool`, default `True`. When `True`,
  statistics (e.g., mean, mode, variance) use the value `NaN` to
  indicate the result is undefined. When `False`, an exception is raised
  if one or more of the statistic's batch members are undefined.
  Default value: `False`.
* <b>`name`</b>: Python `str` name prefixed to Ops created by this class.
  Default value: 'GaussianProcessRegressionModel'.


#### Raises:

* <b>`ValueError`</b>: if either
  - only one of `observations` and `observation_index_points` is given, or
  - `mean_fn` is not `None` and not callable.