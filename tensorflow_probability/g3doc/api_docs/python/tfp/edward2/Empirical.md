<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.edward2.Empirical" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.edward2.Empirical

Create a random variable for Empirical.

``` python
tfp.edward2.Empirical(
    *args,
    **kwargs
)
```



Defined in [`python/edward2/interceptor.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/edward2/interceptor.py).

<!-- Placeholder for "Used in" -->

See Empirical for more details.

#### Returns:

  RandomVariable.

#### Original Docstring for Distribution

Initialize `Empirical` distributions.


#### Args:

* <b>`samples`</b>: Numeric `Tensor` of shape [B1, ..., Bk, S, E1, ..., En]`,
  `k, n >= 0`. Samples or batches of samples on which the distribution
  is based. The first `k` dimensions index into a batch of independent
  distributions. Length of `S` dimension determines number of samples
  in each multiset. The last `n` dimension represents samples for each
  distribution. n is specified by argument event_ndims.
* <b>`event_ndims`</b>: Python `int32`, default `0`. number of dimensions for each
  event. When `0` this distribution has scalar samples. When `1` this
  distribution has vector-like samples.
* <b>`validate_args`</b>: Python `bool`, default `False`. When `True` distribution
  parameters are checked for validity despite possibly degrading runtime
  performance. When `False` invalid inputs may silently render incorrect
  outputs.
* <b>`allow_nan_stats`</b>: Python `bool`, default `True`. When `True`, statistics
  (e.g., mean, mode, variance) use the value `NaN` to indicate the
  result is undefined. When `False`, an exception is raised if one or
  more of the statistic's batch members are undefined.
* <b>`name`</b>: Python `str` name prefixed to Ops created by this class.


#### Raises:

* <b>`ValueError`</b>: if the rank of `samples` < event_ndims + 1.