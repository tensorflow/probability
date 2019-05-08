<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.edward2.Autoregressive" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.edward2.Autoregressive

Create a random variable for Autoregressive.

``` python
tfp.edward2.Autoregressive(
    *args,
    **kwargs
)
```



Defined in [`python/edward2/interceptor.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/edward2/interceptor.py).

<!-- Placeholder for "Used in" -->

See Autoregressive for more details.

#### Returns:

  RandomVariable.

#### Original Docstring for Distribution

Construct an `Autoregressive` distribution.


#### Args:

* <b>`distribution_fn`</b>: Python `callable` which constructs a
  `tfd.Distribution`-like instance from a `Tensor` (e.g.,
  `sample0`). The function must respect the "autoregressive property",
  i.e., there exists a permutation of event such that each coordinate is a
  diffeomorphic function of on preceding coordinates.
* <b>`sample0`</b>: Initial input to `distribution_fn`; used to
  build the distribution in `__init__` which in turn specifies this
  distribution's properties, e.g., `event_shape`, `batch_shape`, `dtype`.
  If unspecified, then `distribution_fn` should be default constructable.
* <b>`num_steps`</b>: Number of times `distribution_fn` is composed from samples,
  e.g., `num_steps=2` implies
  `distribution_fn(distribution_fn(sample0).sample(n)).sample()`.
* <b>`validate_args`</b>: Python `bool`.  Whether to validate input with asserts.
  If `validate_args` is `False`, and the inputs are invalid,
  correct behavior is not guaranteed.
* <b>`allow_nan_stats`</b>: Python `bool`, default `True`. When `True`, statistics
  (e.g., mean, mode, variance) use the value "`NaN`" to indicate the
  result is undefined. When `False`, an exception is raised if one or
  more of the statistic's batch members are undefined.
* <b>`name`</b>: Python `str` name prefixed to Ops created by this class.
  Default value: "Autoregressive".


#### Raises:

* <b>`ValueError`</b>: if `num_steps` and
  `num_elements(distribution_fn(sample0).event_shape)` are both `None`.
* <b>`ValueError`</b>: if `num_steps < 1`.