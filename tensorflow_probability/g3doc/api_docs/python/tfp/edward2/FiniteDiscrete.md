<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.edward2.FiniteDiscrete" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.edward2.FiniteDiscrete

Create a random variable for FiniteDiscrete.

``` python
tfp.edward2.FiniteDiscrete(
    *args,
    **kwargs
)
```



Defined in [`python/edward2/interceptor.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/edward2/interceptor.py).

<!-- Placeholder for "Used in" -->

See FiniteDiscrete for more details.

#### Returns:

RandomVariable.


#### Original Docstring for Distribution

Construct a finite discrete contribution.

#### Args:


* <b>`outcomes`</b>: A 1-D floating or integer `Tensor`, representing a list of
  possible outcomes in strictly ascending order.
* <b>`logits`</b>: A floating N-D `Tensor`, `N >= 1`, representing the log
  probabilities of a set of FiniteDiscrete distributions. The first `N -
  1` dimensions index into a batch of independent distributions and the
  last dimension represents a vector of logits for each discrete value.
  Only one of `logits` or `probs` should be passed in.
* <b>`probs`</b>: A floating  N-D `Tensor`, `N >= 1`, representing the probabilities
  of a set of FiniteDiscrete distributions. The first `N - 1` dimensions
  index into a batch of independent distributions and the last dimension
  represents a vector of probabilities for each discrete value. Only one
  of `logits` or `probs` should be passed in.
* <b>`rtol`</b>: `Tensor` with same `dtype` as `outcomes`. The relative tolerance for
  floating number comparison. Only effective when `outcomes` is a floating
  `Tensor`. Default is `10 * eps`.
* <b>`atol`</b>: `Tensor` with same `dtype` as `outcomes`. The absolute tolerance for
  floating number comparison. Only effective when `outcomes` is a floating
  `Tensor`. Default is `10 * eps`.
* <b>`validate_args`</b>:  Python `bool`, default `False`. When `True` distribution
  parameters are checked for validity despite possibly degrading runtime
  performance. When `False` invalid inputs may render incorrect outputs.
* <b>`allow_nan_stats`</b>: Python `bool`, default `True`. When `True`, statistics
  (e.g., mean, mode, variance) use the value '`NaN`' to indicate the
  result is undefined. When `False`, an exception is raised if one or more
  of the statistic's batch members are undefined.
* <b>`name`</b>: Python `str` name prefixed to Ops created by this class.