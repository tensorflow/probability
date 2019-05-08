<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.edward2.Multinomial" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.edward2.Multinomial

Create a random variable for Multinomial.

``` python
tfp.edward2.Multinomial(
    *args,
    **kwargs
)
```



Defined in [`python/edward2/interceptor.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/edward2/interceptor.py).

<!-- Placeholder for "Used in" -->

See Multinomial for more details.

#### Returns:

  RandomVariable.

#### Original Docstring for Distribution

Initialize a batch of Multinomial distributions.


#### Args:

* <b>`total_count`</b>: Non-negative floating point tensor with shape broadcastable
  to `[N1,..., Nm]` with `m >= 0`. Defines this as a batch of
  `N1 x ... x Nm` different Multinomial distributions. Its components
  should be equal to integer values.
* <b>`logits`</b>: Floating point tensor representing unnormalized log-probabilities
  of a positive event with shape broadcastable to
  `[N1,..., Nm, K]` `m >= 0`, and the same dtype as `total_count`. Defines
  this as a batch of `N1 x ... x Nm` different `K` class Multinomial
  distributions. Only one of `logits` or `probs` should be passed in.
* <b>`probs`</b>: Positive floating point tensor with shape broadcastable to
  `[N1,..., Nm, K]` `m >= 0` and same dtype as `total_count`. Defines
  this as a batch of `N1 x ... x Nm` different `K` class Multinomial
  distributions. `probs`'s components in the last portion of its shape
  should sum to `1`. Only one of `logits` or `probs` should be passed in.
* <b>`validate_args`</b>: Python `bool`, default `False`. When `True` distribution
  parameters are checked for validity despite possibly degrading runtime
  performance. When `False` invalid inputs may silently render incorrect
  outputs.
* <b>`allow_nan_stats`</b>: Python `bool`, default `True`. When `True`, statistics
  (e.g., mean, mode, variance) use the value "`NaN`" to indicate the
  result is undefined. When `False`, an exception is raised if one or
  more of the statistic's batch members are undefined.
* <b>`name`</b>: Python `str` name prefixed to Ops created by this class.