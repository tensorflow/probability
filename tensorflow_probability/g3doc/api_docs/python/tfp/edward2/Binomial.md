<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.edward2.Binomial" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.edward2.Binomial

Create a random variable for Binomial.

``` python
tfp.edward2.Binomial(
    *args,
    **kwargs
)
```



Defined in [`python/edward2/interceptor.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/edward2/interceptor.py).

<!-- Placeholder for "Used in" -->

See Binomial for more details.

#### Returns:

  RandomVariable.

#### Original Docstring for Distribution

Initialize a batch of Binomial distributions.


#### Args:

* <b>`total_count`</b>: Non-negative floating point tensor with shape broadcastable
  to `[N1,..., Nm]` with `m >= 0` and the same dtype as `probs` or
  `logits`. Defines this as a batch of `N1 x ...  x Nm` different Binomial
  distributions. Its components should be equal to integer values.
* <b>`logits`</b>: Floating point tensor representing the log-odds of a
  positive event with shape broadcastable to `[N1,..., Nm]` `m >= 0`, and
  the same dtype as `total_count`. Each entry represents logits for the
  probability of success for independent Binomial distributions. Only one
  of `logits` or `probs` should be passed in.
* <b>`probs`</b>: Positive floating point tensor with shape broadcastable to
  `[N1,..., Nm]` `m >= 0`, `probs in [0, 1]`. Each entry represents the
  probability of success for independent Binomial distributions. Only one
  of `logits` or `probs` should be passed in.
* <b>`validate_args`</b>: Python `bool`, default `False`. When `True` distribution
  parameters are checked for validity despite possibly degrading runtime
  performance. When `False` invalid inputs may silently render incorrect
  outputs.
* <b>`allow_nan_stats`</b>: Python `bool`, default `True`. When `True`, statistics
  (e.g., mean, mode, variance) use the value "`NaN`" to indicate the
  result is undefined. When `False`, an exception is raised if one or
  more of the statistic's batch members are undefined.
* <b>`name`</b>: Python `str` name prefixed to Ops created by this class.