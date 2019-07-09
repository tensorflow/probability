<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.edward2.OneHotCategorical" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.edward2.OneHotCategorical

Create a random variable for OneHotCategorical.

``` python
tfp.edward2.OneHotCategorical(
    *args,
    **kwargs
)
```



Defined in [`python/edward2/interceptor.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/edward2/interceptor.py).

<!-- Placeholder for "Used in" -->

See OneHotCategorical for more details.

#### Returns:

RandomVariable.


#### Original Docstring for Distribution

Initialize OneHotCategorical distributions using class log-probabilities.

#### Args:


* <b>`logits`</b>: An N-D `Tensor`, `N >= 1`, representing the log probabilities of a
  set of Categorical distributions. The first `N - 1` dimensions index
  into a batch of independent distributions and the last dimension
  represents a vector of logits for each class. Only one of `logits` or
  `probs` should be passed in.
* <b>`probs`</b>: An N-D `Tensor`, `N >= 1`, representing the probabilities of a set
  of Categorical distributions. The first `N - 1` dimensions index into a
  batch of independent distributions and the last dimension represents a
  vector of probabilities for each class. Only one of `logits` or `probs`
  should be passed in.
* <b>`dtype`</b>: The type of the event samples (default: int32).
* <b>`validate_args`</b>: Python `bool`, default `False`. When `True` distribution
  parameters are checked for validity despite possibly degrading runtime
  performance. When `False` invalid inputs may silently render incorrect
  outputs.
* <b>`allow_nan_stats`</b>: Python `bool`, default `True`. When `True`, statistics
  (e.g., mean, mode, variance) use the value "`NaN`" to indicate the
  result is undefined. When `False`, an exception is raised if one or
  more of the statistic's batch members are undefined.
* <b>`name`</b>: Python `str` name prefixed to Ops created by this class.