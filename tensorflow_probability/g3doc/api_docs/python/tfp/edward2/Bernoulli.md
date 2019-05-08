<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.edward2.Bernoulli" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.edward2.Bernoulli

Create a random variable for Bernoulli.

``` python
tfp.edward2.Bernoulli(
    *args,
    **kwargs
)
```



Defined in [`python/edward2/interceptor.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/edward2/interceptor.py).

<!-- Placeholder for "Used in" -->

See Bernoulli for more details.

#### Returns:

  RandomVariable.

#### Original Docstring for Distribution

Construct Bernoulli distributions.


#### Args:

* <b>`logits`</b>: An N-D `Tensor` representing the log-odds of a `1` event. Each
  entry in the `Tensor` parametrizes an independent Bernoulli distribution
  where the probability of an event is sigmoid(logits). Only one of
  `logits` or `probs` should be passed in.
* <b>`probs`</b>: An N-D `Tensor` representing the probability of a `1`
  event. Each entry in the `Tensor` parameterizes an independent
  Bernoulli distribution. Only one of `logits` or `probs` should be passed
  in.
* <b>`dtype`</b>: The type of the event samples. Default: `int32`.
* <b>`validate_args`</b>: Python `bool`, default `False`. When `True` distribution
  parameters are checked for validity despite possibly degrading runtime
  performance. When `False` invalid inputs may silently render incorrect
  outputs.
* <b>`allow_nan_stats`</b>: Python `bool`, default `True`. When `True`,
  statistics (e.g., mean, mode, variance) use the value "`NaN`" to
  indicate the result is undefined. When `False`, an exception is raised
  if one or more of the statistic's batch members are undefined.
* <b>`name`</b>: Python `str` name prefixed to Ops created by this class.


#### Raises:

* <b>`ValueError`</b>: If p and logits are passed, or if neither are passed.