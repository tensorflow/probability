<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.edward2.RelaxedBernoulli" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.edward2.RelaxedBernoulli


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/experimental/edward2/interceptor.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Create a random variable for RelaxedBernoulli.

### Aliases:

* `tfp.experimental.edward2.RelaxedBernoulli`


``` python
tfp.edward2.RelaxedBernoulli(
    *args,
    **kwargs
)
```



<!-- Placeholder for "Used in" -->

See RelaxedBernoulli for more details.

#### Returns:

RandomVariable.


#### Original Docstring for Distribution

Construct RelaxedBernoulli distributions.

#### Args:


* <b>`temperature`</b>: A `Tensor`, representing the temperature of a set of
  RelaxedBernoulli distributions. The temperature values should be
  positive.
* <b>`logits`</b>: An N-D `Tensor` representing the log-odds
  of a positive event. Each entry in the `Tensor` parametrizes
  an independent RelaxedBernoulli distribution where the probability of an
  event is sigmoid(logits). Only one of `logits` or `probs` should be
  passed in.
* <b>`probs`</b>: An N-D `Tensor` representing the probability of a positive event.
  Each entry in the `Tensor` parameterizes an independent Bernoulli
  distribution. Only one of `logits` or `probs` should be passed in.
* <b>`validate_args`</b>: Python `bool`, default `False`. When `True` distribution
  parameters are checked for validity despite possibly degrading runtime
  performance. When `False` invalid inputs may silently render incorrect
  outputs.
* <b>`allow_nan_stats`</b>: Python `bool`, default `True`. When `True`, statistics
  (e.g., mean, mode, variance) use the value "`NaN`" to indicate the
  result is undefined. When `False`, an exception is raised if one or
  more of the statistic's batch members are undefined.
* <b>`name`</b>: Python `str` name prefixed to Ops created by this class.


#### Raises:


* <b>`ValueError`</b>: If both `probs` and `logits` are passed, or if neither.