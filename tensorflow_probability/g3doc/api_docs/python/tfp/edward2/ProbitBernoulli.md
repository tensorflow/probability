<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.edward2.ProbitBernoulli" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.edward2.ProbitBernoulli


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/experimental/edward2/interceptor.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Create a random variable for ProbitBernoulli.

### Aliases:

* `tfp.experimental.edward2.ProbitBernoulli`


``` python
tfp.edward2.ProbitBernoulli(
    *args,
    **kwargs
)
```



<!-- Placeholder for "Used in" -->

See ProbitBernoulli for more details.

#### Returns:

RandomVariable.


#### Original Docstring for Distribution

Construct ProbitBernoulli distributions.

#### Args:


* <b>`probits`</b>: An N-D `Tensor` representing the probit-odds of a `1` event. Each
  entry in the `Tensor` parametrizes an independent ProbitBernoulli
  distribution where the probability of an event is normal_cdf(probits).
  Only one of `probits` or `probs` should be passed in.
* <b>`probs`</b>: An N-D `Tensor` representing the probability of a `1`
  event. Each entry in the `Tensor` parameterizes an independent
  ProbitBernoulli distribution. Only one of `probits` or `probs` should be
  passed in.
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


* <b>`ValueError`</b>: If probs and probits are passed, or if neither are passed.