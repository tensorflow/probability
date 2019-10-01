<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.edward2.TruncatedNormal" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.edward2.TruncatedNormal


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/experimental/edward2/interceptor.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Create a random variable for TruncatedNormal.

### Aliases:

* `tfp.experimental.edward2.TruncatedNormal`


``` python
tfp.edward2.TruncatedNormal(
    *args,
    **kwargs
)
```



<!-- Placeholder for "Used in" -->

See TruncatedNormal for more details.

#### Returns:

RandomVariable.


#### Original Docstring for Distribution

Construct TruncatedNormal.

All parameters of the distribution will be broadcast to the same shape,
so the resulting distribution will have a batch_shape of the broadcast
shape of all parameters.

#### Args:


* <b>`loc`</b>: Floating point tensor; the mean of the normal distribution(s) (
  note that the mean of the resulting distribution will be different
  since it is modified by the bounds).
* <b>`scale`</b>: Floating point tensor; the std deviation of the normal
  distribution(s).
* <b>`low`</b>: `float` `Tensor` representing lower bound of the distribution's
  support. Must be such that `low < high`.
* <b>`high`</b>: `float` `Tensor` representing upper bound of the distribution's
  support. Must be such that `low < high`.
* <b>`validate_args`</b>: Python `bool`, default `False`. When `True` distribution
  parameters are checked at run-time.
* <b>`allow_nan_stats`</b>: Python `bool`, default `True`. When `True`,
  statistics (e.g., mean, mode, variance) use the value '`NaN`' to
  indicate the result is undefined. When `False`, an exception is raised
  if one or more of the statistic's batch members are undefined.
* <b>`name`</b>: Python `str` name prefixed to Ops created by this class.