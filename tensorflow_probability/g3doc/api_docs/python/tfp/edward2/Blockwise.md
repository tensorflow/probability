<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.edward2.Blockwise" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.edward2.Blockwise


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/experimental/edward2/interceptor.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Create a random variable for Blockwise.

### Aliases:

* `tfp.experimental.edward2.Blockwise`


``` python
tfp.edward2.Blockwise(
    *args,
    **kwargs
)
```



<!-- Placeholder for "Used in" -->

See Blockwise for more details.

#### Returns:

RandomVariable.


#### Original Docstring for Distribution

Construct the `Blockwise` distribution.

#### Args:


* <b>`distributions`</b>: Python `list` of <a href="../../tfp/distributions/Distribution.md"><code>tfp.distributions.Distribution</code></a>
  instances. All distribution instances must have the same `batch_shape`
  and all must have `event_ndims==1`, i.e., be vector-variate
  distributions.
* <b>`dtype_override`</b>: samples of `distributions` will be cast to this `dtype`.
  If unspecified, all `distributions` must have the same `dtype`.
  Default value: `None` (i.e., do not cast).
* <b>`validate_args`</b>: Python `bool`, default `False`. When `True` distribution
  parameters are checked for validity despite possibly degrading runtime
  performance. When `False` invalid inputs may silently render incorrect
  outputs.
* <b>`allow_nan_stats`</b>: Python `bool`, default `True`. When `True`, statistics
  (e.g., mean, mode, variance) use the value "`NaN`" to indicate the
  result is undefined. When `False`, an exception is raised if one or more
  of the statistic's batch members are undefined.
* <b>`name`</b>: Python `str` name prefixed to Ops created by this class.