<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.edward2.VonMisesFisher" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.edward2.VonMisesFisher


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/experimental/edward2/interceptor.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Create a random variable for VonMisesFisher.

### Aliases:

* `tfp.experimental.edward2.VonMisesFisher`


``` python
tfp.edward2.VonMisesFisher(
    *args,
    **kwargs
)
```



<!-- Placeholder for "Used in" -->

See VonMisesFisher for more details.

#### Returns:

RandomVariable.


#### Original Docstring for Distribution

Creates a new `VonMisesFisher` instance.

#### Args:


* <b>`mean_direction`</b>: Floating-point `Tensor` with shape [B1, ... Bn, D].
  A unit vector indicating the mode of the distribution, or the
  unit-normalized direction of the mean. (This is *not* in general the
  mean of the distribution; the mean is not generally in the support of
  the distribution.) NOTE: `D` is currently restricted to <= 5.
* <b>`concentration`</b>: Floating-point `Tensor` having batch shape [B1, ... Bn]
  broadcastable with `mean_direction`. The level of concentration of
  samples around the `mean_direction`. `concentration=0` indicates a
  uniform distribution over the unit hypersphere, and `concentration=+inf`
  indicates a `Deterministic` distribution (delta function) at
  `mean_direction`.
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


* <b>`ValueError`</b>: For known-bad arguments, i.e. unsupported event dimension.