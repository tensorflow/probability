<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.edward2.GeneralizedPareto" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.edward2.GeneralizedPareto


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/experimental/edward2/interceptor.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Create a random variable for GeneralizedPareto.

### Aliases:

* `tfp.experimental.edward2.GeneralizedPareto`


``` python
tfp.edward2.GeneralizedPareto(
    *args,
    **kwargs
)
```



<!-- Placeholder for "Used in" -->

See GeneralizedPareto for more details.

#### Returns:

RandomVariable.


#### Original Docstring for Distribution

Construct a Generalized Pareto distribution.

#### Args:


* <b>`loc`</b>: The location / shift of the distribution. GeneralizedPareto is a
  location-scale distribution. This parameter lower bounds the
  distribution's support. Must broadcast with `scale`, `concentration`.
  Floating point `Tensor`.
* <b>`scale`</b>: The scale of the distribution. GeneralizedPareto is a
  location-scale distribution, so doubling the `scale` doubles a sample
  and halves the density. Strictly positive floating point `Tensor`. Must
  broadcast with `loc`, `concentration`.
* <b>`concentration`</b>: The shape parameter of the distribution. The larger the
  magnitude, the more the distribution concentrates near `loc` (for
  `concentration >= 0`) or near `loc - (scale/concentration)` (for
  `concentration < 0`). Floating point `Tensor`.
* <b>`validate_args`</b>: Python `bool`, default `False`. When `True` distribution
  parameters are checked for validity despite possibly degrading runtime
  performance. When `False` invalid inputs may silently render incorrect
  outputs.
* <b>`allow_nan_stats`</b>: Python `bool`, default `True`. When `True`, statistics
  (e.g., mean, variance) use the value "`NaN`" to indicate the result is
  undefined. When `False`, an exception is raised if one or more of the
  statistic's batch members are undefined.
* <b>`name`</b>: Python `str` name prefixed to Ops created by this class.


#### Raises:


* <b>`TypeError`</b>: if `loc`, `scale`, or `concentration` have different dtypes.