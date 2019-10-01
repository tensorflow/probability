<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.edward2.HalfCauchy" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.edward2.HalfCauchy


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/experimental/edward2/interceptor.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Create a random variable for HalfCauchy.

### Aliases:

* `tfp.experimental.edward2.HalfCauchy`


``` python
tfp.edward2.HalfCauchy(
    *args,
    **kwargs
)
```



<!-- Placeholder for "Used in" -->

See HalfCauchy for more details.

#### Returns:

RandomVariable.


#### Original Docstring for Distribution

Construct a half-Cauchy distribution with `loc` and `scale`.

#### Args:


* <b>`loc`</b>: Floating-point `Tensor`; the location(s) of the distribution(s).
* <b>`scale`</b>: Floating-point `Tensor`; the scale(s) of the distribution(s).
  Must contain only positive values.
* <b>`validate_args`</b>: Python `bool`, default `False`. When `True` distribution
  parameters are checked for validity despite possibly degrading runtime
  performance. When `False` invalid inputs may silently render incorrect
  outputs. Default value: `False` (i.e. do not validate args).
* <b>`allow_nan_stats`</b>: Python `bool`, default `True`. When `True`, statistics
  (e.g., mean, mode, variance) use the value "`NaN`" to indicate the
  result is undefined. When `False`, an exception is raised if one or
  more of the statistic's batch members are undefined.
  Default value: `True`.
* <b>`name`</b>: Python `str` name prefixed to Ops created by this class.
  Default value: 'HalfCauchy'.


#### Raises:


* <b>`TypeError`</b>: if `loc` and `scale` have different `dtype`.