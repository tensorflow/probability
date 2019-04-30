<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.edward2.HalfCauchy" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.edward2.HalfCauchy

``` python
tfp.edward2.HalfCauchy(
    *args,
    **kwargs
)
```

Create a random variable for HalfCauchy.

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