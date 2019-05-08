<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.edward2.Pareto" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.edward2.Pareto

Create a random variable for Pareto.

``` python
tfp.edward2.Pareto(
    *args,
    **kwargs
)
```



Defined in [`python/edward2/interceptor.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/edward2/interceptor.py).

<!-- Placeholder for "Used in" -->

See Pareto for more details.

#### Returns:

  RandomVariable.

#### Original Docstring for Distribution

Construct Pareto distribution with `concentration` and `scale`.


#### Args:

* <b>`concentration`</b>: Floating point tensor. Must contain only positive values.
* <b>`scale`</b>: Floating point tensor, equivalent to `mode`. `scale` also
  restricts the domain of this distribution to be in `[scale, inf)`.
  Must contain only positive values. Default value: `1`.
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
  Default value: 'Pareto'.