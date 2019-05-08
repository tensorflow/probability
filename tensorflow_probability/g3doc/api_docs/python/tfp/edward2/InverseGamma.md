<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.edward2.InverseGamma" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.edward2.InverseGamma

Create a random variable for InverseGamma.

``` python
tfp.edward2.InverseGamma(
    *args,
    **kwargs
)
```



Defined in [`python/edward2/interceptor.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/edward2/interceptor.py).

<!-- Placeholder for "Used in" -->

See InverseGamma for more details.

#### Returns:

  RandomVariable.

#### Original Docstring for Distribution

Construct InverseGamma with `concentration` and `scale` parameters. (deprecated arguments)

* <b>`Warning`</b>: SOME ARGUMENTS ARE DEPRECATED: `(rate)`. They will be removed after 2019-05-08.
Instructions for updating:
The `rate` parameter is deprecated. Use `scale` instead.The `rate` parameter was always interpreted as a `scale` parameter, but erroneously misnamed.

The parameters `concentration` and `scale` must be shaped in a way that
supports broadcasting (e.g. `concentration + scale` is a valid operation).


#### Args:

* <b>`concentration`</b>: Floating point tensor, the concentration params of the
  distribution(s). Must contain only positive values.
* <b>`scale`</b>: Floating point tensor, the scale params of the distribution(s).
  Must contain only positive values.
* <b>`validate_args`</b>: Python `bool`, default `False`. When `True` distribution
  parameters are checked for validity despite possibly degrading runtime
  performance. When `False` invalid inputs may silently render incorrect
  outputs.
* <b>`allow_nan_stats`</b>: Python `bool`, default `True`. When `True`, statistics
  (e.g., mean, mode, variance) use the value "`NaN`" to indicate the
  result is undefined. When `False`, an exception is raised if one or
  more of the statistic's batch members are undefined.
* <b>`rate`</b>: Deprecated (mis-named) alias for `scale`.
* <b>`name`</b>: Python `str` name prefixed to Ops created by this class.



#### Raises:

* <b>`TypeError`</b>: if `concentration` and `scale` are different dtypes.