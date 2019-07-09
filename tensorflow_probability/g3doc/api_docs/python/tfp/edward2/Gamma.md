<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.edward2.Gamma" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.edward2.Gamma

Create a random variable for Gamma.

``` python
tfp.edward2.Gamma(
    *args,
    **kwargs
)
```



Defined in [`python/edward2/interceptor.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/edward2/interceptor.py).

<!-- Placeholder for "Used in" -->

See Gamma for more details.

#### Returns:

RandomVariable.


#### Original Docstring for Distribution

Construct Gamma with `concentration` and `rate` parameters.

The parameters `concentration` and `rate` must be shaped in a way that
supports broadcasting (e.g. `concentration + rate` is a valid operation).

#### Args:


* <b>`concentration`</b>: Floating point tensor, the concentration params of the
  distribution(s). Must contain only positive values.
* <b>`rate`</b>: Floating point tensor, the inverse scale params of the
  distribution(s). Must contain only positive values.
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


* <b>`TypeError`</b>: if `concentration` and `rate` are different dtypes.