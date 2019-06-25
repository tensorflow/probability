<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.edward2.Gumbel" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.edward2.Gumbel

Create a random variable for Gumbel.

``` python
tfp.edward2.Gumbel(
    *args,
    **kwargs
)
```



Defined in [`python/edward2/interceptor.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/edward2/interceptor.py).

<!-- Placeholder for "Used in" -->

See Gumbel for more details.

#### Returns:

RandomVariable.


#### Original Docstring for Distribution

Construct Gumbel distributions with location and scale `loc` and `scale`.

The parameters `loc` and `scale` must be shaped in a way that supports
broadcasting (e.g. `loc + scale` is a valid operation).

#### Args:


* <b>`loc`</b>: Floating point tensor, the means of the distribution(s).
* <b>`scale`</b>: Floating point tensor, the scales of the distribution(s).
  scale must contain only positive values.
* <b>`validate_args`</b>: Python `bool`, default `False`. When `True` distribution
  parameters are checked for validity despite possibly degrading runtime
  performance. When `False` invalid inputs may silently render incorrect
  outputs.
  Default value: `False`.
* <b>`allow_nan_stats`</b>: Python `bool`, default `True`. When `True`,
  statistics (e.g., mean, mode, variance) use the value "`NaN`" to
  indicate the result is undefined. When `False`, an exception is raised
  if one or more of the statistic's batch members are undefined.
  Default value: `True`.
* <b>`name`</b>: Python `str` name prefixed to Ops created by this class.
  Default value: `'Gumbel'`.


#### Raises:


* <b>`TypeError`</b>: if loc and scale are different dtypes.