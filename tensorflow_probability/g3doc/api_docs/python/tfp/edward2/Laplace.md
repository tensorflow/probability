<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.edward2.Laplace" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.edward2.Laplace

Create a random variable for Laplace.

``` python
tfp.edward2.Laplace(
    *args,
    **kwargs
)
```



Defined in [`python/edward2/interceptor.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/edward2/interceptor.py).

<!-- Placeholder for "Used in" -->

See Laplace for more details.

#### Returns:

RandomVariable.


#### Original Docstring for Distribution

Construct Laplace distribution with parameters `loc` and `scale`.

The parameters `loc` and `scale` must be shaped in a way that supports
broadcasting (e.g., `loc / scale` is a valid operation).

#### Args:


* <b>`loc`</b>: Floating point tensor which characterizes the location (center)
  of the distribution.
* <b>`scale`</b>: Positive floating point tensor which characterizes the spread of
  the distribution.
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


* <b>`TypeError`</b>: if `loc` and `scale` are of different dtype.