<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.edward2.Logistic" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.edward2.Logistic

Create a random variable for Logistic.

``` python
tfp.edward2.Logistic(
    *args,
    **kwargs
)
```



Defined in [`python/edward2/interceptor.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/edward2/interceptor.py).

<!-- Placeholder for "Used in" -->

See Logistic for more details.

#### Returns:

  RandomVariable.

#### Original Docstring for Distribution

Construct Logistic distributions with mean and scale `loc` and `scale`.

The parameters `loc` and `scale` must be shaped in a way that supports
broadcasting (e.g. `loc + scale` is a valid operation).


#### Args:

* <b>`loc`</b>: Floating point tensor, the means of the distribution(s).
* <b>`scale`</b>: Floating point tensor, the scales of the distribution(s). Must
  contain only positive values.
* <b>`validate_args`</b>: Python `bool`, default `False`. When `True` distribution
  parameters are checked for validity despite possibly degrading runtime
  performance. When `False` invalid inputs may silently render incorrect
  outputs.
* <b>`allow_nan_stats`</b>: Python `bool`, default `True`. When `True`, statistics
  (e.g., mean, mode, variance) use the value "`NaN`" to indicate the
  result is undefined. When `False`, an exception is raised if one or
  more of the statistic's batch members are undefined.
* <b>`name`</b>: The name to give Ops created by the initializer.


#### Raises:

* <b>`TypeError`</b>: if loc and scale are different dtypes.