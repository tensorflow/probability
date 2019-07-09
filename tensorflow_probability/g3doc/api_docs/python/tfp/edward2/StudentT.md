<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.edward2.StudentT" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.edward2.StudentT

Create a random variable for StudentT.

``` python
tfp.edward2.StudentT(
    *args,
    **kwargs
)
```



Defined in [`python/edward2/interceptor.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/edward2/interceptor.py).

<!-- Placeholder for "Used in" -->

See StudentT for more details.

#### Returns:

RandomVariable.


#### Original Docstring for Distribution

Construct Student's t distributions.

The distributions have degree of freedom `df`, mean `loc`, and scale
`scale`.

The parameters `df`, `loc`, and `scale` must be shaped in a way that
supports broadcasting (e.g. `df + loc + scale` is a valid operation).

#### Args:


* <b>`df`</b>: Floating-point `Tensor`. The degrees of freedom of the
  distribution(s). `df` must contain only positive values.
* <b>`loc`</b>: Floating-point `Tensor`. The mean(s) of the distribution(s).
* <b>`scale`</b>: Floating-point `Tensor`. The scaling factor(s) for the
  distribution(s). Note that `scale` is not technically the standard
  deviation of this distribution but has semantics more similar to
  standard deviation than variance.
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


* <b>`TypeError`</b>: if loc and scale are different dtypes.