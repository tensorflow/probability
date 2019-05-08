<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.edward2.Triangular" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.edward2.Triangular

Create a random variable for Triangular.

``` python
tfp.edward2.Triangular(
    *args,
    **kwargs
)
```



Defined in [`python/edward2/interceptor.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/edward2/interceptor.py).

<!-- Placeholder for "Used in" -->

See Triangular for more details.

#### Returns:

  RandomVariable.

#### Original Docstring for Distribution

Initialize a batch of Triangular distributions.


#### Args:

* <b>`low`</b>: Floating point tensor, lower boundary of the output interval. Must
  have `low < high`.
  Default value: `0`.
* <b>`high`</b>: Floating point tensor, upper boundary of the output interval. Must
  have `low < high`.
  Default value: `1`.
* <b>`peak`</b>: Floating point tensor, mode of the output interval. Must have
  `low <= peak` and `peak <= high`.
  Default value: `0.5`.
* <b>`validate_args`</b>: Python `bool`, default `False`. When `True` distribution
  parameters are checked for validity despite possibly degrading runtime
  performance. When `False` invalid inputs may silently render incorrect
  outputs.
  Default value: `False`.
* <b>`allow_nan_stats`</b>: Python `bool`, default `True`. When `True`, statistics
  (e.g., mean, mode, variance) use the value "`NaN`" to indicate the
  result is undefined. When `False`, an exception is raised if one or
  more of the statistic's batch members are undefined.
  Default value: `True`.
* <b>`name`</b>: Python `str` name prefixed to Ops created by this class.
  Default value: `'Triangular'`.


#### Raises:

* <b>`InvalidArgumentError`</b>: if `validate_args=True` and one of the following is
  True:
  * `low >= high`.
  * `peak > high`.
  * `low > peak`.