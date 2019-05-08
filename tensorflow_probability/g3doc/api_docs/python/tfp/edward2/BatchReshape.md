<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.edward2.BatchReshape" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.edward2.BatchReshape

Create a random variable for BatchReshape.

``` python
tfp.edward2.BatchReshape(
    *args,
    **kwargs
)
```



Defined in [`python/edward2/interceptor.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/edward2/interceptor.py).

<!-- Placeholder for "Used in" -->

See BatchReshape for more details.

#### Returns:

  RandomVariable.

#### Original Docstring for Distribution

Construct BatchReshape distribution.


#### Args:

* <b>`distribution`</b>: The base distribution instance to reshape. Typically an
  instance of `Distribution`.
* <b>`batch_shape`</b>: Positive `int`-like vector-shaped `Tensor` representing
  the new shape of the batch dimensions. Up to one dimension may contain
  `-1`, meaning the remainder of the batch size.
* <b>`validate_args`</b>: Python `bool`, default `False`. When `True` distribution
  parameters are checked for validity despite possibly degrading runtime
  performance. When `False` invalid inputs may silently render incorrect
  outputs.
* <b>`allow_nan_stats`</b>: Python `bool`, default `True`. When `True`, statistics
  (e.g., mean, mode, variance) use the value "`NaN`" to indicate the
  result is undefined. When `False`, an exception is raised if one or
  more of the statistic's batch members are undefined.
* <b>`name`</b>: The name to give Ops created by the initializer.
  Default value: `"BatchReshape" + distribution.name`.


#### Raises:

* <b>`ValueError`</b>: if `batch_shape` is not a vector.
* <b>`ValueError`</b>: if `batch_shape` has non-positive elements.
* <b>`ValueError`</b>: if `batch_shape` size is not the same as a
  `distribution.batch_shape` size.