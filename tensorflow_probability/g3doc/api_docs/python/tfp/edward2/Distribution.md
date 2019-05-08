<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.edward2.Distribution" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.edward2.Distribution

Create a random variable for Distribution.

``` python
tfp.edward2.Distribution(
    *args,
    **kwargs
)
```



Defined in [`python/edward2/interceptor.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/edward2/interceptor.py).

<!-- Placeholder for "Used in" -->

See Distribution for more details.

#### Returns:

  RandomVariable.

#### Original Docstring for Distribution

Constructs the `Distribution`.

**This is a private method for subclass use.**


#### Args:

* <b>`dtype`</b>: The type of the event samples. `None` implies no type-enforcement.
* <b>`reparameterization_type`</b>: Instance of `ReparameterizationType`.
  If `tfd.FULLY_REPARAMETERIZED`, this
  `Distribution` can be reparameterized in terms of some standard
  distribution with a function whose Jacobian is constant for the support
  of the standard distribution. If `tfd.NOT_REPARAMETERIZED`,
  then no such reparameterization is available.
* <b>`validate_args`</b>: Python `bool`, default `False`. When `True` distribution
  parameters are checked for validity despite possibly degrading runtime
  performance. When `False` invalid inputs may silently render incorrect
  outputs.
* <b>`allow_nan_stats`</b>: Python `bool`, default `True`. When `True`, statistics
  (e.g., mean, mode, variance) use the value "`NaN`" to indicate the
  result is undefined. When `False`, an exception is raised if one or
  more of the statistic's batch members are undefined.
* <b>`parameters`</b>: Python `dict` of parameters used to instantiate this
  `Distribution`.
* <b>`graph_parents`</b>: Python `list` of graph prerequisites of this
  `Distribution`.
* <b>`name`</b>: Python `str` name prefixed to Ops created by this class. Default:
  subclass name.


#### Raises:

* <b>`ValueError`</b>: if any member of graph_parents is `None` or not a `Tensor`.