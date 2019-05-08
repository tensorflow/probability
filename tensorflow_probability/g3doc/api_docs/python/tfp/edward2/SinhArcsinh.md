<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.edward2.SinhArcsinh" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.edward2.SinhArcsinh

Create a random variable for SinhArcsinh.

``` python
tfp.edward2.SinhArcsinh(
    *args,
    **kwargs
)
```



Defined in [`python/edward2/interceptor.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/edward2/interceptor.py).

<!-- Placeholder for "Used in" -->

See SinhArcsinh for more details.

#### Returns:

  RandomVariable.

#### Original Docstring for Distribution

Construct SinhArcsinh distribution on `(-inf, inf)`.

Arguments `(loc, scale, skewness, tailweight)` must have broadcastable shape
(indexing batch dimensions).  They must all have the same `dtype`.


#### Args:

* <b>`loc`</b>: Floating-point `Tensor`.
* <b>`scale`</b>:  `Tensor` of same `dtype` as `loc`.
* <b>`skewness`</b>:  Skewness parameter.  Default is `0.0` (no skew).
* <b>`tailweight`</b>:  Tailweight parameter. Default is `1.0` (unchanged tailweight)
* <b>`distribution`</b>: `tf.Distribution`-like instance. Distribution that is
  transformed to produce this distribution.
  Default is `tfd.Normal(0., 1.)`.
  Must be a scalar-batch, scalar-event distribution.  Typically
  `distribution.reparameterization_type = FULLY_REPARAMETERIZED` or it is
  a function of non-trainable parameters. WARNING: If you backprop through
  a `SinhArcsinh` sample and `distribution` is not
  `FULLY_REPARAMETERIZED` yet is a function of trainable variables, then
  the gradient will be incorrect!
* <b>`validate_args`</b>: Python `bool`, default `False`. When `True` distribution
  parameters are checked for validity despite possibly degrading runtime
  performance. When `False` invalid inputs may silently render incorrect
  outputs.
* <b>`allow_nan_stats`</b>: Python `bool`, default `True`. When `True`,
  statistics (e.g., mean, mode, variance) use the value "`NaN`" to
  indicate the result is undefined. When `False`, an exception is raised
  if one or more of the statistic's batch members are undefined.
* <b>`name`</b>: Python `str` name prefixed to Ops created by this class.