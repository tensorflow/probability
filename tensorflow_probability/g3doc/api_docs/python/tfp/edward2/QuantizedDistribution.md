<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.edward2.QuantizedDistribution" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.edward2.QuantizedDistribution

Create a random variable for QuantizedDistribution.

``` python
tfp.edward2.QuantizedDistribution(
    *args,
    **kwargs
)
```



Defined in [`python/edward2/interceptor.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/edward2/interceptor.py).

<!-- Placeholder for "Used in" -->

See QuantizedDistribution for more details.

#### Returns:

  RandomVariable.

#### Original Docstring for Distribution

Construct a Quantized Distribution representing `Y = ceiling(X)`.

Some properties are inherited from the distribution defining `X`. Example:
`allow_nan_stats` is determined for this `QuantizedDistribution` by reading
the `distribution`.


#### Args:

* <b>`distribution`</b>:  The base distribution class to transform. Typically an
  instance of `Distribution`.
* <b>`low`</b>: `Tensor` with same `dtype` as this distribution and shape
  able to be added to samples. Should be a whole number. Default `None`.
  If provided, base distribution's `prob` should be defined at
  `low`.
* <b>`high`</b>: `Tensor` with same `dtype` as this distribution and shape
  able to be added to samples. Should be a whole number. Default `None`.
  If provided, base distribution's `prob` should be defined at
  `high - 1`.
  `high` must be strictly greater than `low`.
* <b>`validate_args`</b>: Python `bool`, default `False`. When `True` distribution
  parameters are checked for validity despite possibly degrading runtime
  performance. When `False` invalid inputs may silently render incorrect
  outputs.
* <b>`name`</b>: Python `str` name prefixed to Ops created by this class.


#### Raises:

* <b>`TypeError`</b>: If `dist_cls` is not a subclass of
    `Distribution` or continuous.
* <b>`NotImplementedError`</b>:  If the base distribution does not implement `cdf`.