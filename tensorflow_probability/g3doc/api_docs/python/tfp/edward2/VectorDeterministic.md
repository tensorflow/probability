<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.edward2.VectorDeterministic" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.edward2.VectorDeterministic

Create a random variable for VectorDeterministic.

``` python
tfp.edward2.VectorDeterministic(
    *args,
    **kwargs
)
```



Defined in [`python/edward2/interceptor.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/edward2/interceptor.py).

<!-- Placeholder for "Used in" -->

See VectorDeterministic for more details.

#### Returns:

  RandomVariable.

#### Original Docstring for Distribution

Initialize a `VectorDeterministic` distribution on `R^k`, for `k >= 0`.

Note that there is only one point in `R^0`, the "point" `[]`.  So if `k = 0`
then `self.prob([]) == 1`.

The `atol` and `rtol` parameters allow for some slack in `pmf`
computations, e.g. due to floating-point error.

```
pmf(x; loc)
  = 1, if All[Abs(x - loc) <= atol + rtol * Abs(loc)],
  = 0, otherwise
```


#### Args:

* <b>`loc`</b>: Numeric `Tensor` of shape `[B1, ..., Bb, k]`, with `b >= 0`, `k >= 0`
  The point (or batch of points) on which this distribution is supported.
* <b>`atol`</b>:  Non-negative `Tensor` of same `dtype` as `loc` and broadcastable
  shape.  The absolute tolerance for comparing closeness to `loc`.
  Default is `0`.
* <b>`rtol`</b>:  Non-negative `Tensor` of same `dtype` as `loc` and broadcastable
  shape.  The relative tolerance for comparing closeness to `loc`.
  Default is `0`.
* <b>`validate_args`</b>: Python `bool`, default `False`. When `True` distribution
  parameters are checked for validity despite possibly degrading runtime
  performance. When `False` invalid inputs may silently render incorrect
  outputs.
* <b>`allow_nan_stats`</b>: Python `bool`, default `True`. When `True`, statistics
  (e.g., mean, mode, variance) use the value "`NaN`" to indicate the
  result is undefined. When `False`, an exception is raised if one or
  more of the statistic's batch members are undefined.
* <b>`name`</b>: Python `str` name prefixed to Ops created by this class.