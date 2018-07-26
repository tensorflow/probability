Project: /probability/_project.yaml
Book: /probability/_book.yaml
page_type: reference
<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.edward2.Deterministic" />
</div>

# tfp.edward2.Deterministic

``` python
tfp.edward2.Deterministic(
    *args,
    **kwargs
)
```

Create a random variable for Deterministic.

See Deterministic for more details.

#### Returns:

  RandomVariable.

#### Original Docstring for Distribution

Initialize a scalar `Deterministic` distribution.

The `atol` and `rtol` parameters allow for some slack in `pmf`, `cdf`
computations, e.g. due to floating-point error.

```
pmf(x; loc)
  = 1, if Abs(x - loc) <= atol + rtol * Abs(loc),
  = 0, otherwise.
```


#### Args:

* <b>`loc`</b>: Numeric `Tensor` of shape `[B1, ..., Bb]`, with `b >= 0`.
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