<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.edward2.Zipf" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.edward2.Zipf

Create a random variable for Zipf.

``` python
tfp.edward2.Zipf(
    *args,
    **kwargs
)
```



Defined in [`python/edward2/interceptor.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/edward2/interceptor.py).

<!-- Placeholder for "Used in" -->

See Zipf for more details.

#### Returns:

  RandomVariable.

#### Original Docstring for Distribution

Initialize a batch of Zipf distributions.


#### Args:

* <b>`power`</b>: `Float` like `Tensor` representing the power parameter. Must be
  strictly greater than `1`.
* <b>`dtype`</b>: The `dtype` of `Tensor` returned by `sample`.
  Default value: `tf.int32`.
* <b>`interpolate_nondiscrete`</b>: Python `bool`. When `False`, `log_prob` returns
  `-inf` (and `prob` returns `0`) for non-integer inputs. When `True`,
  `log_prob` evaluates the continuous function `-power log(k) -
  log(zeta(power))` , which matches the Zipf pmf at integer arguments `k`
  (note that this function is not itself a normalized probability
  log-density).
  Default value: `True`.
* <b>`sample_maximum_iterations`</b>: Maximum number of iterations of allowable
  iterations in `sample`. When `validate_args=True`, samples which fail to
  reach convergence (subject to this cap) are masked out with
  `self.dtype.min` or `nan` depending on `self.dtype.is_integer`.
  Default value: `100`.
* <b>`validate_args`</b>: Python `bool`, default `False`. When `True` distribution
  parameters are checked for validity despite possibly degrading runtime
  performance. When `False` invalid inputs may silently render incorrect
  outputs.
  Default value: `False`.
* <b>`allow_nan_stats`</b>: Python `bool`, default `True`. When `True`, statistics
  (e.g., mean, mode, variance) use the value "`NaN`" to indicate the
  result is undefined. When `False`, an exception is raised if one or more
  of the statistic's batch members are undefined.
  Default value: `False`.
* <b>`name`</b>: Python `str` name prefixed to Ops created by this class.
  Default value: `'Zipf'`.


#### Raises:

* <b>`TypeError`</b>: if `power` is not `float` like.