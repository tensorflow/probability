<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.edward2.Poisson" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.edward2.Poisson

Create a random variable for Poisson.

``` python
tfp.edward2.Poisson(
    *args,
    **kwargs
)
```



Defined in [`python/edward2/interceptor.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/edward2/interceptor.py).

<!-- Placeholder for "Used in" -->

See Poisson for more details.

#### Returns:

RandomVariable.


#### Original Docstring for Distribution

Initialize a batch of Poisson distributions.

#### Args:


* <b>`rate`</b>: Floating point tensor, the rate parameter. `rate` must be positive.
  Must specify exactly one of `rate` and `log_rate`.
* <b>`log_rate`</b>: Floating point tensor, the log of the rate parameter.
  Must specify exactly one of `rate` and `log_rate`.
* <b>`interpolate_nondiscrete`</b>: Python `bool`. When `False`,
  `log_prob` returns `-inf` (and `prob` returns `0`) for non-integer
  inputs. When `True`, `log_prob` evaluates the continuous function
  `k * log_rate - lgamma(k+1) - rate`, which matches the Poisson pmf
  at integer arguments `k` (note that this function is not itself
  a normalized probability log-density).
  Default value: `True`.
* <b>`validate_args`</b>: Python `bool`. When `True` distribution
  parameters are checked for validity despite possibly degrading runtime
  performance. When `False` invalid inputs may silently render incorrect
  outputs.
  Default value: `False`.
* <b>`allow_nan_stats`</b>: Python `bool`. When `True`, statistics
  (e.g., mean, mode, variance) use the value "`NaN`" to indicate the
  result is undefined. When `False`, an exception is raised if one or
  more of the statistic's batch members are undefined.
  Default value: `True`.
* <b>`name`</b>: Python `str` name prefixed to Ops created by this class.


#### Raises:


* <b>`ValueError`</b>: if none or both of `rate`, `log_rate` are specified.
* <b>`TypeError`</b>: if `rate` is not a float-type.
* <b>`TypeError`</b>: if `log_rate` is not a float-type.