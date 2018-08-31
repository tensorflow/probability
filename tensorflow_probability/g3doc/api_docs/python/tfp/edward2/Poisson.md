<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.edward2.Poisson" />
</div>

# tfp.edward2.Poisson

``` python
tfp.edward2.Poisson(
    *args,
    **kwargs
)
```

Create a random variable for Poisson.

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
* <b>`validate_args`</b>: Python `bool`, default `False`. When `True` distribution
    parameters are checked for validity despite possibly degrading runtime
    performance. When `False` invalid inputs may silently render incorrect
    outputs.
* <b>`allow_nan_stats`</b>: Python `bool`, default `True`. When `True`, statistics
    (e.g., mean, mode, variance) use the value "`NaN`" to indicate the
    result is undefined. When `False`, an exception is raised if one or
    more of the statistic's batch members are undefined.
* <b>`name`</b>: Python `str` name prefixed to Ops created by this class.


#### Raises:

* <b>`ValueError`</b>: if none or both of `rate`, `log_rate` are specified.
* <b>`TypeError`</b>: if `rate` is not a float-type.
* <b>`TypeError`</b>: if `log_rate` is not a float-type.