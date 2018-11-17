<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.edward2.MixtureSameFamily" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.edward2.MixtureSameFamily

``` python
tfp.edward2.MixtureSameFamily(
    *args,
    **kwargs
)
```

Create a random variable for MixtureSameFamily.

See MixtureSameFamily for more details.

#### Returns:

  RandomVariable.

#### Original Docstring for Distribution

Construct a `MixtureSameFamily` distribution.


#### Args:

* <b>`mixture_distribution`</b>: <a href="../../tfp/distributions/Categorical.md"><code>tfp.distributions.Categorical</code></a>-like instance.
    Manages the probability of selecting components. The number of
    categories must match the rightmost batch dimension of the
    `components_distribution`. Must have either scalar `batch_shape` or
    `batch_shape` matching `components_distribution.batch_shape[:-1]`.
* <b>`components_distribution`</b>: <a href="../../tfp/distributions/Distribution.md"><code>tfp.distributions.Distribution</code></a>-like instance.
    Right-most batch dimension indexes components.
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

* <b>`ValueError`</b>: `if not mixture_distribution.dtype.is_integer`.
* <b>`ValueError`</b>: if mixture_distribution does not have scalar `event_shape`.
* <b>`ValueError`</b>: if `mixture_distribution.batch_shape` and
    `components_distribution.batch_shape[:-1]` are both fully defined and
    the former is neither scalar nor equal to the latter.
* <b>`ValueError`</b>: if `mixture_distribution` categories does not equal
    `components_distribution` rightmost batch shape.