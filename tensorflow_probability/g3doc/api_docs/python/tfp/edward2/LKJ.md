<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.edward2.LKJ" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.edward2.LKJ

``` python
tfp.edward2.LKJ(
    *args,
    **kwargs
)
```

Create a random variable for LKJ.

See LKJ for more details.

#### Returns:

  RandomVariable.

#### Original Docstring for Distribution

Construct LKJ distributions.


#### Args:

* <b>`dimension`</b>: Python `int`. The dimension of the correlation matrices
    to sample.
* <b>`concentration`</b>: `float` or `double` `Tensor`. The positive concentration
    parameter of the LKJ distributions. The pdf of a sample matrix `X` is
    proportional to `det(X) ** (concentration - 1)`.
* <b>`validate_args`</b>: Python `bool`, default `False`. When `True` distribution
    parameters are checked for validity despite possibly degrading runtime
    performance. When `False` invalid inputs may silently render incorrect
    outputs.
* <b>`allow_nan_stats`</b>: Python `bool`, default `True`. When `True`, statistics
    (e.g., mean, mode, variance) use the value `NaN` to indicate the
    result is undefined. When `False`, an exception is raised if one or
    more of the statistic's batch members are undefined.
* <b>`name`</b>: Python `str` name prefixed to Ops created by this class.


#### Raises:

* <b>`ValueError`</b>: If `dimension` is negative.