<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.edward2.VonMises" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.edward2.VonMises

``` python
tfp.edward2.VonMises(
    *args,
    **kwargs
)
```

Create a random variable for VonMises.

See VonMises for more details.

#### Returns:

  RandomVariable.

#### Original Docstring for Distribution

Construct von Mises distributions with given location and concentration.

The parameters `loc` and `concentration` must be shaped in a way that
supports broadcasting (e.g. `loc + concentration` is a valid operation).


#### Args:

* <b>`loc`</b>: Floating point tensor, the circular means of the distribution(s).
* <b>`concentration`</b>: Floating point tensor, the level of concentration of the
    distribution(s) around `loc`. Must take non-negative values.
    `concentration = 0` defines a Uniform distribution, while
    `concentration = +inf` indicates a Deterministic distribution at `loc`.
* <b>`validate_args`</b>: Python `bool`, default `False`. When `True` distribution
    parameters are checked for validity despite possibly degrading runtime
    performance. When `False` invalid inputs may silently render incorrect
    outputs.
* <b>`allow_nan_stats`</b>: Python `bool`, default `True`. When `True`, statistics
    (e.g., mean, mode, variance) use the value "`NaN`" to indicate the
    result is undefined. When `False`, an exception is raised if one or more
    of the statistic's batch members are undefined.
* <b>`name`</b>: Python `str` name prefixed to Ops created by this class.


#### Raises:

* <b>`TypeError`</b>: if loc and concentration are different dtypes.