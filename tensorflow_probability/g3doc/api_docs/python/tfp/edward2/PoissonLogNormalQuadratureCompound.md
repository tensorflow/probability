<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.edward2.PoissonLogNormalQuadratureCompound" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.edward2.PoissonLogNormalQuadratureCompound

Create a random variable for PoissonLogNormalQuadratureCompound.

``` python
tfp.edward2.PoissonLogNormalQuadratureCompound(
    *args,
    **kwargs
)
```



Defined in [`python/edward2/interceptor.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/edward2/interceptor.py).

<!-- Placeholder for "Used in" -->

See PoissonLogNormalQuadratureCompound for more details.

#### Returns:

RandomVariable.


#### Original Docstring for Distribution

Constructs the PoissonLogNormalQuadratureCompound`.

Note: `probs` returned by (optional) `quadrature_fn` are presumed to be
either a length-`quadrature_size` vector or a batch of vectors in 1-to-1
correspondence with the returned `grid`. (I.e., broadcasting is only
partially supported.)

#### Args:


* <b>`loc`</b>: `float`-like (batch of) scalar `Tensor`; the location parameter of
  the LogNormal prior.
* <b>`scale`</b>: `float`-like (batch of) scalar `Tensor`; the scale parameter of
  the LogNormal prior.
* <b>`quadrature_size`</b>: Python `int` scalar representing the number of quadrature
  points.
* <b>`quadrature_fn`</b>: Python callable taking `loc`, `scale`,
  `quadrature_size`, `validate_args` and returning `tuple(grid, probs)`
  representing the LogNormal grid and corresponding normalized weight.
  normalized) weight.
  Default value: `quadrature_scheme_lognormal_quantiles`.
* <b>`validate_args`</b>: Python `bool`, default `False`. When `True` distribution
  parameters are checked for validity despite possibly degrading runtime
  performance. When `False` invalid inputs may silently render incorrect
  outputs.
* <b>`allow_nan_stats`</b>: Python `bool`, default `True`. When `True`,
  statistics (e.g., mean, mode, variance) use the value "`NaN`" to
  indicate the result is undefined. When `False`, an exception is raised
  if one or more of the statistic's batch members are undefined.
* <b>`name`</b>: Python `str` name prefixed to Ops created by this class.


#### Raises:


* <b>`TypeError`</b>: if `quadrature_grid` and `quadrature_probs` have different base
  `dtype`.