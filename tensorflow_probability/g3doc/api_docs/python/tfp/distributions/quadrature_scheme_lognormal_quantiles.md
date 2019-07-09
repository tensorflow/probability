<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.distributions.quadrature_scheme_lognormal_quantiles" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.distributions.quadrature_scheme_lognormal_quantiles

Use LogNormal quantiles to form quadrature on positive-reals.

``` python
tfp.distributions.quadrature_scheme_lognormal_quantiles(
    loc,
    scale,
    quadrature_size,
    validate_args=False,
    name=None
)
```



Defined in [`python/distributions/poisson_lognormal.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/distributions/poisson_lognormal.py).

<!-- Placeholder for "Used in" -->


#### Args:


* <b>`loc`</b>: `float`-like (batch of) scalar `Tensor`; the location parameter of
  the LogNormal prior.
* <b>`scale`</b>: `float`-like (batch of) scalar `Tensor`; the scale parameter of
  the LogNormal prior.
* <b>`quadrature_size`</b>: Python `int` scalar representing the number of quadrature
  points.
* <b>`validate_args`</b>: Python `bool`, default `False`. When `True` distribution
  parameters are checked for validity despite possibly degrading runtime
  performance. When `False` invalid inputs may silently render incorrect
  outputs.
* <b>`name`</b>: Python `str` name prefixed to Ops created by this class.


#### Returns:


* <b>`grid`</b>: (Batch of) length-`quadrature_size` vectors representing the
  `log_rate` parameters of a `Poisson`.
* <b>`probs`</b>: (Batch of) length-`quadrature_size` vectors representing the
  weight associate with each `grid` value.