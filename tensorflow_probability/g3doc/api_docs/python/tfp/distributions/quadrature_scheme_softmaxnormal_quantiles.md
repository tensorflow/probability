<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.distributions.quadrature_scheme_softmaxnormal_quantiles" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.distributions.quadrature_scheme_softmaxnormal_quantiles

Use SoftmaxNormal quantiles to form quadrature on `K - 1` simplex.

``` python
tfp.distributions.quadrature_scheme_softmaxnormal_quantiles(
    normal_loc,
    normal_scale,
    quadrature_size,
    validate_args=False,
    name=None
)
```



Defined in [`python/distributions/vector_diffeomixture.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/distributions/vector_diffeomixture.py).

<!-- Placeholder for "Used in" -->

A `SoftmaxNormal` random variable `Y` may be generated via

```
Y = SoftmaxCentered(X),
X = Normal(normal_loc, normal_scale)
```

#### Args:

* <b>`normal_loc`</b>: `float`-like `Tensor` with shape `[b1, ..., bB, K-1]`, B>=0.
  The location parameter of the Normal used to construct the SoftmaxNormal.
* <b>`normal_scale`</b>: `float`-like `Tensor`. Broadcastable with `normal_loc`.
  The scale parameter of the Normal used to construct the SoftmaxNormal.
* <b>`quadrature_size`</b>: Python `int` scalar representing the number of quadrature
  points.
* <b>`validate_args`</b>: Python `bool`, default `False`. When `True` distribution
  parameters are checked for validity despite possibly degrading runtime
  performance. When `False` invalid inputs may silently render incorrect
  outputs.
* <b>`name`</b>: Python `str` name prefixed to Ops created by this class.


#### Returns:

* <b>`grid`</b>: Shape `[b1, ..., bB, K, quadrature_size]` `Tensor` representing the
  convex combination of affine parameters for `K` components.
  `grid[..., :, n]` is the `n`-th grid point, living in the `K - 1` simplex.
* <b>`probs`</b>:  Shape `[b1, ..., bB, K, quadrature_size]` `Tensor` representing the
  associated with each grid point.