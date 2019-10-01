<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.experimental.substrates.jax.distributions.quadrature_scheme_softmaxnormal_gauss_hermite" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.experimental.substrates.jax.distributions.quadrature_scheme_softmaxnormal_gauss_hermite


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/experimental/substrates/jax/distributions/vector_diffeomixture.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Use Gauss-Hermite quadrature to form quadrature on `K - 1` simplex.

``` python
tfp.experimental.substrates.jax.distributions.quadrature_scheme_softmaxnormal_gauss_hermite(
    normal_loc,
    normal_scale,
    quadrature_size,
    validate_args=False,
    name=None
)
```



<!-- Placeholder for "Used in" -->

A `SoftmaxNormal` random variable `Y` may be generated via

```
Y = SoftmaxCentered(X),
X = Normal(normal_loc, normal_scale)
```

Note: for a given `quadrature_size`, this method is generally less accurate
than `quadrature_scheme_softmaxnormal_quantiles`.

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