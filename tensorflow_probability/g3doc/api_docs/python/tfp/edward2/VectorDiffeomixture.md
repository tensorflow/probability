<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.edward2.VectorDiffeomixture" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.edward2.VectorDiffeomixture

Create a random variable for VectorDiffeomixture.

``` python
tfp.edward2.VectorDiffeomixture(
    *args,
    **kwargs
)
```



Defined in [`python/edward2/interceptor.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/edward2/interceptor.py).

<!-- Placeholder for "Used in" -->

See VectorDiffeomixture for more details.

#### Returns:

  RandomVariable.

#### Original Docstring for Distribution

Constructs the VectorDiffeomixture on `R^d`.

The vector diffeomixture (VDM) approximates the compound distribution

```none
p(x) = int p(x | z) p(z) dz,
where z is in the K-simplex, and
p(x | z) := p(x | loc=sum_k z[k] loc[k], scale=sum_k z[k] scale[k])
```


#### Args:

* <b>`mix_loc`</b>: `float`-like `Tensor` with shape `[b1, ..., bB, K-1]`.
  In terms of samples, larger `mix_loc[..., k]` ==>
  `Z` is more likely to put more weight on its `kth` component.
* <b>`temperature`</b>: `float`-like `Tensor`. Broadcastable with `mix_loc`.
  In terms of samples, smaller `temperature` means one component is more
  likely to dominate.  I.e., smaller `temperature` makes the VDM look more
  like a standard mixture of `K` components.
* <b>`distribution`</b>: <a href="../../tfp/distributions/Distribution.md"><code>tfp.distributions.Distribution</code></a>-like instance. Distribution
  from which `d` iid samples are used as input to the selected affine
  transformation. Must be a scalar-batch, scalar-event distribution.
  Typically `distribution.reparameterization_type = FULLY_REPARAMETERIZED`
  or it is a function of non-trainable parameters. WARNING: If you
  backprop through a VectorDiffeomixture sample and the `distribution`
  is not `FULLY_REPARAMETERIZED` yet is a function of trainable variables,
  then the gradient will be incorrect!
* <b>`loc`</b>: Length-`K` list of `float`-type `Tensor`s. The `k`-th element
  represents the `shift` used for the `k`-th affine transformation.  If
  the `k`-th item is `None`, `loc` is implicitly `0`.  When specified,
  must have shape `[B1, ..., Bb, d]` where `b >= 0` and `d` is the event
  size.
* <b>`scale`</b>: Length-`K` list of `LinearOperator`s. Each should be
  positive-definite and operate on a `d`-dimensional vector space. The
  `k`-th element represents the `scale` used for the `k`-th affine
  transformation. `LinearOperator`s must have shape `[B1, ..., Bb, d, d]`,
  `b >= 0`, i.e., characterizes `b`-batches of `d x d` matrices
* <b>`quadrature_size`</b>: Python `int` scalar representing number of
  quadrature points.  Larger `quadrature_size` means `q_N(x)` better
  approximates `p(x)`.
* <b>`quadrature_fn`</b>: Python callable taking `normal_loc`, `normal_scale`,
  `quadrature_size`, `validate_args` and returning `tuple(grid, probs)`
  representing the SoftmaxNormal grid and corresponding normalized weight.
  normalized) weight.
  Default value: `quadrature_scheme_softmaxnormal_quantiles`.
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

* <b>`ValueError`</b>: if `not scale or len(scale) < 2`.
* <b>`ValueError`</b>: if `len(loc) != len(scale)`
* <b>`ValueError`</b>: if `quadrature_grid_and_probs is not None` and
  `len(quadrature_grid_and_probs[0]) != len(quadrature_grid_and_probs[1])`
* <b>`ValueError`</b>: if `validate_args` and any not scale.is_positive_definite.
* <b>`TypeError`</b>: if any scale.dtype != scale[0].dtype.
* <b>`TypeError`</b>: if any loc.dtype != scale[0].dtype.
* <b>`NotImplementedError`</b>: if `len(scale) != 2`.
* <b>`ValueError`</b>: if `not distribution.is_scalar_batch`.
* <b>`ValueError`</b>: if `not distribution.is_scalar_event`.