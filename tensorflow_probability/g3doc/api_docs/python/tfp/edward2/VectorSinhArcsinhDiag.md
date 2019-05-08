<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.edward2.VectorSinhArcsinhDiag" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.edward2.VectorSinhArcsinhDiag

Create a random variable for VectorSinhArcsinhDiag.

``` python
tfp.edward2.VectorSinhArcsinhDiag(
    *args,
    **kwargs
)
```



Defined in [`python/edward2/interceptor.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/edward2/interceptor.py).

<!-- Placeholder for "Used in" -->

See VectorSinhArcsinhDiag for more details.

#### Returns:

  RandomVariable.

#### Original Docstring for Distribution

Construct VectorSinhArcsinhDiag distribution on `R^k`.

The arguments `scale_diag` and `scale_identity_multiplier` combine to
define the diagonal `scale` referred to in this class docstring:

```none
scale = diag(scale_diag + scale_identity_multiplier * ones(k))
```

The `batch_shape` is the broadcast shape between `loc` and `scale`
arguments.

The `event_shape` is given by last dimension of the matrix implied by
`scale`. The last dimension of `loc` (if provided) must broadcast with this

Additional leading dimensions (if any) will index batches.


#### Args:

* <b>`loc`</b>: Floating-point `Tensor`. If this is set to `None`, `loc` is
  implicitly `0`. When specified, may have shape `[B1, ..., Bb, k]` where
  `b >= 0` and `k` is the event size.
* <b>`scale_diag`</b>: Non-zero, floating-point `Tensor` representing a diagonal
  matrix added to `scale`. May have shape `[B1, ..., Bb, k]`, `b >= 0`,
  and characterizes `b`-batches of `k x k` diagonal matrices added to
  `scale`. When both `scale_identity_multiplier` and `scale_diag` are
  `None` then `scale` is the `Identity`.
* <b>`scale_identity_multiplier`</b>: Non-zero, floating-point `Tensor` representing
  a scale-identity-matrix added to `scale`. May have shape
  `[B1, ..., Bb]`, `b >= 0`, and characterizes `b`-batches of scale
  `k x k` identity matrices added to `scale`. When both
  `scale_identity_multiplier` and `scale_diag` are `None` then `scale`
  is the `Identity`.
* <b>`skewness`</b>:  Skewness parameter.  floating-point `Tensor` with shape
  broadcastable with `event_shape`.
* <b>`tailweight`</b>:  Tailweight parameter.  floating-point `Tensor` with shape
  broadcastable with `event_shape`.
* <b>`distribution`</b>: `tf.Distribution`-like instance. Distribution from which `k`
  iid samples are used as input to transformation `F`.  Default is
  `tfd.Normal(loc=0., scale=1.)`.
  Must be a scalar-batch, scalar-event distribution.  Typically
  `distribution.reparameterization_type = FULLY_REPARAMETERIZED` or it is
  a function of non-trainable parameters. WARNING: If you backprop through
  a VectorSinhArcsinhDiag sample and `distribution` is not
  `FULLY_REPARAMETERIZED` yet is a function of trainable variables, then
  the gradient will be incorrect!
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

* <b>`ValueError`</b>: if at most `scale_identity_multiplier` is specified.