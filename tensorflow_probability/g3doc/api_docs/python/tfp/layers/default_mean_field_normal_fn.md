<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.layers.default_mean_field_normal_fn" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.layers.default_mean_field_normal_fn

Creates a function to build Normal distributions with trainable params.

### Aliases:

* `tfp.layers.default_mean_field_normal_fn`
* `tfp.layers.util.default_mean_field_normal_fn`

``` python
tfp.layers.default_mean_field_normal_fn(
    is_singular=False,
    loc_initializer=tf.compat.v1.initializers.random_normal(stddev=0.1),
    untransformed_scale_initializer=tf.compat.v1.initializers.random_normal(mean=-3.0, stddev=0.1),
    loc_regularizer=None,
    untransformed_scale_regularizer=None,
    loc_constraint=None,
    untransformed_scale_constraint=None
)
```



Defined in [`python/layers/util.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/layers/util.py).

<!-- Placeholder for "Used in" -->

This function produces a closure which produces `tfd.Normal`
parameterized by a loc` and `scale` each created using `tf.get_variable`.

#### Args:

* <b>`is_singular`</b>: Python `bool` if `True`, forces the special case limit of
  `scale->0`, i.e., a `Deterministic` distribution.
* <b>`loc_initializer`</b>: Initializer function for the `loc` parameters.
  The default is `tf.random_normal_initializer(mean=0., stddev=0.1)`.
* <b>`untransformed_scale_initializer`</b>: Initializer function for the `scale`
  parameters. Default value: `tf.random_normal_initializer(mean=-3.,
  stddev=0.1)`. This implies the softplus transformed result is initialized
  near `0`. It allows a `Normal` distribution with `scale` parameter set to
  this value to approximately act like a point mass.
* <b>`loc_regularizer`</b>: Regularizer function for the `loc` parameters.
* <b>`untransformed_scale_regularizer`</b>: Regularizer function for the `scale`
  parameters.
* <b>`loc_constraint`</b>: An optional projection function to be applied to the
  loc after being updated by an `Optimizer`. The function must take as input
  the unprojected variable and must return the projected variable (which
  must have the same shape). Constraints are not safe to use when doing
  asynchronous distributed training.
* <b>`untransformed_scale_constraint`</b>: An optional projection function to be
  applied to the `scale` parameters after being updated by an `Optimizer`
  (e.g. used to implement norm constraints or value constraints). The
  function must take as input the unprojected variable and must return the
  projected variable (which must have the same shape). Constraints are not
  safe to use when doing asynchronous distributed training.


#### Returns:

* <b>`make_normal_fn`</b>: Python `callable` which creates a `tfd.Normal`
  using from args: `dtype, shape, name, trainable, add_variable_fn`.