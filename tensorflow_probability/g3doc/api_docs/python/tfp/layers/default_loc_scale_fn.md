<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.layers.default_loc_scale_fn" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.layers.default_loc_scale_fn

Makes closure which creates `loc`, `scale` params from `tf.get_variable`.

### Aliases:

* `tfp.layers.default_loc_scale_fn`
* `tfp.layers.util.default_loc_scale_fn`

``` python
tfp.layers.default_loc_scale_fn(
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

This function produces a closure which produces `loc`, `scale` using
`tf.get_variable`. The closure accepts the following arguments:

  dtype: Type of parameter's event.
  shape: Python `list`-like representing the parameter's event shape.
  name: Python `str` name prepended to any created (or existing)
    `tf.Variable`s.
  trainable: Python `bool` indicating all created `tf.Variable`s should be
    added to the graph collection `GraphKeys.TRAINABLE_VARIABLES`.
  add_variable_fn: `tf.get_variable`-like `callable` used to create (or
    access existing) `tf.Variable`s.

#### Args:


* <b>`is_singular`</b>: Python `bool` indicating if `scale is None`. Default: `False`.
* <b>`loc_initializer`</b>: Initializer function for the `loc` parameters.
  The default is `tf.random_normal_initializer(mean=0., stddev=0.1)`.
* <b>`untransformed_scale_initializer`</b>: Initializer function for the `scale`
  parameters. Default value: `tf.random_normal_initializer(mean=-3.,
  stddev=0.1)`. This implies the softplus transformed result is initialized
  near `0`. It allows a `Normal` distribution with `scale` parameter set to
  this value to approximately act like a point mass.
* <b>`loc_regularizer`</b>: Regularizer function for the `loc` parameters.
  The default (`None`) is to use the `tf.get_variable` default.
* <b>`untransformed_scale_regularizer`</b>: Regularizer function for the `scale`
  parameters. The default (`None`) is to use the `tf.get_variable` default.
* <b>`loc_constraint`</b>: An optional projection function to be applied to the
  loc after being updated by an `Optimizer`. The function must take as input
  the unprojected variable and must return the projected variable (which
  must have the same shape). Constraints are not safe to use when doing
  asynchronous distributed training.
  The default (`None`) is to use the `tf.get_variable` default.
* <b>`untransformed_scale_constraint`</b>: An optional projection function to be
  applied to the `scale` parameters after being updated by an `Optimizer`
  (e.g. used to implement norm constraints or value constraints). The
  function must take as input the unprojected variable and must return the
  projected variable (which must have the same shape). Constraints are not
  safe to use when doing asynchronous distributed training. The default
  (`None`) is to use the `tf.get_variable` default.


#### Returns:


* <b>`default_loc_scale_fn`</b>: Python `callable` which instantiates `loc`, `scale`
parameters from args: `dtype, shape, name, trainable, add_variable_fn`.