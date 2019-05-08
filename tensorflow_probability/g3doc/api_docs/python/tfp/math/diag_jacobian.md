<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.math.diag_jacobian" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.math.diag_jacobian

Computes diagonal of the Jacobian matrix of `ys=fn(xs)` wrt `xs`.

``` python
tfp.math.diag_jacobian(
    xs,
    ys=None,
    sample_shape=None,
    fn=None,
    parallel_iterations=10,
    name=None
)
```



Defined in [`python/math/diag_jacobian.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/math/diag_jacobian.py).

<!-- Placeholder for "Used in" -->

  If `ys` is a tensor or a list of tensors of the form `(ys_1, .., ys_n)` and
  `xs` is of the form `(xs_1, .., xs_n)`, the function `jacobians_diag`
  computes the diagonal of the Jacobian matrix, i.e., the partial derivatives
  `(dys_1/dxs_1,.., dys_n/dxs_n`). For definition details, see
  https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant
#### Example

##### Diagonal Hessian of the log-density of a 3D Gaussian distribution

In this example we sample from a standard univariate normal
distribution using MALA with `step_size` equal to 0.75.

```python
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

tfd = tfp.distributions

dtype = np.float32
with tf.Session(graph=tf.Graph()) as sess:
  true_mean = dtype([0, 0, 0])
  true_cov = dtype([[1, 0.25, 0.25], [0.25, 2, 0.25], [0.25, 0.25, 3]])
  chol = tf.linalg.cholesky(true_cov)
  target = tfd.MultivariateNormalTriL(loc=true_mean, scale_tril=chol)

  # Assume that the state is passed as a list of tensors `x` and `y`.
  # Then the target function is defined as follows:
  def target_fn(x, y):
    # Stack the input tensors together
    z = tf.concat([x, y], axis=-1) - true_mean
    return target.log_prob(z)

  sample_shape = [3, 5]
  state = [tf.ones(sample_shape + [2], dtype=dtype),
           tf.ones(sample_shape + [1], dtype=dtype)]
  fn_val, grads = tfp.math.value_and_gradient(target_fn, state)

  # We can either pass the `sample_shape` of the `state` or not, which impacts
  # computational speed of `diag_jacobian`
  _, diag_jacobian_shape_passed = diag_jacobian(
      xs=state, ys=grads, sample_shape=tf.shape(fn_val))
  _, diag_jacobian_shape_none = diag_jacobian(
      xs=state, ys=grads)

  diag_jacobian_shape_passed_ = sess.run(diag_jacobian_shape_passed)
  diag_jacobian_shape_none_ = sess.run(diag_jacobian_shape_none)

print('hessian computed through `diag_jacobian`, sample_shape passed: ',
      np.concatenate(diag_jacobian_shape_passed_, -1))
print('hessian computed through `diag_jacobian`, sample_shape skipped',
      np.concatenate(diag_jacobian_shape_none_, -1))

```

#### Args:

* <b>`xs`</b>: `Tensor` or a python `list` of `Tensors` of real-like dtypes and shapes
  `sample_shape` + `event_shape_i`, where `event_shape_i` can be different
  for different tensors.
* <b>`ys`</b>: `Tensor` or a python `list` of `Tensors` of the same dtype as `xs`. Must
    broadcast with the shape of `xs`. Can be omitted if `fn` is provided.
* <b>`sample_shape`</b>: A common `sample_shape` of the input tensors of `xs`. If not,
  provided, assumed to be `[1]`, which may result in a slow performance of
  `jacobians_diag`.
* <b>`fn`</b>: Python callable that takes `xs` as an argument (or `*xs`, if it is a
  list) and returns `ys`. Might be skipped if `ys` is provided and
  `tf.enable_eager_execution()` is disabled.
* <b>`parallel_iterations`</b>: `int` that specifies the allowed number of coordinates
  of the input tensor `xs`, for which the partial derivatives `dys_i/dxs_i`
  can be computed in parallel.
* <b>`name`</b>: Python `str` name prefixed to `Ops` created by this function.
  Default value: `None` (i.e., "diag_jacobian").


#### Returns:

* <b>`ys`</b>: a list, which coincides with the input `ys`, when provided.
  If the input `ys` is None, `fn(*xs)` gets computed and returned as a list.
* <b>`jacobians_diag_res`</b>: a `Tensor` or a Python list of `Tensor`s of the same
  dtypes and shapes as the input `xs`. This is the diagonal of the Jacobian
  of ys wrt xs.


#### Raises:

* <b>`ValueError`</b>: if lists `xs` and `ys` have different length or both `ys` and
  `fn` are `None`, or `fn` is None in the eager execution mode.