# Copyright 2018 The TensorFlow Probability Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Utility function to construct the diagonal of a Jacobian matrix."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import tensorflow as tf

from tensorflow_probability.python.math.gradient import value_and_gradient


__all__ = [
    'diag_jacobian',
]


def diag_jacobian(xs,
                  ys=None,
                  sample_shape=None,
                  fn=None,
                  parallel_iterations=10,
                  name=None):
  """Computes diagonal of the Jacobian matrix of `ys=fn(xs)` wrt `xs`.

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

  Args:
    xs: `Tensor` or a python `list` of `Tensors` of real-like dtypes and shapes
      `sample_shape` + `event_shape_i`, where `event_shape_i` can be different
      for different tensors.
    ys: `Tensor` or a python `list` of `Tensors` of the same dtype as `xs`. Must
        broadcast with the shape of `xs`. Can be omitted if `fn` is provided.
    sample_shape: A common `sample_shape` of the input tensors of `xs`. If not,
      provided, assumed to be `[1]`, which may result in a slow performance of
      `jacobians_diag`.
    fn: Python callable that takes `xs` as an argument (or `*xs`, if it is a
      list) and returns `ys`. Might be skipped if `ys` is provided and
      `tf.enable_eager_execution()` is disabled.
    parallel_iterations: `int` that specifies the allowed number of coordinates
      of the input tensor `xs`, for which the partial derivatives `dys_i/dxs_i`
      can be computed in parallel.
    name: Python `str` name prefixed to `Ops` created by this function.
      Default value: `None` (i.e., "diag_jacobian").

  Returns:
    ys: a list, which coincides with the input `ys`, when provided.
      If the input `ys` is None, `fn(*xs)` gets computed and returned as a list.
    jacobians_diag_res: a `Tensor` or a Python list of `Tensor`s of the same
      dtypes and shapes as the input `xs`. This is the diagonal of the Jacobian
      of ys wrt xs.

  Raises:
    ValueError: if lists `xs` and `ys` have different length or both `ys` and
      `fn` are `None`, or `fn` is None in the eager execution mode.
  """
  with tf.compat.v1.name_scope(name, 'jacobians_diag', [xs, ys]):
    if sample_shape is None:
      sample_shape = [1]
    # Output Jacobian diagonal
    jacobians_diag_res = []
    # Convert input `xs` to a list
    xs = list(xs) if _is_list_like(xs) else [xs]
    xs = [tf.convert_to_tensor(value=x) for x in xs]
    if not tf.executing_eagerly():
      if ys is None:
        if fn is None:
          raise ValueError('Both `ys` and `fn` can not be `None`')
        else:
          ys = fn(*xs)
      # Convert ys to a list
      ys = list(ys) if _is_list_like(ys) else [ys]
      if len(xs) != len(ys):
        raise ValueError('`xs` and `ys` should have the same length')
      for y, x in zip(ys, xs):
        # Broadcast `y` to the shape of `x`.
        y_ = y + tf.zeros_like(x)
        # Change `event_shape` to one-dimension
        y_ = tf.reshape(y, tf.concat([sample_shape, [-1]], -1))

        # Declare an iterator and tensor array loop variables for the gradients.
        n = tf.size(input=x) / tf.cast(
            tf.reduce_prod(input_tensor=sample_shape), dtype=tf.int32)
        n = tf.cast(n, dtype=tf.int32)
        loop_vars = [
            0,
            tf.TensorArray(x.dtype, n)
        ]

        def loop_body(j):
          """Loop function to compute gradients of the each direction."""
          # Gradient along direction `j`.
          res = tf.gradients(ys=y_[..., j], xs=x)[0]  # pylint: disable=cell-var-from-loop
          if res is None:
            # Return zero, if the gradient is `None`.
            res = tf.zeros(tf.concat([sample_shape, [1]], -1),
                           dtype=x.dtype)  # pylint: disable=cell-var-from-loop
          else:
            # Reshape `event_shape` to 1D
            res = tf.reshape(res, tf.concat([sample_shape, [-1]], -1))
            # Add artificial dimension for the case of zero shape input tensor
            res = tf.expand_dims(res, 0)
            res = res[..., j]
          return res  # pylint: disable=cell-var-from-loop

        # Iterate over all elements of the gradient and compute second order
        # derivatives.
        _, jacobian_diag_res = tf.while_loop(
            cond=lambda j, _: j < n,  # pylint: disable=cell-var-from-loop
            body=lambda j, result: (j + 1, result.write(j, loop_body(j))),
            loop_vars=loop_vars,
            parallel_iterations=parallel_iterations)

        shape_x = tf.shape(input=x)
        # Stack gradients together and move flattened `event_shape` to the
        # zero position
        reshaped_jacobian_diag = tf.transpose(a=jacobian_diag_res.stack())
        # Reshape to the original tensor
        reshaped_jacobian_diag = tf.reshape(reshaped_jacobian_diag, shape_x)
        jacobians_diag_res.append(reshaped_jacobian_diag)

    else:
      if fn is None:
        raise ValueError('`fn` can not be `None` when eager execution is '
                         'enabled')
      if ys is None:
        ys = fn(*xs)

      def fn_slice(i, j):
        """Broadcast y[i], flatten event shape of y[i], return y[i][..., j]."""
        def fn_broadcast(*state):
          res = fn(*state)
          res = list(res) if _is_list_like(res) else [res]
          if len(res) != len(state):
            res *= len(state)
          res = [tf.reshape(r + tf.zeros_like(s),
                            tf.concat([sample_shape, [-1]], -1))
                 for r, s in zip(res, state)]
          return res
        # Expand dimensions before returning in order to support 0D input `xs`
        return lambda *state: tf.expand_dims(fn_broadcast(*state)[i], 0)[..., j]

      def make_loop_body(i, x):
        """Loop function to compute gradients of the each direction."""
        def _fn(j, result):
          res = value_and_gradient(fn_slice(i, j), xs)[1][i]
          if res is None:
            res = tf.zeros(tf.concat([sample_shape, [1]], -1), dtype=x.dtype)
          else:
            res = tf.reshape(res, tf.concat([sample_shape, [-1]], -1))
            res = res[..., j]
          return j + 1, result.write(j, res)
        return _fn

      for i, x in enumerate(xs):
        # Declare an iterator and tensor array loop variables for the gradients.
        n = tf.size(input=x) / tf.cast(
            tf.reduce_prod(input_tensor=sample_shape), dtype=tf.int32)
        n = tf.cast(n, dtype=tf.int32)
        loop_vars = [
            0,
            tf.TensorArray(x.dtype, n)
        ]

        # Iterate over all elements of the gradient and compute second order
        # derivatives.
        _, jacobian_diag_res = tf.while_loop(
            cond=lambda j, _: j < n,
            body=make_loop_body(i, x),
            loop_vars=loop_vars,
            parallel_iterations=parallel_iterations)

        shape_x = tf.shape(input=x)
        # Stack gradients together and move flattened `event_shape` to the
        # zero position
        reshaped_jacobian_diag = tf.transpose(a=jacobian_diag_res.stack())
        # Reshape to the original tensor
        reshaped_jacobian_diag = tf.reshape(reshaped_jacobian_diag, shape_x)
        jacobians_diag_res.append(reshaped_jacobian_diag)

  return ys, jacobians_diag_res


def _is_list_like(x):
  """Helper which returns `True` if input is `list`-like."""
  return isinstance(x, (tuple, list))
