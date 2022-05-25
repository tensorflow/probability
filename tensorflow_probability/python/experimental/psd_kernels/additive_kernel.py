# Copyright 2021 The TensorFlow Probability Authors.
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
"""Additive kernel."""

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.math.psd_kernels import positive_semidefinite_kernel as psd_kernel
from tensorflow_probability.python.math.psd_kernels.internal import util


__all__ = [
    'AdditiveKernel',
]


class AdditiveKernel(psd_kernel.AutoCompositeTensorPsdKernel):
  """Additive Kernel.

  This kernel has the following form
  ```none
  k(x, y) = sum k_add_i(x, y)
  k_add_n(x, y) = a_n**2 sum_{1<=i1<i2<...in} prod k_i(x[i], y[i])
  ```
  Where $k_i$ is the one-dimensional base kernel for the `i`th dimension.

  In other words, this computes sums of elementary symmetric polynomials
  over `k_i(x[i], y[i])`.

  This kernel is very related to the ANOVA kernel defined as:
  `k_{ANOVA}(x, y) = prod (1 + k_i(x[i], x[i])`. `k_{ANOVA}` is
  equivalent to a special case of this kernel where the `amplitudes` are
  all one, along with a constant shift by 1.

  #### References

  [1] D. Duvenaud, H. Nickish, C. E. Rasmussen, Additive Gaussian Process.
  https://hannes.nickisch.org/papers/conferences/duvenaud11gpadditive.pdf

  [2] M. Stitson, A. Gammerman, V. Vapnik, V. Vovk, et al.
      Support Vector Regression with ANOVA Decomposition Kernels
      http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.34.7818&rep=rep1&type=pdf
  """

  def __init__(
      self,
      kernel,
      amplitudes,
      validate_args=False,
      name='AdditiveKernel'):
    """Instantiates an `AdditiveKernel`.

    Args:
      kernel: An instance of `PositiveSemidefiniteKernel`s that are defined
        within this class (specifically they allow for reinterpreting
        batch dimensions as feature dimensions) that act on inputs of
        the form `[B1, ...., Bk, D, 1]`; that is, `kernel` is a batch of
        D-kernels, each acting on 1-dimensional inputs. We assume that the
        kernel has a batch dimension broadcastable with `[D]`. `kernel` must
        inherit from `tf.__internal__.CompositeTensor`.
      amplitudes: `Tensor` of shape `[B1, ...., Bk, M]`, where `M` is the order
        of the additive kernel. `M` must be statically identifiable.
      validate_args: Python `bool`, default `False`. When `True` kernel
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      name: Python `str` name prefixed to Ops created by this class. Default:
        subclass name.
    Raises:
      TypeError: if `kernel` is not an instance of
        `tf.__internal__.CompositeTensor`.
    """
    parameters = dict(locals())
    with tf.name_scope(name):
      if not isinstance(kernel, tf.__internal__.CompositeTensor):
        raise TypeError('`kernel` must inherit from '
                        '`tf.__internal__.CompositeTensor`.')
      dtype = util.maybe_get_common_dtype([kernel, amplitudes])
      self._kernel = kernel
      self._amplitudes = tensor_util.convert_nonref_to_tensor(
          amplitudes, name='amplitudes', dtype=dtype)
      super(AdditiveKernel, self).__init__(
          feature_ndims=self.kernel.feature_ndims,
          dtype=dtype,
          name=name,
          validate_args=validate_args,
          parameters=parameters)

  @property
  def amplitudes(self):
    """Amplitude parameter for each additive kernel."""
    return self._amplitudes

  @property
  def kernel(self):
    """Inner kernel used for scalar kernel computations."""
    return self._kernel

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
    from tensorflow_probability.python.bijectors import softplus as softplus_bijector  # pylint:disable=g-import-not-at-top
    return dict(
        amplitudes=parameter_properties.ParameterProperties(
            event_ndims=1,
            default_constraining_bijector_fn=(
                softplus_bijector.Softplus(low=dtype_util.eps(dtype)))),
        kernel=parameter_properties.BatchedComponentProperties(event_ndims=1))

  # Below, the Additive Kernel is computed via a recurrence on elementary
  # symmetric polynomials.
  # Let z_i = k[i](x[i], y[i])
  # Then we are computing the elementary symmetric polynomials
  # S_n(z_1, ..., z_k) = \sum_i \prod_{1 <= j_1 < j_2, ... < j_n <= k} z_j
  # Elementary symmetric polynomials satisfy the recurrence:
  # S_n(z_1, ..., z_k) = S_n(z_1, ..., z_{k-1}) +
  #                      z_k * S_{n - 1}(z_1, ..., z_{k - 1})
  # Thus, we can use dynamic programming to compute the elementary symmetric
  # polynomials over z_i, and use vectorization to do this in a batched way.

  def _apply(self, x1, x2, example_ndims=0):
    @tf.recompute_grad
    def _inner_apply(x1, x2):
      order = ps.shape(self.amplitudes)[-1]

      def scan_fn(esp, i):
        s = self.kernel[..., i].apply(
            x1[..., i][..., tf.newaxis],
            x2[..., i][..., tf.newaxis],
            example_ndims=example_ndims)
        next_esp = esp[..., 1:] + s[..., tf.newaxis] * esp[..., :-1]
        # Add the zero-th polynomial.
        next_esp = tf.concat(
            [tf.ones_like(esp[..., 0][..., tf.newaxis]), next_esp], axis=-1)
        return next_esp

      batch_shape = ps.broadcast_shape(
          ps.shape(x1)[:-self.kernel.feature_ndims],
          ps.shape(x2)[:-self.kernel.feature_ndims])

      batch_shape = ps.broadcast_shape(
          batch_shape,
          ps.concat([
              self.batch_shape_tensor(),
              [1] * example_ndims], axis=0))

      initializer = tf.concat(
          [tf.ones(ps.concat([batch_shape, [1]], axis=0),
                   dtype=self.dtype),
           tf.zeros(ps.concat([batch_shape, [order]], axis=0),
                    dtype=self.dtype)], axis=-1)

      esps = tf.scan(
          scan_fn,
          elems=ps.range(0, ps.shape(x1)[-1], dtype=tf.int32),
          parallel_iterations=32,
          initializer=initializer)[-1, ..., 1:]
      amplitudes = util.pad_shape_with_ones(
          self.amplitudes, ndims=example_ndims, start=-2)
      return tf.reduce_sum(esps * tf.math.square(amplitudes), axis=-1)
    return _inner_apply(x1, x2)

  def _matrix(self, x1, x2):
    @tf.recompute_grad
    def _inner_matrix(x1, x2):
      order = ps.shape(self.amplitudes)[-1]

      def scan_fn(esp, i):
        s = self.kernel[..., i].matrix(
            x1[..., i][..., tf.newaxis], x2[..., i][..., tf.newaxis])
        next_esp = esp[..., 1:] + s[..., tf.newaxis] * esp[..., :-1]
        # Add the zero-th polynomial.
        next_esp = tf.concat(
            [tf.ones_like(esp[..., 0][..., tf.newaxis]), next_esp], axis=-1)
        return next_esp

      batch_shape = ps.broadcast_shape(
          ps.shape(x1)[:-(self.kernel.feature_ndims + 1)],
          ps.shape(x2)[:-(self.kernel.feature_ndims + 1)])
      batch_shape = ps.broadcast_shape(
          batch_shape, self.batch_shape_tensor())
      matrix_shape = [
          ps.shape(x1)[-(self.kernel.feature_ndims + 1)],
          ps.shape(x2)[-(self.kernel.feature_ndims + 1)]]
      total_shape = ps.concat([batch_shape, matrix_shape], axis=0)

      initializer = tf.concat(
          [tf.ones(ps.concat([total_shape, [1]], axis=0),
                   dtype=self.dtype),
           tf.zeros(ps.concat([total_shape, [order]], axis=0),
                    dtype=self.dtype)], axis=-1)

      esps = tf.scan(
          scan_fn,
          elems=ps.range(0, ps.shape(x1)[-1], dtype=tf.int32),
          parallel_iterations=32,
          initializer=initializer)[-1, ..., 1:]
      amplitudes = util.pad_shape_with_ones(
          self.amplitudes, ndims=2, start=-2)
      return tf.reduce_sum(esps * tf.math.square(amplitudes), axis=-1)
    return _inner_matrix(x1, x2)
