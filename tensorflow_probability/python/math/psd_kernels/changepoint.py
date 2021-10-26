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
"""Change point kernel."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.math.psd_kernels import positive_semidefinite_kernel as psd_kernel
from tensorflow_probability.python.math.psd_kernels.internal import util

__all__ = ['ChangePoint']


class ChangePoint(psd_kernel.AutoCompositeTensorPsdKernel):
  """Changepoint Kernel.


  Given a list of kernels `k_1`, `k_2`, ..., `k_n`, and 1-D inputs `x` and `y`,
  this kernel computes a smooth interpolant between the kernels.

  ```none
  k(x, y) = (1 - s_1(x)) * k_1(x, y) * (1 - s1(y)) +
            (1 - s_2(x)) * s_1(x) * k_2(x, y) * (1 - s_2(y)) * s_1(y) +
            (1 - s_3(x)) * s_2(x) * k_3(x, y) * (1 - s_3(y)) * s_2(y) +
            ...
            s_{n-1}(x) * kn(x, y) * s_{n-1}(y)
  ```

  where:
    * `s_i(x) = sigmoid(slopes[i] * (x - locs[i]))`
    * `locs` is a `Tensor` of length `n - 1` that's in ascending order.
    * `slopes` is a positive `Tensor` of length `n - 1`.

  If we have 2 kernels `k1` and `k2`, this takes the form:

  ```none
  k(x, y) = (1 - s_1(x)) * k_1(x, y) * (1 - s_1(y)) +
            s_1(x) * k_2(x, y) * s_1(y)
  ```

  When `x` and `y` are much less than `locs[0]`, `k(x, y) ~= k1(x, y)`, while
  `x` and `y` are much greater than `locs[0]`, `k(x, y) ~= k2(x, y)`.

  In general, this kernel performs a smooth interpolation between `ki(x, y)`
  where `k(x, y) ~= ki(x, y)` when `locs[i - 1] < x, y < locs[i]`.

  This kernel accepts an optional `weight_fn` which consumes `x` and returns
  a scalar. This is used when computing the sigmoids
  `si(x) = sigmoid(slopes[i] * (w(x) - locs[i]))`, which allows this kernel
  to be computed on arbitrary dimensional input. For instance, a `weight_fn`
  that is `tf.linalg.norm` would smoothly interpolate between different kernels
  over different annuli in the plane.

  #### References
  [1]: Andrew Gordon Wilson. The Change Point Kernel.
       https://www.cs.cmu.edu/~andrewgw/changepoints.pdf
  """

  def __init__(
      self,
      kernels,
      locs,
      slopes,
      weight_fn=util.sum_rightmost_ndims_preserving_shape,
      validate_args=False,
      name='ChangePoint'):
    """Construct a ChangePoint kernel instance.

    Args:
      kernels: List of size `[N]` of `PositiveSemidefiniteKernel` instances to
        interpolate between.
      locs: Ascending Floating-point `Tensor` of shape broadcastable to
        `[..., N - 1]` that controls the regions for the interpolation.
        If `kernels` are a list of 1-D kernels with the default `weight_fn`,
        then between `locs[i - 1]` and `locs[i]`, this kernel acts like
        `kernels[i]`.
      slopes: Positive Floating-point `Tensor` of shape broadcastable to
        `[..., N - 1]` that controls how smooth the interpolation between
        kernels is (larger `slopes` means more discrete transitions).
      weight_fn: Python `callable` which takes an input `x` and `feature_ndims`
        argument, and returns a `Tensor` where a scalar is returned for
        each right-most `feature_ndims` of the input.
        (in other words, if `x` is a batch of inputs, `weight_fn` returns
        a batch of scalar, with the same batch shape).
        Default value: Sums over the last `feature_ndims` of the input `x`.
      validate_args: If `True`, parameters are checked for validity despite
        possibly degrading runtime performance
      name: Python `str` name prefixed to Ops created by this class.
    """

    parameters = dict(locals())
    with tf.name_scope(name):
      dtype = util.maybe_get_common_dtype(
          [kernels, locs, slopes])

      if not callable(weight_fn):
        raise TypeError(f'fn is not callable: {weight_fn}')
      if len(kernels) < 2:
        raise ValueError(
            f'Expecting at least 2 kernels, got {len(kernels)}')
      if not all(k.feature_ndims == kernels[0].feature_ndims for k in kernels):
        raise ValueError(
            'Expect that all `kernels` have the same feature_ndims')
      self._kernels = kernels
      self._locs = tensor_util.convert_nonref_to_tensor(
          locs, name='locs', dtype=dtype)
      self._slopes = tensor_util.convert_nonref_to_tensor(
          slopes, name='slopes', dtype=dtype)
      self._weight_fn = weight_fn
      super(ChangePoint, self).__init__(
          kernels[0].feature_ndims,
          dtype=dtype,
          name=name,
          validate_args=validate_args,
          parameters=parameters)

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
    from tensorflow_probability.python.bijectors import ascending  # pylint:disable=g-import-not-at-top
    from tensorflow_probability.python.bijectors import softplus  # pylint:disable=g-import-not-at-top
    return dict(
        kernels=parameter_properties.BatchedComponentProperties(
            event_ndims=lambda self: [0 for _ in self.kernels]),
        locs=parameter_properties.ParameterProperties(
            event_ndims=1,
            default_constraining_bijector_fn=lambda: ascending.Ascending()),  # pylint:disable=unnecessary-lambda
        slopes=parameter_properties.ParameterProperties(
            event_ndims=1,
            default_constraining_bijector_fn=(
                lambda: softplus.Softplus(low=dtype_util.eps(dtype)))))

  @property
  def kernels(self):
    return self._kernels

  @property
  def locs(self):
    return self._locs

  @property
  def slopes(self):
    return self._slopes

  @property
  def weight_fn(self):
    return self._weight_fn

  def _apply(self, x1, x2, example_ndims):
    locs = util.pad_shape_with_ones(self.locs, example_ndims, start=-2)
    slopes = util.pad_shape_with_ones(self.slopes, example_ndims, start=-2)

    weights_x1 = tf.math.sigmoid(
        slopes * (
            self.weight_fn(x1, self.feature_ndims)[..., tf.newaxis] - locs))
    weights_x2 = tf.math.sigmoid(
        slopes * (
            self.weight_fn(x2, self.feature_ndims)[..., tf.newaxis] - locs))

    initial_weights = (1. - weights_x1) * (1. - weights_x2)
    initial_weights = tf.concat([
        initial_weights,
        tf.ones_like(initial_weights[..., 0])[..., tf.newaxis]], axis=-1)
    end_weights = weights_x1 * weights_x2
    end_weights = tf.concat([
        tf.ones_like(end_weights[..., 0])[..., tf.newaxis],
        end_weights], axis=-1)

    results = [k.apply(x1, x2, example_ndims)[
        ..., tf.newaxis] for k in self.kernels]
    broadcasted_shape = distribution_util.get_broadcast_shape(*results)
    results = tf.concat(
        [ps.broadcast_to(r, broadcasted_shape) for r in results], axis=-1)
    return tf.math.reduce_sum(initial_weights * results * end_weights, axis=-1)

  def _matrix(self, x1, x2):
    locs = util.pad_shape_with_ones(self.locs, ndims=1, start=-2)
    slopes = util.pad_shape_with_ones(self.slopes, ndims=1, start=-2)

    weights_x1 = tf.math.sigmoid(
        slopes * (
            self.weight_fn(x1, self.feature_ndims)[..., tf.newaxis] - locs))
    weights_x1 = weights_x1[..., tf.newaxis, :]
    weights_x2 = tf.math.sigmoid(
        slopes * (
            self.weight_fn(x2, self.feature_ndims)[..., tf.newaxis] - locs))
    weights_x2 = weights_x2[..., tf.newaxis, :, :]

    initial_weights = (1. - weights_x1) * (1. - weights_x2)
    initial_weights = tf.concat([
        initial_weights,
        tf.ones_like(initial_weights[..., 0])[..., tf.newaxis]], axis=-1)
    end_weights = weights_x1 * weights_x2
    end_weights = tf.concat([
        tf.ones_like(end_weights[..., 0])[..., tf.newaxis],
        end_weights], axis=-1)

    results = [k.matrix(x1, x2)[..., tf.newaxis] for k in self.kernels]
    broadcasted_shape = distribution_util.get_broadcast_shape(*results)
    results = tf.concat(
        [ps.broadcast_to(r, broadcasted_shape) for r in results], axis=-1)
    return tf.math.reduce_sum(initial_weights * results * end_weights, axis=-1)

  def _parameter_control_dependencies(self, is_init):
    if is_init:
      # Check that locs and slopes have the same last dimension.
      if (self.locs.shape is not None and
          self.locs.shape[-1] is not None and
          self.slopes.shape is not None and
          self.slopes.shape[-1] is not None):
        k = self.locs.shape[-1]
        l = self.slopes.shape[-1]
        if not(k == l or k == 1 or l == 1):
          raise ValueError(
              'Expect that `locs` and `slopes` are broadcastable.')

      if self.locs.shape is not None and self.locs.shape[-1] is not None:
        if not (len(self.kernels) == self.locs.shape[-1] + 1 or
                self.locs.shape[-1] == 1):
          raise ValueError(
              'Expect that `locs` has last dimension `1` or `N - 1` where '
              f'`N` is the number of kernels, but got {self.locs.shape[-1]}')

      if self.slopes.shape is not None and self.slopes.shape[-1] is not None:
        if not (len(self.kernels) == self.slopes.shape[-1] + 1 or
                self.slopes.shape[-1] == 1):
          raise ValueError(
              'Expect that `slopes` has last dimension `1` or `N - 1` where '
              f'`N` is the number of kernels, but got {self.slopes.shape[-1]}')

    assertions = []
    if not self.validate_args:
      return assertions

    if is_init != tensor_util.is_ref(self.locs):
      locs = tf.convert_to_tensor(self.locs)

      assertions.append(
          assert_util.assert_greater(
              locs[..., 1:], locs[..., :-1],
              message='Expect that elements of `locs` are ascending.'))

    if is_init != tensor_util.is_ref(self.slopes):
      slopes = tf.convert_to_tensor(self.slopes)
      assertions.append(assert_util.assert_positive(
          slopes, message='`slopes` must be positive.'))

    return assertions
