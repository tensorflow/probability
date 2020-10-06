# Copyright 2020 The TensorFlow Probability Authors.
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
"""ExpSinSquared kernel."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.math.psd_kernels.internal import util
from tensorflow_probability.python.math.psd_kernels.positive_semidefinite_kernel import PositiveSemidefiniteKernel

__all__ = ['Parabolic']


class Parabolic(PositiveSemidefiniteKernel):
  """The Parabolic kernel.

  ```none
  k(x, y) = 3 / (4 * sqrt(5)) * amplitude *
            max(0, 1 - (||x_k - y_k|| / (length_scale * sqrt(5)))**2)
  ```

  where the double-bars represent vector length (ie, Euclidean, or L2 norm).

  When `amplitude = 1` and `length_scale = 1`, this is the Epanechnikov kernel,
  which is often used for density estimation because of its optimality according
  to a notion of efficiency as
  `efficiency = sqrt(integral(u**2 k(u) du)) integral(k(u)**2 du)`. This
  optimality was first derived in a different context [1], and suggested for use
  in KDE by Epanechnikov in [2]. This is nicely summarized in [3], adjacent to
  Fig 3.1.

  #### References

  [1] Hodges, Joseph L., and Erich L. Lehmann. "The efficiency of some
      nonparametric competitors of the $ t $-test." The Annals of Mathematical
      Statistics 27.2 (1956): 324-335.
  [2] Epanechnikov, Vassiliy A. "Non-parametric estimation of a multivariate
      probability density." Theory of Probability & Its Applications 14.1
      (1969): 153-158.
  [3] Silverman, Bernard W. Density estimation for statistics and data analysis.
      Vol. 26. CRC press, 1986.
  """

  def __init__(self,
               amplitude=None,
               length_scale=None,
               feature_ndims=1,
               validate_args=False,
               name='Parabolic'):
    """Construct a Parabolic kernel instance.

    Args:
      amplitude: Positive floating point `Tensor` that controls the maximum
        value of the kernel. Must be broadcastable with `period`, `length_scale`
        and inputs to `apply` and `matrix` methods. A value of `None` is treated
        like 1.
      length_scale: Positive floating point `Tensor` that controls how sharp or
        wide the kernel shape is. This provides a characteristic "unit" of
        length against which `|x - y|` can be compared for scale. Must be
        broadcastable with `amplitude`, `period`  and inputs to `apply` and
        `matrix` methods. A value of `None` is treated like 1.
      feature_ndims: Python `int` number of rightmost dims to include in kernel
        computation.
      validate_args: If `True`, parameters are checked for validity despite
        possibly degrading runtime performance
      name: Python `str` name prefixed to Ops created by this class.
    """
    parameters = dict(locals())
    with tf.name_scope(name):
      dtype = util.maybe_get_common_dtype([amplitude, length_scale])
      self._amplitude = tensor_util.convert_nonref_to_tensor(
          amplitude, name='amplitude', dtype=dtype)
      self._length_scale = tensor_util.convert_nonref_to_tensor(
          length_scale, name='length_scale', dtype=dtype)
      super(Parabolic, self).__init__(
          feature_ndims,
          dtype=dtype,
          name=name,
          validate_args=validate_args,
          parameters=parameters)

  @property
  def amplitude(self):
    """Amplitude parameter."""
    return self._amplitude

  @property
  def length_scale(self):
    """Length scale parameter."""
    return self._length_scale

  def _batch_shape(self):
    scalar_shape = tf.TensorShape([])
    return tf.broadcast_static_shape(
        scalar_shape if self.amplitude is None else self.amplitude.shape,
        scalar_shape if self.length_scale is None else self.length_scale.shape)

  def _batch_shape_tensor(self):
    return tf.broadcast_dynamic_shape(
        [] if self.amplitude is None else ps.shape(self.amplitude),
        [] if self.length_scale is None else ps.shape(self.length_scale))

  def _apply_with_distance(
      self, x1, x2, pairwise_square_distance, example_ndims=0):
    default_bandwidth_sq = 5.
    pairwise_square_distance = pairwise_square_distance / default_bandwidth_sq
    if self.length_scale is not None:
      length_scale = tf.convert_to_tensor(self.length_scale)
      length_scale = util.pad_shape_with_ones(
          length_scale, example_ndims)
      pairwise_square_distance = pairwise_square_distance / length_scale**2

    default_scale = tf.cast(.75 / np.sqrt(5.), pairwise_square_distance.dtype)
    result = tf.nn.relu(1 - pairwise_square_distance) * default_scale

    if self.amplitude is not None:
      amplitude = tf.convert_to_tensor(self.amplitude)
      amplitude = util.pad_shape_with_ones(amplitude, example_ndims)
      result = result * amplitude

    return result

  def _apply(self, x1, x2, example_ndims=0):
    pairwise_square_distance = util.sum_rightmost_ndims_preserving_shape(
        tf.math.squared_difference(x1, x2), self.feature_ndims)
    return self._apply_with_distance(
        x1, x2, pairwise_square_distance, example_ndims=example_ndims)

  def _matrix(self, x1, x2):
    pairwise_square_distance = util.pairwise_square_distance_matrix(
        x1, x2, self.feature_ndims)
    return self._apply_with_distance(
        x1, x2, pairwise_square_distance, example_ndims=2)

  def _tensor(self, x1, x2, x1_example_ndims, x2_example_ndims):
    pairwise_square_distance = util.pairwise_square_distance_tensor(
        x1, x2, self.feature_ndims, x1_example_ndims, x2_example_ndims)
    return self._apply_with_distance(
        x1, x2, pairwise_square_distance,
        example_ndims=(x1_example_ndims + x2_example_ndims))

  def _parameter_control_dependencies(self, is_init):
    if not self.validate_args:
      return []
    assertions = []
    for arg_name, arg in dict(amplitude=self.amplitude,
                              length_scale=self.length_scale).items():
      if arg is not None and is_init != tensor_util.is_ref(arg):
        assertions.append(assert_util.assert_positive(
            arg,
            message='{} must be positive.'.format(arg_name)))
    return assertions

