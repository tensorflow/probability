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
"""Change point Tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.math.psd_kernels import changepoint
from tensorflow_probability.python.math.psd_kernels import exponentiated_quadratic
from tensorflow_probability.python.math.psd_kernels import matern


@test_util.test_all_tf_execution_regimes
class ChangePointTest(test_util.TestCase):

  def testShape(self):
    k1 = exponentiated_quadratic.ExponentiatedQuadratic(
        amplitude=np.ones([3]), length_scale=np.ones([2, 1]))
    k2 = matern.MaternFiveHalves(
        amplitude=np.ones([5, 1, 1]), length_scale=np.ones([2, 1, 1, 1]))
    k3 = exponentiated_quadratic.ExponentiatedQuadratic(
        amplitude=np.ones([3, 1, 1, 1, 1]),
        length_scale=np.ones([2, 1, 1, 1, 1, 1]))
    locs = np.ones([2, 1, 2])
    slopes = np.ones([1, 3, 2])
    weight_fn = lambda x, f: tf.linalg.norm(x, axis=-1)
    k = changepoint.ChangePoint([k1, k2, k3],
                                locs=locs,
                                slopes=slopes,
                                weight_fn=weight_fn)

    batch_shape = [2, 3, 2, 5, 2, 3]

    self.assertIs(weight_fn, k.weight_fn)
    self.assertAllEqual(batch_shape, k.batch_shape)
    self.assertAllEqual(
        batch_shape, self.evaluate(k.batch_shape_tensor()))

    x = np.ones([7, 3])
    y = np.ones([11, 3])

    self.assertAllEqual(batch_shape + [7, 11], k.matrix(x, y).shape)
    self.assertAllEqual(
        batch_shape + [7, 11],
        k.matrix(tf.stack([x]*3), tf.stack([y]*3)).shape)

  def testSingleChangePoint(self):
    amplitude = 5.
    length_scale = .2

    k1 = exponentiated_quadratic.ExponentiatedQuadratic(amplitude, length_scale)
    k2 = matern.MaternThreeHalves(amplitude, length_scale)

    x = np.random.uniform(-1, 1, size=[30, 1]).astype(np.float32)
    y = np.random.uniform(-1, 1, size=[30, 1]).astype(np.float32)

    locs = np.array([0.], dtype=np.float32)
    slopes = np.array([3.], dtype=np.float32)
    k = changepoint.ChangePoint([k1, k2], slopes=slopes, locs=locs)

    weights_x = tf.math.sigmoid(slopes * (x - locs))[..., 0]
    weights_y = tf.math.sigmoid(slopes * (y - locs))[..., 0]
    expected_kernel = (
        k1.apply(x, y, example_ndims=1) * (1 -  weights_x) * (1 - weights_y) +
        k2.apply(x, y, example_ndims=1) * weights_x * weights_y)
    actual_kernel = k.apply(x, y, example_ndims=1)
    self.assertAllClose(
        self.evaluate(expected_kernel), self.evaluate(actual_kernel))

    expected_kernel = (
        k1.matrix(x, y) * (1 -  weights_x)[..., tf.newaxis] * (1 - weights_y) +
        k2.matrix(x, y) * weights_x[..., tf.newaxis] * weights_y)
    actual_kernel = k.matrix(x, y)

    self.assertAllClose(
        self.evaluate(expected_kernel), self.evaluate(actual_kernel))

  @parameterized.parameters(
      {'feature_ndims': 1, 'dims': 3},
      {'feature_ndims': 1, 'dims': 4},
      {'feature_ndims': 2, 'dims': 2},
      {'feature_ndims': 2, 'dims': 3},
      {'feature_ndims': 3, 'dims': 2},
      {'feature_ndims': 3, 'dims': 3})
  def testValuesBetweenChangePoints(self, feature_ndims, dims):
    amplitude = 5.
    length_scale = 2.

    k1 = exponentiated_quadratic.ExponentiatedQuadratic(
        amplitude, length_scale, feature_ndims=feature_ndims)
    k2 = matern.MaternThreeHalves(
        amplitude, length_scale, feature_ndims=feature_ndims)
    k3 = matern.MaternOneHalf(
        amplitude, length_scale, feature_ndims=feature_ndims)

    # Default weight function is to sum over the dimensions.
    locs = 5. * feature_ndims * dims * np.array([-1., 1.], dtype=np.float32)
    # Use a large slope to ensure that we switch between the kernels.
    slopes = np.array([30.], dtype=np.float32)

    c = changepoint.ChangePoint([k1, k2, k3], locs=locs, slopes=slopes)

    shape = [dims] * feature_ndims
    for k, lims in [(k1, [-20., -10.]), (k2, [-2., 2.]), (k3, [10., 20.])]:
      x = np.random.uniform(*lims, size=shape).astype(np.float32)
      y = np.random.uniform(*lims, size=shape).astype(np.float32)
      self.assertAllClose(
          self.evaluate(k.apply(x, y)), self.evaluate(c.apply(x, y)))

      x = np.random.uniform(*lims, size=[3] + shape).astype(np.float32)
      y = np.random.uniform(*lims, size=[4] + shape).astype(np.float32)
      self.assertAllClose(
          self.evaluate(k.matrix(x, y)), self.evaluate(c.matrix(x, y)))


if __name__ == '__main__':
  test_util.main()
