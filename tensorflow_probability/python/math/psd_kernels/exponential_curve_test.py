# Copyright 2022 The TensorFlow Probability Authors.
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
"""Tests for ExponentialCurve kernel."""

from absl.testing import parameterized

import numpy as np
import tensorflow.compat.v2 as tf

import tensorflow_probability as tfp

from tensorflow_probability.python.internal import test_util


@test_util.test_all_tf_execution_regimes
class ExponentialCurveTest(test_util.TestCase):

  def _numpyKernel(self, concentration, rate, x, y):
    return np.power(rate / (x[..., 0] + y[..., 0] + rate), concentration)

  def testBatchShape(self):
    concentration = np.random.uniform(2, 3., size=[3, 1, 2]).astype(np.float32)
    rate = np.random.uniform(2, 3., size=[1, 3, 1]).astype(np.float32)
    k = tfp.math.psd_kernels.ExponentialCurve(concentration, rate)
    self.assertAllEqual(tf.TensorShape([3, 3, 2]), k.batch_shape)
    self.assertAllEqual([3, 3, 2], self.evaluate(k.batch_shape_tensor()))

  def testValidateArgs(self):
    with self.assertRaisesOpError('concentration must be positive'):
      k = tfp.math.psd_kernels.ExponentialCurve(
          concentration=-1., rate=1., validate_args=True)
      self.evaluate(k.apply([[1.]], [[1.]]))

    if not tf.executing_eagerly():
      with self.assertRaisesOpError('rate must be positive'):
        k = tfp.math.psd_kernels.ExponentialCurve(
            concentration=1., rate=-1., validate_args=True)
        self.evaluate(k.apply([[1.]], [[1.]]))

  @parameterized.parameters({
      'dtype': np.float32,
      'batch_size': 3
  }, {
      'dtype': np.float32,
      'batch_size': 4
  })
  def testValuesAreCorrect(self, dtype, batch_size):
    concentration = np.array(5., dtype=dtype)
    rate = np.array(.2, dtype=dtype)

    rng = test_util.test_np_rng()
    k = tfp.math.psd_kernels.ExponentialCurve(concentration, rate)
    for _ in range(5):
      x = rng.uniform(0, 2, size=[batch_size, 3]).astype(dtype)
      y = rng.uniform(0, 2, size=[batch_size, 1]).astype(dtype)
      self.assertAllClose(
          self._numpyKernel(concentration, rate,
                            x.sum(-1, keepdims=True),
                            3.0 * y.sum(axis=-1, keepdims=True)),
          self.evaluate(k.apply(x, y)))

  def testShapesAreCorrect(self):
    k = tfp.math.psd_kernels.ExponentialCurve(concentration=1., rate=1.)

    x = np.ones([4, 1], np.float32)
    y = np.ones([5, 1], np.float32)

    self.assertAllEqual([4, 5], k.matrix(x, y).shape)
    self.assertAllEqual(
        k.matrix(tf.stack([x] * 2), tf.stack([y] * 2)).shape, [2, 4, 5])

    k = tfp.math.psd_kernels.ExponentialCurve(
        concentration=np.ones([2, 1, 1], np.float32),
        rate=np.ones([1, 3, 1], np.float32))
    self.assertAllEqual(
        [2, 3, 2, 4, 5],
        #`--'  |  `--'
        #  |   |    `- matrix shape
        #  |   `- from input batch shapes
        #  `- from broadcasting kernel params
        k.matrix(
            tf.stack([x] * 2),  # shape [2, 4, 3]
            tf.stack([y] * 2)  # shape [2, 5, 3]
        ).shape)

  @test_util.numpy_disable_gradient_test
  def testGradsAtIdenticalInputsAreNotNaN(self):
    k = tfp.math.psd_kernels.ExponentialCurve(concentration=1., rate=1.)
    x = tf.constant(np.arange(3 * 5, dtype=np.float32).reshape(15, 1))

    grads = [
        tfp.math.value_and_gradient(
            lambda x: k.apply(x, x)[i], x)[1]  # pylint: disable=cell-var-from-loop
        for i in range(3)]

    self.assertAllNotNan(self.evaluate(grads))


if __name__ == '__main__':
  test_util.main()
