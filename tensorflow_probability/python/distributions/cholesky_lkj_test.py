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
"""Tests for the Cholesky LKJ distribution."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import test_case
from tensorflow_probability.python.internal import test_util as tfp_test_util
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import,


tfb = tfp.bijectors
tfd = tfp.distributions


@test_util.run_all_in_graph_and_eager_modes
@parameterized.parameters(np.float32, np.float64)
class CholeksyLKJTest(parameterized.TestCase, test_case.TestCase):

  def testLogProbMatchesTransformedDistribution(self, dtype):
    dtype = np.float64
    for dims in (3, 4, 5):
      concentration = np.linspace(2., 5., 10, dtype=dtype)
      cholesky_lkj = tfd.CholeskyLKJ(
          concentration=concentration, dimension=dims)
      transformed_lkj = tfd.TransformedDistribution(
          bijector=tfb.Invert(tfb.CholeskyOuterProduct()),
          distribution=tfd.LKJ(concentration=concentration, dimension=dims))

      # Choose input that has well conditioned matrices.
      x = self.evaluate(cholesky_lkj.sample(10, seed=tfp_test_util.test_seed()))
      self.assertAllClose(
          self.evaluate(cholesky_lkj.log_prob(x)),
          self.evaluate(transformed_lkj.log_prob(x)))

  def testDimensionGuard(self, dtype):
    testee_lkj = tfd.CholeskyLKJ(
        dimension=3, concentration=dtype([1., 4.]), validate_args=True)
    with self.assertRaisesRegexp(ValueError, 'dimension mismatch'):
      testee_lkj.log_prob(tf.eye(4))

  def testZeroDimension(self, dtype):
    testee_lkj = tfd.CholeskyLKJ(
        dimension=0, concentration=dtype([1., 4.]), validate_args=True)
    results = testee_lkj.sample(sample_shape=[4, 3])
    self.assertEqual(results.shape, [4, 3, 2, 0, 0])

  def testOneDimension(self, dtype):
    testee_lkj = tfd.CholeskyLKJ(
        dimension=1, concentration=dtype([1., 4.]), validate_args=True)
    results = testee_lkj.sample(sample_shape=[4, 3])
    self.assertEqual(results.shape, [4, 3, 2, 1, 1])

  def testValidateLowerTriangularInput(self, dtype):
    testee_lkj = tfd.CholeskyLKJ(
        dimension=2, concentration=dtype(4.), validate_args=True)
    with self.assertRaisesOpError('must be lower triangular'):
      self.evaluate(testee_lkj.log_prob(dtype([[1., 1.], [1., 1.]])))

  def testValidateConcentration(self, dtype):
    dimension = 3
    concentration = tf.Variable(0.5, dtype=dtype)
    d = tfd.CholeskyLKJ(dimension, concentration, validate_args=True)
    with self.assertRaisesOpError('Argument `concentration` must be >= 1.'):
      self.evaluate([v.initializer for v in d.variables])
      self.evaluate(d.sample())

  def testValidateConcentrationAfterMutation(self, dtype):
    dimension = 3
    concentration = tf.Variable(1.5, dtype=dtype)
    d = tfd.CholeskyLKJ(dimension, concentration, validate_args=True)
    self.evaluate([v.initializer for v in d.variables])
    with self.assertRaisesOpError('Argument `concentration` must be >= 1.'):
      with tf.control_dependencies([concentration.assign(0.5)]):
        self.evaluate(d.sample())


if __name__ == '__main__':
  tf.test.main()
