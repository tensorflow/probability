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
"""Tests for PoissonLogNormalQuadratureCompoundTest."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.internal import test_util


class _PoissonLogNormalQuadratureCompoundTest(
    test_util.DiscreteScalarDistributionTestHelpers):
  """Tests the PoissonLogNormalQuadratureCompoundTest distribution."""

  def testSampleProbConsistent(self):
    pln = tfd.PoissonLogNormalQuadratureCompound(
        loc=tf1.placeholder_with_default(
            -2., shape=[] if self.static_shape else None),
        scale=tf1.placeholder_with_default(
            1.1, shape=[] if self.static_shape else None),
        quadrature_size=10,
        validate_args=True)
    self.run_test_sample_consistent_log_prob(
        self.evaluate, pln, batch_size=1, rtol=0.1)

  def testMeanVariance(self):
    pln = tfd.PoissonLogNormalQuadratureCompound(
        loc=tf1.placeholder_with_default(
            0., shape=[] if self.static_shape else None),
        scale=tf1.placeholder_with_default(
            1., shape=[] if self.static_shape else None),
        quadrature_size=10,
        validate_args=True)
    self.run_test_sample_consistent_mean_variance(self.evaluate, pln, rtol=0.02)

  def testSampleProbConsistentBroadcastScalar(self):
    pln = tfd.PoissonLogNormalQuadratureCompound(
        loc=tf1.placeholder_with_default(
            [0., -0.5], shape=[2] if self.static_shape else None),
        scale=tf1.placeholder_with_default(
            1., shape=[] if self.static_shape else None),
        quadrature_size=10,
        validate_args=True)
    self.run_test_sample_consistent_log_prob(
        self.evaluate, pln, batch_size=2, rtol=0.1, atol=0.01)

  def testMeanVarianceBroadcastScalar(self):
    pln = tfd.PoissonLogNormalQuadratureCompound(
        loc=tf1.placeholder_with_default(
            [0., -0.5], shape=[2] if self.static_shape else None),
        scale=tf1.placeholder_with_default(
            1., shape=[] if self.static_shape else None),
        quadrature_size=10,
        validate_args=True)
    self.run_test_sample_consistent_mean_variance(
        self.evaluate, pln, rtol=0.1, atol=0.01)

  def testSampleProbConsistentBroadcastBoth(self):
    pln = tfd.PoissonLogNormalQuadratureCompound(
        loc=tf1.placeholder_with_default(
            [[0.], [-0.5]], shape=[2, 1] if self.static_shape else None),
        scale=tf1.placeholder_with_default(
            [[1., 0.9]], shape=[1, 2] if self.static_shape else None),
        quadrature_size=10,
        validate_args=True)
    self.run_test_sample_consistent_log_prob(
        self.evaluate, pln, batch_size=4, rtol=0.1, atol=0.08)

  def testMeanVarianceBroadcastBoth(self):
    pln = tfd.PoissonLogNormalQuadratureCompound(
        loc=tf1.placeholder_with_default(
            [[0.], [-0.5]], shape=[2, 1] if self.static_shape else None),
        scale=tf1.placeholder_with_default(
            [[1., 0.9]], shape=[1, 2] if self.static_shape else None),
        quadrature_size=10,
        validate_args=True)
    self.run_test_sample_consistent_mean_variance(
        self.evaluate, pln, rtol=0.1, atol=0.01)

  @test_util.tf_tape_safety_test
  def testGradientThroughParams(self):
    pln = tfd.PoissonLogNormalQuadratureCompound(
        loc=tf.Variable([0., -0.5], shape=[2] if self.static_shape
                        else None),
        scale=tf.Variable([1., 0.9], shape=[2] if self.static_shape
                          else None),
        quadrature_size=10, validate_args=True)
    with tf.GradientTape() as tape:
      loss = -pln.log_prob([1., 2.])
    grad = tape.gradient(loss, pln.trainable_variables)
    self.assertLen(grad, 2)
    self.assertFalse(any([g is None for g in grad]))

  @test_util.tf_tape_safety_test
  def testGradientThroughNonVariableParams(self):
    pln = tfd.PoissonLogNormalQuadratureCompound(
        loc=tf.convert_to_tensor([0., -0.5]),
        scale=tf.convert_to_tensor([1., 0.9]),
        quadrature_size=10, validate_args=True)
    with tf.GradientTape() as tape:
      tape.watch(pln.loc)
      tape.watch(pln.scale)
      loss = -pln.log_prob([1., 2.])
    grad = tape.gradient(loss, [pln.loc, pln.scale])
    self.assertLen(grad, 2)
    self.assertFalse(any([g is None for g in grad]))

  def testAssertValidSample(self):
    pln = tfd.PoissonLogNormalQuadratureCompound(
        loc=tf1.placeholder_with_default(
            0., shape=[] if self.static_shape else None),
        scale=tf1.placeholder_with_default(
            1., shape=[] if self.static_shape else None),
        quadrature_size=10,
        validate_args=True)
    with self.assertRaisesOpError('Sample must be non-negative.'):
      self.evaluate(pln.log_prob([-1.2, 3., 4.2]))

  def testPdfBoundary(self):
    pln = tfd.PoissonLogNormalQuadratureCompound(
        loc=tf1.placeholder_with_default(
            0., shape=[] if self.static_shape else None),
        scale=tf1.placeholder_with_default(
            1., shape=[] if self.static_shape else None),
        quadrature_size=10,
        validate_args=True)

    pdf = self.evaluate(pln.prob(0.))
    log_pdf = self.evaluate(pln.log_prob(0.))
    self.assertAllFinite(pdf)
    self.assertAllFinite(log_pdf)


@test_util.test_all_tf_execution_regimes
class PoissonLogNormalQuadratureCompoundStaticShapeTest(
    _PoissonLogNormalQuadratureCompoundTest, test_util.TestCase):

  @property
  def static_shape(self):
    return True


@test_util.test_all_tf_execution_regimes
class PoissonLogNormalQuadratureCompoundDynamicShapeTest(
    _PoissonLogNormalQuadratureCompoundTest, test_util.TestCase):

  @property
  def static_shape(self):
    return False


if __name__ == '__main__':
  tf.test.main()
