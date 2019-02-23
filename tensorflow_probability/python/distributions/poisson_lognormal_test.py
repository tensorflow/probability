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

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.internal import test_util as tfp_test_util

tfd = tfp.distributions
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top


class _PoissonLogNormalQuadratureCompoundTest(
    tfp_test_util.DiscreteScalarDistributionTestHelpers):
  """Tests the PoissonLogNormalQuadratureCompoundTest distribution."""

  def testSampleProbConsistent(self):
    pln = tfd.PoissonLogNormalQuadratureCompound(
        loc=tf.compat.v1.placeholder_with_default(
            -2., shape=[] if self.static_shape else None),
        scale=tf.compat.v1.placeholder_with_default(
            1.1, shape=[] if self.static_shape else None),
        quadrature_size=10,
        validate_args=True)
    self.run_test_sample_consistent_log_prob(
        self.evaluate, pln, batch_size=1, rtol=0.1)

  def testMeanVariance(self):
    pln = tfd.PoissonLogNormalQuadratureCompound(
        loc=tf.compat.v1.placeholder_with_default(
            0., shape=[] if self.static_shape else None),
        scale=tf.compat.v1.placeholder_with_default(
            1., shape=[] if self.static_shape else None),
        quadrature_size=10,
        validate_args=True)
    self.run_test_sample_consistent_mean_variance(self.evaluate, pln, rtol=0.02)

  def testSampleProbConsistentBroadcastScalar(self):
    pln = tfd.PoissonLogNormalQuadratureCompound(
        loc=tf.compat.v1.placeholder_with_default(
            [0., -0.5], shape=[2] if self.static_shape else None),
        scale=tf.compat.v1.placeholder_with_default(
            1., shape=[] if self.static_shape else None),
        quadrature_size=10,
        validate_args=True)
    self.run_test_sample_consistent_log_prob(
        self.evaluate, pln, batch_size=2, rtol=0.1, atol=0.01)

  def testMeanVarianceBroadcastScalar(self):
    pln = tfd.PoissonLogNormalQuadratureCompound(
        loc=tf.compat.v1.placeholder_with_default(
            [0., -0.5], shape=[2] if self.static_shape else None),
        scale=tf.compat.v1.placeholder_with_default(
            1., shape=[] if self.static_shape else None),
        quadrature_size=10,
        validate_args=True)
    self.run_test_sample_consistent_mean_variance(
        self.evaluate, pln, rtol=0.1, atol=0.01)

  def testSampleProbConsistentBroadcastBoth(self):
    pln = tfd.PoissonLogNormalQuadratureCompound(
        loc=tf.compat.v1.placeholder_with_default(
            [[0.], [-0.5]], shape=[2, 1] if self.static_shape else None),
        scale=tf.compat.v1.placeholder_with_default(
            [[1., 0.9]], shape=[1, 2] if self.static_shape else None),
        quadrature_size=10,
        validate_args=True)
    self.run_test_sample_consistent_log_prob(
        self.evaluate, pln, batch_size=4, rtol=0.1, atol=0.08)

  def testMeanVarianceBroadcastBoth(self):
    pln = tfd.PoissonLogNormalQuadratureCompound(
        loc=tf.compat.v1.placeholder_with_default(
            [[0.], [-0.5]], shape=[2, 1] if self.static_shape else None),
        scale=tf.compat.v1.placeholder_with_default(
            [[1., 0.9]], shape=[1, 2] if self.static_shape else None),
        quadrature_size=10,
        validate_args=True)
    self.run_test_sample_consistent_mean_variance(
        self.evaluate, pln, rtol=0.1, atol=0.01)


@test_util.run_all_in_graph_and_eager_modes
class PoissonLogNormalQuadratureCompoundStaticShapeTest(
    _PoissonLogNormalQuadratureCompoundTest, tf.test.TestCase):

  @property
  def static_shape(self):
    return True


@test_util.run_all_in_graph_and_eager_modes
class PoissonLogNormalQuadratureCompoundDynamicShapeTest(
    _PoissonLogNormalQuadratureCompoundTest, tf.test.TestCase):

  @property
  def static_shape(self):
    return False


if __name__ == "__main__":
  tf.test.main()
