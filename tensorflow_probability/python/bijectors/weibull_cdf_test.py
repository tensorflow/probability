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
"""Tests for Bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import numpy as np
from scipy import stats
import tensorflow.compat.v2 as tf
from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python.bijectors import bijector_test_util
from tensorflow_probability.python.internal import test_util


@test_util.test_all_tf_execution_regimes
class WeibullCDFBijectorTest(test_util.TestCase):
  """Tests correctness of the weibull bijector."""

  def testBijector(self):
    scale = 5.
    concentration = 0.3
    bijector = tfb.WeibullCDF(
        scale=scale, concentration=concentration, validate_args=True)
    self.assertStartsWith(bijector.name, 'weibull')
    x = np.array([[[0.], [1.], [14.], [20.], [100.]]], dtype=np.float32)
    # Weibull distribution
    weibull_dist = stats.frechet_r(c=concentration, scale=scale)
    y = weibull_dist.cdf(x).astype(np.float32)
    self.assertAllClose(y, self.evaluate(bijector.forward(x)))
    self.assertAllClose(x, self.evaluate(bijector.inverse(y)))
    self.assertAllClose(
        weibull_dist.logpdf(x),
        self.evaluate(bijector.forward_log_det_jacobian(x, event_ndims=0)))
    self.assertAllClose(
        self.evaluate(-bijector.inverse_log_det_jacobian(y, event_ndims=0)),
        self.evaluate(bijector.forward_log_det_jacobian(x, event_ndims=0)),
        rtol=1e-4,
        atol=0.)

  def testBijectorConcentration1LogDetJacobianFiniteAtZero(self):
    # When concentration = 1., forward_log_det_jacobian should be finite at
    # zero.
    scale = np.logspace(0.1, 10., num=20).astype(np.float32)
    bijector = tfb.WeibullCDF(scale, concentration=1.)
    fldj = self.evaluate(bijector.forward_log_det_jacobian(0., event_ndims=0))
    self.assertAllEqual(np.ones_like(fldj, dtype=np.bool), np.isfinite(fldj))

  def testScalarCongruency(self):
    bijector_test_util.assert_scalar_congruency(
        tfb.WeibullCDF(scale=20., concentration=0.3),
        lower_x=1.,
        upper_x=100.,
        eval_func=self.evaluate,
        rtol=0.05)

  def testBijectiveAndFinite(self):
    bijector = tfb.WeibullCDF(scale=20., concentration=2., validate_args=True)
    x = np.linspace(1., 8., num=10).astype(np.float32)
    y = np.linspace(
        -np.expm1(-1 / 400.),
        -np.expm1(-16), num=10).astype(np.float32)
    bijector_test_util.assert_bijective_and_finite(
        bijector, x, y, eval_func=self.evaluate, event_ndims=0, rtol=1e-3)

  def testAsserts(self):
    with self.assertRaisesOpError('Argument `scale` must be positive.'):
      b = tfb.WeibullCDF(
          concentration=1., scale=-1., validate_args=True)
      self.evaluate(b.forward(3.))
    with self.assertRaisesOpError('Argument `concentration` must be positive.'):
      b = tfb.WeibullCDF(
          concentration=-1., scale=1., validate_args=True)
      self.evaluate(b.inverse(0.5))

  @test_util.jax_disable_variable_test
  def testVariableAssertsScale(self):
    concentration = tf.Variable(1.)
    scale = tf.Variable(1.)
    b = tfb.WeibullCDF(
        concentration=concentration, scale=scale, validate_args=True)
    self.evaluate([concentration.initializer, scale.initializer])
    with self.assertRaisesOpError('Argument `scale` must be positive.'):
      with tf.control_dependencies([scale.assign(-1.)]):
        self.evaluate(b.forward(3.))

  @test_util.jax_disable_variable_test
  def testVariableAssertsConcentration(self):
    concentration = tf.Variable(1.)
    scale = tf.Variable(1.)
    b = tfb.WeibullCDF(
        concentration=concentration, scale=scale, validate_args=True)
    self.evaluate([concentration.initializer, scale.initializer])
    with self.assertRaisesOpError('Argument `concentration` must be positive.'):
      with tf.control_dependencies([concentration.assign(-1.)]):
        self.evaluate(b.inverse(0.5))


if __name__ == '__main__':
  tf.test.main()
