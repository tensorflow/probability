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

# Dependency imports

import numpy as np
from scipy import stats as scipy_stats
import tensorflow.compat.v2 as tf
from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python.bijectors import bijector_test_util
from tensorflow_probability.python.internal import test_util


@test_util.test_all_tf_execution_regimes
class RayleighCDFBijectorTest(test_util.TestCase):
  """Tests correctness of the rayleigh bijector."""

  def testBijector(self):
    scale = 50.
    bijector = tfb.RayleighCDF(scale=scale, validate_args=True)
    self.assertStartsWith(bijector.name, 'rayleigh')
    test_cdf_func = scipy_stats.rayleigh.cdf
    x = np.array([[[.1], [1.], [14.], [20.], [100.]]], dtype=np.float32)
    y = test_cdf_func(x, scale=scale).astype(np.float32)
    self.assertAllClose(y, self.evaluate(bijector.forward(x)))
    self.assertAllClose(x, self.evaluate(bijector.inverse(y)))
    self.assertAllClose(
        scipy_stats.rayleigh.logpdf(x, scale=scale),
        self.evaluate(bijector.forward_log_det_jacobian(x, event_ndims=0)))
    self.assertAllClose(
        self.evaluate(-bijector.inverse_log_det_jacobian(y, event_ndims=0)),
        self.evaluate(bijector.forward_log_det_jacobian(x, event_ndims=0)),
        rtol=1e-4,
        atol=0.)

  def testBijectorLogDetJacobianZeroAtZero(self):
    scale = np.logspace(0.1, 10., num=20).astype(np.float32)
    bijector = tfb.RayleighCDF(scale)
    fldj = self.evaluate(bijector.forward_log_det_jacobian(0., event_ndims=0))
    self.assertAllNegativeInf(fldj)

  def testScalarCongruency(self):
    bijector_test_util.assert_scalar_congruency(
        tfb.RayleighCDF(scale=50.),
        lower_x=1.,
        upper_x=100.,
        eval_func=self.evaluate,
        rtol=0.05)

  def testBijectiveAndFinite(self):
    bijector = tfb.RayleighCDF(scale=20., validate_args=True)
    x = np.linspace(1., 8., num=10).astype(np.float32)
    y = np.linspace(
        -np.expm1(-1 / 400.),
        -np.expm1(-16), num=10).astype(np.float32)
    bijector_test_util.assert_bijective_and_finite(
        bijector, x, y, eval_func=self.evaluate, event_ndims=0, rtol=1e-3)

  def testAsserts(self):
    with self.assertRaisesOpError('Argument `scale` must be positive.'):
      b = tfb.RayleighCDF(scale=-1., validate_args=True)
      self.evaluate(b.forward(3.))

  @test_util.jax_disable_variable_test
  def testVariableAssertsScale(self):
    scale = tf.Variable(1.)
    b = tfb.RayleighCDF(scale=scale, validate_args=True)
    self.evaluate([scale.initializer])
    with self.assertRaisesOpError('Argument `scale` must be positive.'):
      with tf.control_dependencies([scale.assign(-1.)]):
        self.evaluate(b.forward(3.))


if __name__ == '__main__':
  test_util.main()
