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
"""Tests for Bijector."""

import numpy as np
from scipy import stats
import tensorflow.compat.v2 as tf
from tensorflow_probability.python.bijectors import bijector_test_util
from tensorflow_probability.python.bijectors import gev_cdf
from tensorflow_probability.python.internal import test_util


@test_util.test_all_tf_execution_regimes
class GeneralizedExtremeValueCDFTest(test_util.TestCase):
  """Tests correctness of the GeneralizedExtremeValue bijector."""

  def testBijector(self):
    loc = 0.3
    scale = 5.
    concentration = np.array([[[-5.5], [-20], [0.], [1.]]], dtype=np.float32)
    bijector = gev_cdf.GeneralizedExtremeValueCDF(
        loc=loc, scale=scale, concentration=concentration, validate_args=True)
    self.assertStartsWith(bijector.name, "generalizedextremevalue")
    x = np.array([[[0.], [-3.], [0.], [4.2]]], dtype=np.float32)
    # GeneralizedExtremeValue distribution
    gev_dist = stats.genextreme(-concentration, loc=loc, scale=scale)
    y = gev_dist.cdf(x).astype(np.float32)
    self.assertAllClose(y, self.evaluate(bijector.forward(x)))
    self.assertAllClose(x, self.evaluate(bijector.inverse(y)))
    self.assertAllClose(
        np.squeeze(gev_dist.logpdf(x), axis=-1),
        self.evaluate(bijector.forward_log_det_jacobian(x, event_ndims=1)))
    self.assertAllClose(
        self.evaluate(-bijector.inverse_log_det_jacobian(y, event_ndims=1)),
        self.evaluate(bijector.forward_log_det_jacobian(x, event_ndims=1)),
        rtol=1e-4,
        atol=0.)

  def testScalarCongruency(self):
    bijector_test_util.assert_scalar_congruency(
        gev_cdf.GeneralizedExtremeValueCDF(
            loc=0.3, scale=20., concentration=0.2),
        lower_x=1.,
        upper_x=100.,
        eval_func=self.evaluate,
        rtol=0.05)

  def testBijectiveAndFinite(self):
    bijector = gev_cdf.GeneralizedExtremeValueCDF(
        loc=0., scale=3.0, concentration=2.0, validate_args=True)
    x = np.linspace(-1.4, 10., num=10).astype(np.float32)
    y = np.linspace(0.01, 0.99, num=10).astype(np.float32)
    bijector_test_util.assert_bijective_and_finite(
        bijector, x, y, eval_func=self.evaluate, event_ndims=0, rtol=1e-3)

  @test_util.jax_disable_variable_test
  def testVariableScale(self):
    x = tf.Variable(1.)
    b = gev_cdf.GeneralizedExtremeValueCDF(
        loc=0., scale=x, concentration=0.6, validate_args=True)
    self.evaluate(x.initializer)
    self.assertIs(x, b.scale)
    self.assertEqual((), self.evaluate(b.forward(1.)).shape)
    with self.assertRaisesOpError("Argument `scale` must be positive."):
      with tf.control_dependencies([x.assign(-1.)]):
        self.evaluate(b.forward(2.))


if __name__ == "__main__":
  test_util.main()
