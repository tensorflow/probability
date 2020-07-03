# Copyright 2020 The TensorFlow Probability Authors.#
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
"""Tests for the Frechet Bijector."""

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
class FrechetCDFTest(test_util.TestCase):
  """Tests correctness of the Frechet bijector."""

  def testBijector(self):
    # TODO(b/152525415): varied/extreme values, forward(+inf/-inf) and inv(0/1)
    loc = np.array(0.3, dtype=np.float64)
    scale = np.array(5., dtype=np.float64)
    concentration = np.array(2., dtype=np.float64)
    bijector = tfb.FrechetCDF(loc=loc, scale=scale, concentration=concentration,
                              validate_args=True)
    self.assertStartsWith(bijector.name, 'frechet')
    # Frechet distribution
    frechet_dist = stats.invweibull(c=concentration, loc=loc, scale=scale)
    x = np.array([[[0.3001], [0.8], [2.], [4.2], [12.]]], dtype=np.float64)
    y = frechet_dist.cdf(x).astype(np.float64)
    self.assertAllClose(y, self.evaluate(bijector.forward(x)))
    x = np.array([[[0.49], [0.8], [2.], [4.2], [12.]]], dtype=np.float64)
    y = frechet_dist.cdf(x).astype(np.float64)
    # the below tests fail if x < 0.49
    self.assertAllClose(x, self.evaluate(bijector.inverse(y)))
    self.assertAllClose(
        np.squeeze(frechet_dist.logpdf(x), axis=-1),
        self.evaluate(bijector.forward_log_det_jacobian(x, event_ndims=1)))
    self.assertAllClose(
        self.evaluate(-bijector.inverse_log_det_jacobian(y, event_ndims=1)),
        self.evaluate(bijector.forward_log_det_jacobian(x, event_ndims=1)),
        rtol=1e-4,
        atol=0.)
    with self.assertRaisesOpError(r'Forward transformation input.*than `loc`.'):
      self.evaluate(bijector.forward(0.1))
    with self.assertRaisesOpError(r'Inverse transformation input.* 0.'):
      self.evaluate(bijector.inverse(-0.1))
    with self.assertRaisesOpError(r'Inverse transformation input.* 1.'):
      self.evaluate(bijector.inverse(1.1))

  def testScalarCongruency(self):
    loc = np.array(-1., np.float64)
    bijector_test_util.assert_scalar_congruency(
        tfb.FrechetCDF(loc=loc, scale=20., concentration=0.5), lower_x=1.,
        upper_x=100., eval_func=self.evaluate, rtol=0.05)

  def testBijectiveAndFinite(self):
    loc = np.array(-1., np.float64)
    bijector = tfb.FrechetCDF(loc=loc, scale=3.0, concentration=2.,
                              validate_args=True)
    x = np.linspace(loc+0.25, 10., num=10).astype(np.float64)
    y = np.linspace(0.01, 0.99, num=10).astype(np.float64)
    bijector_test_util.assert_bijective_and_finite(
        bijector, x, y, eval_func=self.evaluate, event_ndims=0, rtol=1e-3)

  @test_util.jax_disable_variable_test
  def testVariablesScaleAndconcentration(self):
    x = tf.Variable(1.)
    y = tf.Variable(1.)
    b = tfb.FrechetCDF(loc=0., scale=x, concentration=y, validate_args=True)
    self.evaluate(x.initializer)
    self.evaluate(y.initializer)
    self.assertIs(x, b.scale)
    self.assertIs(y, b.concentration)
    self.assertEqual((), self.evaluate(b.forward(3.)).shape)
    with self.assertRaisesOpError('Argument `scale` must be positive.'):
      with tf.control_dependencies([x.assign(-1.)]):
        self.evaluate(b.forward(3.))
    with self.assertRaisesOpError('Argument `concentration` must be positive.'):
      with tf.control_dependencies([x.assign(1), y.assign(-1.)]):
        self.evaluate(b.forward(3.))


if __name__ == '__main__':
  tf.test.main()
