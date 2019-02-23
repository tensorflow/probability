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
"""Tests for Kumaraswamy Bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
import tensorflow as tf
from tensorflow_probability.python import bijectors as tfb

from tensorflow_probability.python.bijectors import bijector_test_util
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top


@test_util.run_all_in_graph_and_eager_modes
class KumaraswamyBijectorTest(tf.test.TestCase):
  """Tests correctness of the Kumaraswamy bijector."""

  def testBijector(self):
    a = 2.
    b = 0.3
    bijector = tfb.Kumaraswamy(
        concentration1=a, concentration0=b, validate_args=True)
    self.assertEqual("kumaraswamy", bijector.name)
    x = np.array([[[0.1], [0.2], [0.3], [0.4], [0.5]]], dtype=np.float32)
    # Kumaraswamy cdf. This is the same as inverse(x).
    y = 1. - (1. - x ** a) ** b
    self.assertAllClose(y, self.evaluate(bijector.inverse(x)))
    self.assertAllClose(x, self.evaluate(bijector.forward(y)))
    kumaraswamy_log_pdf = (np.log(a) + np.log(b) + (a - 1) * np.log(x) +
                           (b - 1) * np.log1p(-x ** a))

    self.assertAllClose(
        np.squeeze(kumaraswamy_log_pdf, axis=-1),
        self.evaluate(bijector.inverse_log_det_jacobian(x, event_ndims=1)))
    self.assertAllClose(
        self.evaluate(-bijector.inverse_log_det_jacobian(x, event_ndims=1)),
        self.evaluate(bijector.forward_log_det_jacobian(y, event_ndims=1)),
        rtol=1e-4,
        atol=0.)

  def testBijectorConcentration1LogDetJacobianFiniteAtZero(self):
    # When concentration = 1., forward_log_det_jacobian should be finite at
    # zero.
    concentration0 = np.logspace(0.1, 10., num=20).astype(np.float32)
    bijector = tfb.Kumaraswamy(concentration1=1., concentration0=concentration0)
    fldj = self.evaluate(
        bijector.forward_log_det_jacobian(0., event_ndims=0))
    self.assertAllEqual(np.ones_like(fldj, dtype=np.bool), np.isfinite(fldj))

  def testScalarCongruency(self):
    bijector_test_util.assert_scalar_congruency(
        tfb.Kumaraswamy(concentration1=0.5, concentration0=1.1),
        lower_x=0.,
        upper_x=1.,
        eval_func=self.evaluate,
        n=int(10e3),
        rtol=0.02)

  def testBijectiveAndFinite(self):
    concentration1 = 1.2
    concentration0 = 2.
    bijector = tfb.Kumaraswamy(
        concentration1=concentration1,
        concentration0=concentration0,
        validate_args=True)
    # Omitting the endpoints 0 and 1, since idlj will be infinity at these
    # endpoints.
    y = np.linspace(.01, 0.99, num=10).astype(np.float32)
    x = 1 - (1 - y ** concentration1) ** concentration0
    bijector_test_util.assert_bijective_and_finite(
        bijector, x, y, eval_func=self.evaluate, event_ndims=0,
        rtol=1e-3)


if __name__ == "__main__":
  tf.test.main()
