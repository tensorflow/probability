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
"""Tests for NormalCDF Bijector."""

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
class NormalCDFBijectorTest(test_util.TestCase):
  """Tests correctness of the NormalCDF bijector."""

  def testBijector(self):
    bijector = tfb.NormalCDF(validate_args=True)
    self.assertStartsWith(bijector.name, "normal")
    x = np.array([[[-3.], [0.], [0.5], [4.2], [5.]]], dtype=np.float64)
    # Normal distribution
    normal_dist = stats.norm(loc=0., scale=1.)
    y = normal_dist.cdf(x).astype(np.float64)
    self.assertAllClose(y, self.evaluate(bijector.forward(x)))
    self.assertAllClose(x, self.evaluate(bijector.inverse(y)), rtol=1e-4)
    self.assertAllClose(
        np.squeeze(normal_dist.logpdf(x), axis=-1),
        self.evaluate(bijector.forward_log_det_jacobian(x, event_ndims=1)))
    self.assertAllClose(
        self.evaluate(-bijector.inverse_log_det_jacobian(y, event_ndims=1)),
        self.evaluate(bijector.forward_log_det_jacobian(x, event_ndims=1)),
        rtol=1e-4)

  def testValidateArgs(self):
    bijector = tfb.NormalCDF(validate_args=True)
    with self.assertRaisesOpError("must be greater than 0"):
      self.evaluate(bijector.inverse(-1.))

    with self.assertRaisesOpError("must be greater than 0"):
      self.evaluate(bijector.inverse_log_det_jacobian(-1., event_ndims=0))

    with self.assertRaisesOpError("must be less than or equal to 1"):
      self.evaluate(bijector.inverse(2.))

    with self.assertRaisesOpError("must be less than or equal to 1"):
      self.evaluate(bijector.inverse_log_det_jacobian(2., event_ndims=0))

  def testScalarCongruency(self):
    bijector_test_util.assert_scalar_congruency(
        tfb.NormalCDF(), lower_x=0., upper_x=1.,
        eval_func=self.evaluate, rtol=0.02)

  def testBijectiveAndFinite(self):
    bijector = tfb.NormalCDF(validate_args=True)
    x = np.linspace(-10., 10., num=10).astype(np.float32)
    y = np.linspace(0.1, 0.9, num=10).astype(np.float32)
    bijector_test_util.assert_bijective_and_finite(
        bijector, x, y, eval_func=self.evaluate, event_ndims=0, rtol=1e-4)


if __name__ == "__main__":
  tf.test.main()
