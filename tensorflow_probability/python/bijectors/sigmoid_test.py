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
"""Sigmoid Tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
from scipy import special
import tensorflow as tf
from tensorflow_probability.python import bijectors as tfb

from tensorflow.python.ops.distributions.bijector_test_util import assert_bijective_and_finite
from tensorflow.python.ops.distributions.bijector_test_util import assert_scalar_congruency


class SigmoidBijectorTest(tf.test.TestCase):
  """Tests correctness of the Y = g(X) = (1 + exp(-X))^-1 transformation."""

  def testBijector(self):
    with self.test_session():
      self.assertEqual("sigmoid", tfb.Sigmoid().name)
      x = np.linspace(-10., 10., 100).reshape([2, 5, 10]).astype(np.float32)
      y = special.expit(x)
      ildj = -np.log(y) - np.log1p(-y)
      bijector = tfb.Sigmoid()
      self.assertAllClose(y, bijector.forward(x).eval(), atol=0., rtol=1e-2)
      self.assertAllClose(x, bijector.inverse(y).eval(), atol=0., rtol=1e-4)
      self.assertAllClose(ildj, bijector.inverse_log_det_jacobian(
          y, event_ndims=0).eval(), atol=0., rtol=1e-6)
      self.assertAllClose(-ildj, bijector.forward_log_det_jacobian(
          x, event_ndims=0).eval(), atol=0., rtol=1e-4)

  def testScalarCongruency(self):
    with self.test_session():
      assert_scalar_congruency(tfb.Sigmoid(), lower_x=-7., upper_x=7.)

  def testBijectiveAndFinite(self):
    with self.test_session():
      x = np.linspace(-7., 7., 100).astype(np.float32)
      eps = 1e-3
      y = np.linspace(eps, 1. - eps, 100).astype(np.float32)
      assert_bijective_and_finite(
          tfb.Sigmoid(), x, y, event_ndims=0, atol=0., rtol=1e-4)


if __name__ == "__main__":
  tf.test.main()
