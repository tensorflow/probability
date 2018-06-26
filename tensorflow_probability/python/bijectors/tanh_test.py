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
"""Tanh Tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
import tensorflow as tf
from tensorflow_probability.python import bijectors as tfb

from tensorflow.python.ops.distributions.bijector_test_util import assert_bijective_and_finite
from tensorflow.python.ops.distributions.bijector_test_util import assert_scalar_congruency


class TanhBijectorTest(tf.test.TestCase):
  """Tests correctness of the Y = g(X) = tanh(X) transformation."""

  def testBijector(self):
    with self.test_session():
      self.assertEqual("tanh", tfb.Tanh().name)
      x = np.linspace(-3., 3., 100).reshape([2, 5, 10]).astype(np.float32)
      y = np.tanh(x)
      ildj = -np.log1p(-np.square(np.tanh(x)))
      bijector = tfb.Tanh()
      self.assertAllClose(y, bijector.forward(x).eval(), atol=0., rtol=1e-2)
      self.assertAllClose(x, bijector.inverse(y).eval(), atol=0., rtol=1e-4)
      self.assertAllClose(ildj, bijector.inverse_log_det_jacobian(
          y, event_ndims=0).eval(), atol=0., rtol=1e-6)
      self.assertAllClose(-ildj, bijector.forward_log_det_jacobian(
          x, event_ndims=0).eval(), atol=0., rtol=1e-4)

  def testScalarCongruency(self):
    with self.test_session():
      assert_scalar_congruency(tfb.Tanh(), lower_x=-9., upper_x=9., n=int(10e4))

  def testBijectiveAndFinite(self):
    with self.test_session():
      x = np.linspace(-5., 5., 100).astype(np.float32)
      eps = 1e-3
      y = np.linspace(eps, 1. - eps, 100).astype(np.float32)
      assert_bijective_and_finite(
          tfb.Tanh(), x, y, event_ndims=0, atol=0., rtol=1e-4)

  def testMatchWithAffineTransform(self):
    with self.test_session():
      direct_bj = tfb.Tanh()
      indirect_bj = tfb.Chain([
          tfb.AffineScalar(shift=tf.to_double(-1.0), scale=tf.to_double(2.0)),
          tfb.Sigmoid(),
          tfb.AffineScalar(scale=tf.to_double(2.0))])

      x = np.linspace(-3.0, 3.0, 100)
      y = np.tanh(x)
      self.assertAllClose(direct_bj.forward(x).eval(),
                          indirect_bj.forward(x).eval())
      self.assertAllClose(direct_bj.inverse(y).eval(),
                          indirect_bj.inverse(y).eval())
      self.assertAllClose(
          direct_bj.inverse_log_det_jacobian(y, event_ndims=0).eval(),
          indirect_bj.inverse_log_det_jacobian(y, event_ndims=0).eval())
      self.assertAllClose(
          direct_bj.forward_log_det_jacobian(x, event_ndims=0).eval(),
          indirect_bj.forward_log_det_jacobian(x, event_ndims=0).eval())


if __name__ == "__main__":
  tf.test.main()
