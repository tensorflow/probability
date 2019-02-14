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
"""Affine Scalar Tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.bijectors import bijector_test_util

from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top

tfb = tfp.bijectors


@test_util.run_all_in_graph_and_eager_modes
class AffineScalarBijectorTest(tf.test.TestCase):
  """Tests correctness of the Y = scale @ x + shift transformation."""

  def testProperties(self):
    mu = -1.
    # scale corresponds to 1.
    bijector = tfb.AffineScalar(shift=mu)
    self.assertEqual("affine_scalar", bijector.name)

  def testNoBatchScalar(self):
    def static_run(fun, x, **kwargs):
      return self.evaluate(fun(x, **kwargs))

    def dynamic_run(fun, x_value, **kwargs):
      x_value = np.array(x_value, dtype=np.float32)
      x = tf.compat.v1.placeholder_with_default(x_value, shape=None)
      return self.evaluate(fun(x, **kwargs))

    for run in (static_run, dynamic_run):
      mu = -1.
      # Corresponds to scale = 2
      bijector = tfb.AffineScalar(shift=mu, scale=2.)
      x = [1., 2, 3]  # Three scalar samples (no batches).
      self.assertAllClose([1., 3, 5], run(bijector.forward, x))
      self.assertAllClose([1., 1.5, 2.], run(bijector.inverse, x))
      self.assertAllClose(
          -np.log(2.),
          run(bijector.inverse_log_det_jacobian, x, event_ndims=0))

  def testOneBatchScalarViaIdentityIn64BitUserProvidesShiftOnly(self):
    def static_run(fun, x, **kwargs):
      return self.evaluate(fun(x, **kwargs))

    def dynamic_run(fun, x_value, **kwargs):
      x_value = np.array(x_value, dtype=np.float64)
      x = tf.compat.v1.placeholder_with_default(x_value, shape=None)
      return self.evaluate(fun(x, **kwargs))

    for run in (static_run, dynamic_run):
      mu = np.float64([1.])
      # One batch, scalar.
      # Corresponds to scale = 1.
      bijector = tfb.AffineScalar(shift=mu)
      x = np.float64([1.])  # One sample from one batches.
      self.assertAllClose([2.], run(bijector.forward, x))
      self.assertAllClose([0.], run(bijector.inverse, x))
      self.assertAllClose(
          0.,
          run(bijector.inverse_log_det_jacobian, x, event_ndims=0))

  def testOneBatchScalarViaIdentityIn64BitUserProvidesScaleOnly(self):
    def static_run(fun, x, **kwargs):
      return self.evaluate(fun(x, **kwargs))

    def dynamic_run(fun, x_value, **kwargs):
      x_value = np.array(x_value)
      x = tf.compat.v1.placeholder_with_default(x_value, shape=None)
      return self.evaluate(fun(x, **kwargs))

    for run in (static_run, dynamic_run):
      multiplier = np.float64([2.])
      # One batch, scalar.
      # Corresponds to scale = 2, shift = 0.
      bijector = tfb.AffineScalar(scale=multiplier)
      x = np.float64([1.])  # One sample from one batches.
      self.assertAllClose([2.], run(bijector.forward, x))
      self.assertAllClose([0.5], run(bijector.inverse, x))
      self.assertAllClose(
          [np.log(0.5)],
          run(bijector.inverse_log_det_jacobian, x, event_ndims=0))

  def testTwoBatchScalarIdentityViaIdentity(self):
    def static_run(fun, x, **kwargs):
      return self.evaluate(fun(x, **kwargs))

    def dynamic_run(fun, x_value, **kwargs):
      x_value = np.array(x_value, dtype=np.float32)
      x = tf.compat.v1.placeholder_with_default(x_value, shape=None)
      return self.evaluate(fun(x, **kwargs))

    for run in (static_run, dynamic_run):
      mu = [1., -1]
      # Univariate, two batches.
      # Corresponds to scale = 1.
      bijector = tfb.AffineScalar(shift=mu)
      x = [1., 1]  # One sample from each of two batches.
      self.assertAllClose([2., 0], run(bijector.forward, x))
      self.assertAllClose([0., 2], run(bijector.inverse, x))
      self.assertAllClose(
          0.,
          run(bijector.inverse_log_det_jacobian, x, event_ndims=0))

  def testTwoBatchScalarIdentityViaScale(self):
    def static_run(fun, x, **kwargs):
      return self.evaluate(fun(x, **kwargs))

    def dynamic_run(fun, x_value, **kwargs):
      x_value = np.array(x_value, dtype=np.float32)
      x = tf.compat.v1.placeholder_with_default(x_value, shape=None)
      return self.evaluate(fun(x, **kwargs))

    for run in (static_run, dynamic_run):
      mu = [1., -1]
      # Univariate, two batches.
      # Corresponds to scale = 1.
      bijector = tfb.AffineScalar(shift=mu, scale=[2., 1])
      x = [1., 1]  # One sample from each of two batches.
      self.assertAllClose([3., 0], run(bijector.forward, x))
      self.assertAllClose([0., 2], run(bijector.inverse, x))
      self.assertAllClose(
          [-np.log(2), 0.],
          run(bijector.inverse_log_det_jacobian, x, event_ndims=0))

  def testScalarCongruency(self):
    bijector = tfb.AffineScalar(shift=3.6, scale=0.42)
    bijector_test_util.assert_scalar_congruency(
        bijector, lower_x=-2., upper_x=2., eval_func=self.evaluate)


if __name__ == "__main__":
  tf.test.main()
