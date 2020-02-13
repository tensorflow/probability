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
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf

from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python.bijectors import bijector_test_util
from tensorflow_probability.python.internal import test_util


@test_util.test_all_tf_execution_regimes
class _AffineScalarBijectorTest(object):
  """Tests correctness of the Y = scale @ x + shift transformation."""

  def testProperties(self):
    # scale corresponds to 1.
    bijector = tfb.AffineScalar(shift=-1.)
    self.assertStartsWith(bijector.name, "affine_scalar")

  def testTinyScale(self):
    log_scale = tf.cast(-2000., self.dtype)
    x = tf.cast(1., self.dtype)
    scale = tf.exp(log_scale)
    fldj_linear = tfb.AffineScalar(scale=scale).forward_log_det_jacobian(
        x, event_ndims=0)
    fldj_log = tfb.AffineScalar(log_scale=log_scale).forward_log_det_jacobian(
        x, event_ndims=0)
    fldj_linear_, fldj_log_ = self.evaluate([fldj_linear, fldj_log])
    # Using the linear scale will saturate to 0, and produce bad log-det
    # Jacobians.
    self.assertNotEqual(fldj_linear_, fldj_log_)
    self.assertAllClose(-2000., fldj_log_)

  def testNoBatchScalar(self):
    def static_run(fun, x, **kwargs):
      return self.evaluate(fun(x, **kwargs))

    def dynamic_run(fun, x_value, **kwargs):
      x_value = np.array(x_value, dtype=self.dtype)
      x = tf1.placeholder_with_default(x_value, shape=None)
      return self.evaluate(fun(x, **kwargs))

    for run in (static_run, dynamic_run):
      bijector = tfb.AffineScalar(shift=self.dtype(-1.), scale=self.dtype(2.))
      x = self.dtype([1., 2, 3])  # Three scalar samples (no batches).
      self.assertAllClose([1., 3, 5], run(bijector.forward, x))
      self.assertAllClose([1., 1.5, 2.], run(bijector.inverse, x))
      self.assertAllClose(
          -np.log(2.),
          run(bijector.inverse_log_det_jacobian, x, event_ndims=0))

  def testOneBatchScalarViaIdentityUserProvidesShiftOnly(self):
    def static_run(fun, x, **kwargs):
      return self.evaluate(fun(x, **kwargs))

    def dynamic_run(fun, x_value, **kwargs):
      x_value = np.array(x_value, dtype=self.dtype)
      x = tf1.placeholder_with_default(x_value, shape=None)
      return self.evaluate(fun(x, **kwargs))

    for run in (static_run, dynamic_run):
      # Batched shift
      bijector = tfb.AffineScalar(shift=self.dtype([1.]))
      x = self.dtype([1.])  # One sample from one batches.
      self.assertAllClose([2.], run(bijector.forward, x))
      self.assertAllClose([0.], run(bijector.inverse, x))
      self.assertAllClose(
          0.,
          run(bijector.inverse_log_det_jacobian, x, event_ndims=0))

  def testOneBatchScalarViaIdentityUserProvidesScaleOnly(self):
    def static_run(fun, x, **kwargs):
      return self.evaluate(fun(x, **kwargs))

    def dynamic_run(fun, x_value, **kwargs):
      x_value = np.array(x_value)
      x = tf1.placeholder_with_default(x_value, shape=None)
      return self.evaluate(fun(x, **kwargs))

    for run in (static_run, dynamic_run):
      # Batched scale
      bijector = tfb.AffineScalar(scale=self.dtype([2.]))
      x = self.dtype([1.])  # One sample from one batches.
      self.assertAllClose([2.], run(bijector.forward, x))
      self.assertAllClose([0.5], run(bijector.inverse, x))
      self.assertAllClose(
          [np.log(0.5)],
          run(bijector.inverse_log_det_jacobian, x, event_ndims=0))

  def testTwoBatchScalarIdentityViaIdentity(self):
    def static_run(fun, x, **kwargs):
      return self.evaluate(fun(x, **kwargs))

    def dynamic_run(fun, x_value, **kwargs):
      x_value = np.array(x_value, dtype=self.dtype)
      x = tf1.placeholder_with_default(x_value, shape=None)
      return self.evaluate(fun(x, **kwargs))

    for run in (static_run, dynamic_run):
      # Batch of 2 shifts
      bijector = tfb.AffineScalar(shift=self.dtype([1., -1]))
      x = self.dtype([1., 1])  # One sample from each of two batches.
      self.assertAllClose([2., 0], run(bijector.forward, x))
      self.assertAllClose([0., 2], run(bijector.inverse, x))
      self.assertAllClose(
          0.,
          run(bijector.inverse_log_det_jacobian, x, event_ndims=0))

  def testTwoBatchScalarIdentityViaScale(self):
    def static_run(fun, x, **kwargs):
      return self.evaluate(fun(x, **kwargs))

    def dynamic_run(fun, x_value, **kwargs):
      x_value = np.array(x_value, dtype=self.dtype)
      x = tf1.placeholder_with_default(x_value, shape=None)
      return self.evaluate(fun(x, **kwargs))

    for run in (static_run, dynamic_run):
      # Batch of 2 scales and 2 shifts
      bijector = tfb.AffineScalar(
          shift=self.dtype([1., -1]),
          scale=self.dtype([2., 1]))
      x = self.dtype([1., 1])  # One sample from each of two batches.
      self.assertAllClose([3., 0], run(bijector.forward, x))
      self.assertAllClose([0., 2], run(bijector.inverse, x))
      self.assertAllClose(
          [-np.log(2), 0.],
          run(bijector.inverse_log_det_jacobian, x, event_ndims=0))

  def testScalarCongruency(self):
    bijector = tfb.AffineScalar(shift=self.dtype(3.6), scale=self.dtype(0.42))
    bijector_test_util.assert_scalar_congruency(
        bijector,
        lower_x=self.dtype(-2.),
        upper_x=self.dtype(2.),
        eval_func=self.evaluate)

  def testScalarCongruencyLogScale(self):
    bijector = tfb.AffineScalar(
        shift=self.dtype(3.6), log_scale=self.dtype(np.log(0.42)))
    bijector_test_util.assert_scalar_congruency(
        bijector,
        lower_x=self.dtype(-2.),
        upper_x=self.dtype(2.),
        eval_func=self.evaluate)

  @test_util.jax_disable_variable_test
  def testVariableGradients(self):
    b = tfb.AffineScalar(
        shift=tf.Variable(1.),
        scale=tf.Variable(2.))

    with tf.GradientTape() as tape:
      y = b.forward(.1)
    self.assertAllNotNone(tape.gradient(y, [b.shift, b.scale]))

  def testImmutableScaleAssertion(self):
    with self.assertRaisesOpError("Argument `scale` must be non-zero"):
      b = tfb.AffineScalar(scale=0., validate_args=True)
      _ = self.evaluate(b.forward(1.))

  def testVariableScaleAssertion(self):
    v = tf.Variable(0.)
    self.evaluate(v.initializer)
    with self.assertRaisesOpError("Argument `scale` must be non-zero"):
      b = tfb.AffineScalar(scale=v, validate_args=True)
      _ = self.evaluate(b.forward(1.))

  def testModifiedVariableScaleAssertion(self):
    v = tf.Variable(1.)
    self.evaluate(v.initializer)
    b = tfb.AffineScalar(scale=v, validate_args=True)
    with self.assertRaisesOpError("Argument `scale` must be non-zero"):
      with tf.control_dependencies([v.assign(0.)]):
        _ = self.evaluate(b.forward(1.))


class AffineScalarBijectorTestFloat32(test_util.TestCase,
                                      _AffineScalarBijectorTest):
  dtype = np.float32


class AffineScalarBijectorTestFloat64(test_util.TestCase,
                                      _AffineScalarBijectorTest):
  dtype = np.float64


if __name__ == "__main__":
  tf.test.main()
