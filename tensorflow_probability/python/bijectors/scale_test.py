# Copyright 2019 The TensorFlow Probability Authors.
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
"""Scalar Tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# Dependency imports
from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python.bijectors import bijector_test_util
from tensorflow_probability.python.internal import test_util


@test_util.test_all_tf_execution_regimes
class ScaleBijectorTest(test_util.TestCase, parameterized.TestCase):
  """Tests correctness of the Y = scale @ x transformation."""

  def testName(self):
    # scale corresponds to 1.
    bijector = tfb.Scale(scale=-1.)
    self.assertStartsWith(bijector.name, 'scale')

  @parameterized.named_parameters(
      dict(testcase_name='static_float32', is_static=True, dtype=np.float32),
      dict(testcase_name='static_float64', is_static=True, dtype=np.float64),
      dict(testcase_name='dynamic_float32', is_static=False, dtype=np.float32),
      dict(testcase_name='dynamic_float64', is_static=False, dtype=np.float64),
  )
  def testNoBatchScale(self, is_static, dtype):
    bijector = tfb.Scale(scale=dtype(2.))
    x = self.maybe_static(np.array([1., 2, 3], dtype), is_static)
    self.assertAllClose([2., 4, 6], bijector.forward(x))
    self.assertAllClose([.5, 1, 1.5], bijector.inverse(x))
    self.assertAllClose(
        -np.log(2.),
        bijector.inverse_log_det_jacobian(x, event_ndims=0))

  @parameterized.named_parameters(
      dict(testcase_name='static_float32', is_static=True, dtype=np.float32),
      dict(testcase_name='static_float64', is_static=True, dtype=np.float64),
      dict(testcase_name='dynamic_float32', is_static=False, dtype=np.float32),
      dict(testcase_name='dynamic_float64', is_static=False, dtype=np.float64),
  )
  def testBatchScale(self, is_static, dtype):
    # Batched scale
    bijector = tfb.Scale(scale=dtype([2., 3.]))
    x = self.maybe_static(np.array([1.], dtype=dtype), is_static)
    self.assertAllClose([2., 3.], bijector.forward(x))
    self.assertAllClose([0.5, 1./3.], bijector.inverse(x))
    self.assertAllClose(
        [-np.log(2.), -np.log(3.)],
        bijector.inverse_log_det_jacobian(x, event_ndims=0))

  @parameterized.named_parameters(
      dict(testcase_name='float32', dtype=np.float32),
      dict(testcase_name='float64', dtype=np.float64),
  )
  def testScalarCongruency(self, dtype):
    bijector = tfb.Scale(scale=dtype(0.42))
    bijector_test_util.assert_scalar_congruency(
        bijector,
        lower_x=dtype(-2.),
        upper_x=dtype(2.),
        eval_func=self.evaluate)

  @test_util.jax_disable_variable_test
  def testVariableGradients(self):
    b = tfb.Scale(scale=tf.Variable(2.))

    with tf.GradientTape() as tape:
      y = b.forward(.1)
    self.assertAllNotNone(tape.gradient(y, b.trainable_variables))

  def testImmutableScaleAssertion(self):
    with self.assertRaisesOpError('Argument `scale` must be non-zero'):
      b = tfb.Scale(scale=0., validate_args=True)
      _ = self.evaluate(b.forward(1.))

  def testVariableScaleAssertion(self):
    v = tf.Variable(0.)
    self.evaluate(v.initializer)
    with self.assertRaisesOpError('Argument `scale` must be non-zero'):
      b = tfb.Scale(scale=v, validate_args=True)
      _ = self.evaluate(b.forward(1.))

  def testModifiedVariableScaleAssertion(self):
    v = tf.Variable(1.)
    self.evaluate(v.initializer)
    b = tfb.Scale(scale=v, validate_args=True)
    with self.assertRaisesOpError('Argument `scale` must be non-zero'):
      with tf.control_dependencies([v.assign(0.)]):
        _ = self.evaluate(b.forward(1.))


if __name__ == '__main__':
  tf.test.main()
