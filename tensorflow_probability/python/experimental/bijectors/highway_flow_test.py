# Copyright 2021 The TensorFlow Probability Authors.
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
"""Tests for HighwayFlow."""

from absl.testing import parameterized

import tensorflow.compat.v2 as tf

import tensorflow_probability as tfp
from tensorflow_probability.python.bijectors import bijector_test_util
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import test_util

tfb = tfp.bijectors
tfd = tfp.distributions

@test_util.test_all_tf_execution_regimes
class HighwayFlowTests(test_util.TestCase):

  '''@parameterized.named_parameters(
      ('scalar', []),
      ('batch', [5, 2]))
  def testBijector(self, sample_shape):
    width = 1
    # Test with 2D tensor + batch
    x = samplers.uniform(sample_shape + [width],
                         minval=-1.,
                         maxval=1.,
                         seed=(0,0))

    bijector = tfp.experimental.bijectors.build_trainable_highway_flow(
        width,
        activation_fn=tf.nn.softplus,
        seed=(0,0))
    self.evaluate([v.initializer for v in bijector.trainable_variables])
    self.assertStartsWith(bijector.name, 'highway_flow')
    self.assertAllClose(x, bijector.inverse(tf.identity(bijector.forward(x))))
    self.assertAllClose(
        bijector.forward_log_det_jacobian(x, event_ndims=1),
        -bijector.inverse_log_det_jacobian(
            tf.identity(bijector.forward(x)), event_ndims=1))

  def testBijectorIsDeterministicGivenSeed(self):
    width = 2

    x = samplers.uniform((width,), minval=-1.,
                         maxval=1.,
                         seed=(0,0))

    bijector1 = tfp.experimental.bijectors.build_highway_flow_layer(
      width, activation_fn=True, seed=(0,0))
    bijector2 = tfp.experimental.bijectors.build_highway_flow_layer(
      width, activation_fn=True, seed=(0,0))
    self.evaluate(
      [v.initializer for v in bijector1.trainable_variables])
    self.evaluate(
      [v.initializer for v in bijector2.trainable_variables])
    self.assertAllClose(bijector1.forward(x), bijector2.forward(x))

  def testBijectorWithoutActivation(self):
    width = 4
    x = samplers.uniform((2, width, width),
                         minval=-1.,
                         maxval=1.,
                         seed=(0,0))

    bijector = tfp.experimental.bijectors.build_trainable_highway_flow(
        width, activation_fn=False, seed=(0,0))
    self.evaluate([v.initializer for v in bijector.trainable_variables])
    self.assertStartsWith(bijector.name, 'highway_flow')
    self.assertAllClose(x, bijector.inverse(tf.identity(bijector.forward(x))))
    self.assertAllClose(
        bijector.forward_log_det_jacobian(x, event_ndims=2),
        -bijector.inverse_log_det_jacobian(
            tf.identity(bijector.forward(x)), event_ndims=2))'''

  def testGating(self):
    '''width = 4
    x = samplers.uniform((2, 2, width),
                         minval=-1.,
                         maxval=1.,
                         seed=(0,0))

    # Test with gating half of the inputs
    bijector = tfp.experimental.bijectors.build_trainable_highway_flow(
        width,
        activation_fn=tf.nn.softplus,
        gate_first_n=2,
        seed=(0,0))
    self.evaluate([v.initializer for v in bijector.trainable_variables])
    self.assertStartsWith(bijector.name, 'highway_flow')
    self.assertAllClose(x, bijector.inverse(tf.identity(bijector.forward(x))))
    self.assertAllClose(
        bijector.forward_log_det_jacobian(x, event_ndims=2),
        -bijector.inverse_log_det_jacobian(
            tf.identity(bijector.forward(x)), event_ndims=2))

    # Test with gating no inputs
    bijector = tfp.experimental.bijectors.build_trainable_highway_flow(
        width,
        activation_fn=tf.nn.softplus,
        gate_first_n=0,
        seed=(0,0))
    self.evaluate([v.initializer for v in bijector.trainable_variables])
    self.assertStartsWith(bijector.name, 'highway_flow')
    self.assertAllClose(x, bijector.inverse(tf.identity(bijector.forward(x))))
    self.assertAllClose(
        bijector.forward_log_det_jacobian(x, event_ndims=2),
        -bijector.inverse_log_det_jacobian(
            tf.identity(bijector.forward(x)), event_ndims=2))'''

    bijector = tfp.experimental.bijectors.HighwayFlow(
      residual_fraction=0.5,
      activation_fn=None,
      bias=tf.zeros(4),
      upper_diagonal_weights_matrix=tf.experimental.numpy.tril(tf.ones((4,4)), 0),
      lower_diagonal_weights_matrix=tf.experimental.numpy.tril(tf.ones((4,4)), 0),
      gate_first_n=2
    )
    x = tf.ones(4)
    self.evaluate([v.initializer for v in bijector.trainable_variables])
    self.assertStartsWith(bijector.name, 'highway_flow')
    expected_y = tf.convert_to_tensor([6.75, 6.5, 5., 3.])
    self.assertAllClose(expected_y, bijector.forward(x))
    bijector.inverse(expected_y)

  '''@test_util.numpy_disable_gradient_test
  def testResidualFractionGradientsWithCenteredDifference(self):
    width = 4
    batch_size = 3
    target = tfd.MultivariateNormalDiag(
        loc=tf.zeros(width), scale_diag=tf.ones(width))
    x = tf.ones((batch_size, width))

    def fn(residual_fraction):
      bijector = tfp.experimental.bijectors.HighwayFlow(
          residual_fraction=residual_fraction,
          activation_fn=tf.nn.softplus,
          bias=tf.zeros(width),
          upper_diagonal_weights_matrix=tf.eye(width),
          lower_diagonal_weights_matrix=tf.eye(width),
          gate_first_n=width)
      return tf.reduce_mean(target.log_prob(bijector.forward(x)))

    _, auto_grad = tfp.math.value_and_gradient(fn, 0.5)

    h = 1e-3
    y1 = fn(0.5 + h)
    y2 = fn(0.5 - h)
    manual_grad = (y1 - y2) / (2 * h)

    self.assertAllClose(auto_grad, manual_grad, rtol=2e-4)

  @test_util.numpy_disable_gradient_test
  def testTheoreticalFldj(self):
    width = 4
    bijector = tfp.experimental.bijectors.build_trainable_highway_flow(
        width,
        activation_fn=tf.nn.softplus,
        gate_first_n=2,
        seed=(0,0))
    self.evaluate([v.initializer for v in bijector.trainable_variables])

    x = self.evaluate(
        samplers.uniform([width], minval=-1., maxval=1.,
                         seed=(0,0)))
    y = self.evaluate(bijector.forward(x))
    bijector_test_util.assert_bijective_and_finite(
        bijector,
        x,
        y,
        eval_func=self.evaluate,
        event_ndims=1,
        inverse_event_ndims=1,
        rtol=1e-5)
    fldj = bijector.forward_log_det_jacobian(x, event_ndims=1)
    # The jacobian is not yet broadcast, since it is constant.
    fldj_theoretical = bijector_test_util.get_fldj_theoretical(
        bijector, x, event_ndims=1)
    self.assertAllClose(
        self.evaluate(fldj_theoretical),
        self.evaluate(fldj),
        atol=1e-5,
        rtol=1e-5)

  def testJacobianWithActivation(self):
    activations = ['softplus']
    batch_size = 3
    width = 4
    dtype = tf.float32
    residual_fraction = tf.constant(0.5)
    for activation in activations:
      if activation == 'sigmoid':
        activation_fn = tf.nn.sigmoid
      elif activation == 'softplus':
        activation_fn = tf.nn.softplus
      elif activation == 'tanh':
        activation_fn = tf.nn.tanh
      elif activation == 'none':
        activation_fn = None
      bijector = tfp.experimental.bijectors.HighwayFlow(
        residual_fraction=residual_fraction,
        activation_fn=activation_fn,
        bias=tf.zeros(width),
        upper_diagonal_weights_matrix=tf.eye(width),
        lower_diagonal_weights_matrix=tf.eye(width),
        gate_first_n=width
      )
      self.evaluate(
        [v.initializer for v in bijector.trainable_variables])
      x = tf.ones((batch_size,
                   width)) * samplers.uniform((batch_size, width), -10.,
                                              10., seed=(0,0))
      if activation == 'none':
        y = x
      else:
        y = residual_fraction * x + (
            1 - residual_fraction) * activation_fn(x)
      expected_forward_log_det_jacobian = \
        _activation_log_det_jacobian(x,
                                     residual_fraction,
                                     activation)
      expected_inverse_log_det_jacobian = \
        -expected_forward_log_det_jacobian
      self.assertAllClose(y, bijector.forward(x))
      self.assertAllClose(x, bijector.inverse(y))
      self.assertAllClose(
        expected_inverse_log_det_jacobian,
        bijector.inverse_log_det_jacobian(y, event_ndims=1),
      )
      self.assertAllClose(
        expected_forward_log_det_jacobian,
        bijector.forward_log_det_jacobian(x, event_ndims=1),
      )

  def testBijectorIsDeterministicGivenSeed(self):
    width = 2

    x = samplers.uniform((width,), minval=-1.,
                         maxval=1.,
                         seed=(0,0))

    bijector1 = tfp.experimental.bijectors.build_trainable_highway_flow(
        width,
        activation_fn=tf.nn.softplus,
        seed=(0,0))
    bijector2 = tfp.experimental.bijectors.build_trainable_highway_flow(
        width,
        activation_fn=tf.nn.softplus,
        seed=(0,0))
    self.evaluate([v.initializer for v in bijector1.trainable_variables])
    self.evaluate([v.initializer for v in bijector2.trainable_variables])
    self.assertAllClose(bijector1.forward(x), bijector2.forward(x))'''

if __name__ == '__main__':
  tf.test.main()
