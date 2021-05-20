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
import tensorflow.compat.v2 as tf

import tensorflow_probability as tfp
from tensorflow_probability.python.distributions import mvn_diag
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import test_util


@test_util.test_all_tf_execution_regimes
class HighwayFlowTests(test_util.TestCase):

  def testBijector(self):
    width = 1
    for dim in range(2):
      if dim == 0:
        # Test generic case with scalar input
        x = samplers.uniform((width,), minval=-1.,
                             maxval=1.,
                             seed=test_util.test_seed(sampler_type='stateless'))
      elif dim == 1:
        # Test with 2D tensor + batch
        x = samplers.uniform((5, width, width),
                             minval=-1.,
                             maxval=1.,
                             seed=test_util.test_seed(sampler_type='stateless'))

      bijector = tfp.experimental.bijectors.build_highway_flow_layer(
        width, activation_fn=True)
      self.evaluate(
        [v.initializer for v in bijector.trainable_variables])
      self.assertStartsWith(bijector.name, 'highway_flow')
      self.assertAllClose(x, bijector.inverse(
        tf.identity(bijector.forward(x))))
      self.assertAllClose(
        bijector.forward_log_det_jacobian(x, event_ndims=dim + 1),
        -bijector.inverse_log_det_jacobian(
          tf.identity(bijector.forward(x)), event_ndims=dim + 1))

  def testBijectorIsDeterministicGivenSeed(self):
    width = 2

    x = samplers.uniform((width,), minval=-1.,
                         maxval=1.,
                         seed=test_util.test_seed(sampler_type='stateless'))

    bijector1 = tfp.experimental.bijectors.build_highway_flow_layer(
      width, activation_fn=True, seed=test_util.test_seed(sampler_type='stateless'))
    bijector2 = tfp.experimental.bijectors.build_highway_flow_layer(
      width, activation_fn=True, seed=test_util.test_seed(sampler_type='stateless'))
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
                         seed=test_util.test_seed(sampler_type='stateless'))

    bijector = tfp.experimental.bijectors.build_highway_flow_layer(
      width, activation_fn=False)
    self.evaluate(
      [v.initializer for v in bijector.trainable_variables])
    self.assertStartsWith(bijector.name, 'highway_flow')
    self.assertAllClose(x, bijector.inverse(
      tf.identity(bijector.forward(x))))
    self.assertAllClose(
      bijector.forward_log_det_jacobian(x, event_ndims=2),
      -bijector.inverse_log_det_jacobian(
        tf.identity(bijector.forward(x)), event_ndims=2))

  def testGating(self):
    width = 4
    x = samplers.uniform((2, width, width),
                         minval=-1.,
                         maxval=1.,
                         seed=test_util.test_seed(sampler_type='stateless'))

    # Test with gating half of the inputs
    bijector = tfp.experimental.bijectors.build_highway_flow_layer(
      width, activation_fn=True, gate_first_n=2)
    self.evaluate(
      [v.initializer for v in bijector.trainable_variables])
    self.assertStartsWith(bijector.name, 'highway_flow')
    self.assertAllClose(x, bijector.inverse(
      tf.identity(bijector.forward(x))))
    self.assertAllClose(
      bijector.forward_log_det_jacobian(x, event_ndims=2),
      -bijector.inverse_log_det_jacobian(
        tf.identity(bijector.forward(x)), event_ndims=2))

    # Test with gating no inputs
    bijector = tfp.experimental.bijectors.build_highway_flow_layer(
      width, activation_fn=True, gate_first_n=0)
    self.evaluate(
      [v.initializer for v in bijector.trainable_variables])
    self.assertStartsWith(bijector.name, 'highway_flow')
    self.assertAllClose(x, bijector.inverse(
      tf.identity(bijector.forward(x))))
    self.assertAllClose(
      bijector.forward_log_det_jacobian(x, event_ndims=2),
      -bijector.inverse_log_det_jacobian(
        tf.identity(bijector.forward(x)), event_ndims=2))

  def testResidualFractionGradientsWithCenteredDifference(self):
    width = 4
    batch_size = 3
    residual_fraction = tf.constant(0.5)
    bijector = tfp.experimental.bijectors.HighwayFlow(
      residual_fraction=residual_fraction,
      activation_fn=tf.nn.softplus,
      bias=tf.zeros(width),
      upper_diagonal_weights_matrix=tf.eye(width),
      lower_diagonal_weights_matrix=tf.eye(width),
      gate_first_n=width
    )
    target = mvn_diag.MultivariateNormalDiag(loc=tf.zeros(width),
                                             scale_diag=tf.ones(width))
    x = tf.ones((batch_size, width))
    with tf.GradientTape() as g:
      g.watch(bijector.residual_fraction)
      y = tf.reduce_mean(target.log_prob(bijector.forward(x)))
    tf_grad = g.gradient(y, bijector.residual_fraction)

    h = 1e-3

    # pylint: disable=protected-access
    bijector._residual_fraction = residual_fraction + h
    y1 = tf.reduce_mean(target.log_prob(bijector.forward(tf.identity(x))))
    bijector._residual_fraction = residual_fraction - h
    y2 = tf.reduce_mean(target.log_prob(bijector.forward(tf.identity(x))))
    # pylint: enable=protected-access

    manual_grad = (y1 - y2) / (2 * h)

    self.assertAllClose(tf_grad, manual_grad, rtol=1e-4)


if __name__ == '__main__':
  tf.test.main()
