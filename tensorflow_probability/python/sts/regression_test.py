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
"""Regression model tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.sts import LinearRegression
from tensorflow_probability.python.sts import Sum

from tensorflow.python.framework import test_util
from tensorflow.python.ops.linalg import linear_operator_util
from tensorflow.python.platform import test

tfl = tf.linalg
tfd = tfp.distributions


class _LinearRegressionTest(object):

  @test_util.run_in_graph_and_eager_modes
  def test_basic_statistics(self):
    # Verify that this model constructs a distribution with mean
    # `matmul(design_matrix, weights)` and stddev 0.
    batch_shape = [4, 3]
    num_timesteps = 10
    num_features = 2
    design_matrix = self._build_placeholder(
        np.random.randn(*(batch_shape + [num_timesteps, num_features])))

    linear_regression = LinearRegression(design_matrix=design_matrix)
    true_weights = self._build_placeholder(
        np.random.randn(*(batch_shape + [num_features])))
    predicted_time_series = linear_operator_util.matmul_with_broadcast(
        design_matrix, true_weights[..., tf.newaxis])

    ssm = linear_regression.make_state_space_model(
        num_timesteps=num_timesteps,
        param_vals={"weights": true_weights})
    self.assertAllEqual(self.evaluate(ssm.mean()), predicted_time_series)
    self.assertAllEqual(*self.evaluate((ssm.stddev(),
                                        tf.zeros_like(predicted_time_series))))

  def test_simple_regression_correctness(self):
    # Verify that optimizing a simple linear regression by gradient descent
    # recovers the known-correct weights.
    batch_shape = [4, 3]
    num_timesteps = 10
    num_features = 2
    design_matrix = self._build_placeholder(
        np.random.randn(*(batch_shape + [num_timesteps, num_features])))

    true_weights = self._build_placeholder([4., -3.])
    predicted_time_series = linear_operator_util.matmul_with_broadcast(
        design_matrix, true_weights[..., tf.newaxis])

    linear_regression = LinearRegression(
        design_matrix=design_matrix,
        weights_prior=tfd.Independent(
            tfd.Cauchy(loc=self._build_placeholder(np.zeros([num_features])),
                       scale=self._build_placeholder(np.ones([num_features]))),
            reinterpreted_batch_ndims=1))
    observation_noise_scale_prior = tfd.LogNormal(
        loc=self._build_placeholder(-2), scale=self._build_placeholder(0.1))
    model = Sum(components=[linear_regression],
                observation_noise_scale_prior=observation_noise_scale_prior)

    learnable_weights = tf.compat.v2.Variable(
        tf.zeros([num_features], dtype=true_weights.dtype))

    def build_loss():
      learnable_ssm = model.make_state_space_model(
          num_timesteps=num_timesteps,
          param_vals={
              "LinearRegression/_weights": learnable_weights,
              "observation_noise_scale": observation_noise_scale_prior.mode()})
      return -learnable_ssm.log_prob(predicted_time_series)

    # We provide graph- and eager-mode optimization for TF 2.0 compatibility.
    num_train_steps = 80
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.1)
    if tf.executing_eagerly():
      for _ in range(num_train_steps):
        optimizer.minimize(build_loss)
    else:
      train_op = optimizer.minimize(build_loss())
      self.evaluate(tf.compat.v1.global_variables_initializer())
      for _ in range(num_train_steps):
        _ = self.evaluate(train_op)
    self.assertAllClose(*self.evaluate((true_weights, learnable_weights)),
                        atol=0.2)

  @test_util.run_in_graph_and_eager_modes
  def test_scalar_priors_broadcast(self):

    batch_shape = [4, 3]
    num_timesteps = 10
    num_features = 2
    design_matrix = self._build_placeholder(
        np.random.randn(*(batch_shape + [num_timesteps, num_features])))

    # Build a model with scalar Normal(0., 1.) prior.
    linear_regression = LinearRegression(
        design_matrix=design_matrix,
        weights_prior=tfd.Normal(loc=self._build_placeholder(0.),
                                 scale=self._build_placeholder(1.)))

    weights_prior = linear_regression.parameters[0].prior
    self.assertAllEqual([num_features],
                        self.evaluate(weights_prior.event_shape_tensor()))
    self.assertAllEqual(batch_shape,
                        self.evaluate(weights_prior.batch_shape_tensor()))

    prior_sampled_weights = weights_prior.sample()
    ssm = linear_regression.make_state_space_model(
        num_timesteps=num_timesteps,
        param_vals={"weights": prior_sampled_weights})

    lp = ssm.log_prob(ssm.sample())
    self.assertAllEqual(batch_shape,
                        self.evaluate(lp).shape)

  def _build_placeholder(self, ndarray):
    """Convert a numpy array to a TF placeholder.

    Args:
      ndarray: any object convertible to a numpy array via `np.asarray()`.

    Returns:
      placeholder: a TensorFlow `placeholder` with default value given by the
      provided `ndarray`, dtype given by `self.dtype`, and shape specified
      statically only if `self.use_static_shape` is `True`.
    """

    ndarray = np.asarray(ndarray).astype(self.dtype)
    return tf.compat.v1.placeholder_with_default(
        input=ndarray, shape=ndarray.shape if self.use_static_shape else None)


class LinearRegressionTestStaticShape32(
    tf.test.TestCase, _LinearRegressionTest):
  dtype = np.float32
  use_static_shape = True


class LinearRegressionTestDynamicShape32(
    tf.test.TestCase, _LinearRegressionTest):
  dtype = np.float32
  use_static_shape = False


class LinearRegressionTestStaticShape64(
    tf.test.TestCase, _LinearRegressionTest):
  dtype = np.float64
  use_static_shape = True


if __name__ == "__main__":
  test.main()
