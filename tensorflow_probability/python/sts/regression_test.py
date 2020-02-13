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
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.sts import LinearRegression
from tensorflow_probability.python.sts import SparseLinearRegression
from tensorflow_probability.python.sts import Sum

from tensorflow.python.platform import test

tfl = tf.linalg


@test_util.test_all_tf_execution_regimes
class _LinearRegressionTest(test_util.TestCase):

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
    predicted_time_series = tf.linalg.matmul(
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
    predicted_time_series = tf.linalg.matmul(
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

    learnable_weights = tf.Variable(
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
    optimizer = tf1.train.AdamOptimizer(learning_rate=0.1)
    if tf.executing_eagerly():
      for _ in range(num_train_steps):
        optimizer.minimize(build_loss)
    else:
      train_op = optimizer.minimize(build_loss())
      self.evaluate(tf1.global_variables_initializer())
      for _ in range(num_train_steps):
        _ = self.evaluate(train_op)
    self.assertAllClose(*self.evaluate((true_weights, learnable_weights)),
                        atol=0.2)

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
    return tf1.placeholder_with_default(
        ndarray, shape=ndarray.shape if self.use_static_shape else None)


@test_util.test_all_tf_execution_regimes
class _SparseLinearRegressionTest(test_util.TestCase):

  def test_builds_without_errors(self):
    batch_shape = [4, 3]
    num_timesteps = 10
    num_features = 2
    design_matrix = self._build_placeholder(
        np.random.randn(*(batch_shape + [num_timesteps, num_features])))

    weights_batch_shape = []
    if not self.use_static_shape:
      weights_batch_shape = tf1.placeholder_with_default(
          np.array(weights_batch_shape, dtype=np.int32), shape=None)
    sparse_regression = SparseLinearRegression(
        design_matrix=design_matrix,
        weights_batch_shape=weights_batch_shape)
    prior_params = [param.prior.sample()
                    for param in sparse_regression.parameters]

    ssm = sparse_regression.make_state_space_model(
        num_timesteps=num_timesteps,
        param_vals=prior_params)
    if self.use_static_shape:
      output_shape = ssm.sample().shape.as_list()
    else:
      output_shape = self.evaluate(tf.shape(ssm.sample()))
    self.assertAllEqual(output_shape, batch_shape + [num_timesteps, 1])

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
    return tf1.placeholder_with_default(
        ndarray, shape=ndarray.shape if self.use_static_shape else None)


class LinearRegressionTestStaticShape64(_LinearRegressionTest):
  dtype = np.float64
  use_static_shape = True


class LinearRegressionTestDynamicShape32(_LinearRegressionTest):
  dtype = np.float32
  use_static_shape = False


class SparseLinearRegressionTestStaticShape64(_SparseLinearRegressionTest):
  dtype = np.float64
  use_static_shape = True


class SparseLinearRegressionTestDynamicShape32(_SparseLinearRegressionTest):
  dtype = np.float32
  use_static_shape = False

del _LinearRegressionTest  # Don't try to run base class tests.
del _SparseLinearRegressionTest  # Don't try to run base class tests.

if __name__ == "__main__":
  test.main()
