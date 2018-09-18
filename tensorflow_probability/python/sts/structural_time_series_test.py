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
"""Tests for tensorflow_probability.python.sts.structural_time_series."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.sts import LocalLinearTrend
from tensorflow_probability.python.sts import Sum

from tensorflow.python.framework import test_util
from tensorflow.python.platform import test

tfd = tfp.distributions


class _StsTestHarness(object):

  def setUp(self):
    np.random.seed(142)

  @test_util.run_in_graph_and_eager_modes
  def test_state_space_model(self):
    model = self._build_sts()

    dummy_param_vals = [p.prior.sample() for p in model.parameters]
    initial_state_prior = tfd.MultivariateNormalDiag(
        loc=-2. + tf.zeros([model.latent_size]),
        scale_diag=3. * tf.ones([model.latent_size]))

    # Verify we build the LGSSM without errors.
    ssm = model.make_state_space_model(
        num_timesteps=10,
        param_vals=dummy_param_vals,
        initial_state_prior=initial_state_prior,
        initial_step=1)

    # Verify that the child class passes the initial step and prior arguments
    # through to the SSM.
    self.assertEqual(ssm.initial_step, 1)
    self.assertEqual(ssm.initial_state_prior, initial_state_prior)

    # Verify the model has the correct latent size.
    self.assertEqual(ssm.latent_size, model.latent_size)

  @test_util.run_in_graph_and_eager_modes
  def test_log_joint(self):
    model = self._build_sts()

    num_timesteps = 5

    # simple case: single observation, and all params unbatched
    log_joint_fn = model.joint_log_prob(
        observed_time_series=np.float32(
            np.random.standard_normal([num_timesteps, 1])))
    lp = self.evaluate(
        log_joint_fn(*[p.prior.sample() for p in model.parameters]))
    self.assertEqual(tf.TensorShape([]), lp.shape)

    # more complex case: y has sample and batch shapes, some parameters
    # have partial batch shape.
    full_batch_shape = [2, 3]
    partial_batch_shape = [3]
    sample_shape = [4]
    log_joint_fn = model.joint_log_prob(
        observed_time_series=np.float32(
            np.random.standard_normal(sample_shape + full_batch_shape +
                                      [num_timesteps, 1])))

    lp = self.evaluate(
        log_joint_fn(*[
            p.prior.sample(
                sample_shape=full_batch_shape if (
                    i % 2 == 1) else partial_batch_shape)
            for (i, p) in enumerate(model.parameters)
        ]))
    self.assertEqual(tf.TensorShape(full_batch_shape), lp.shape)

  @test_util.run_in_graph_and_eager_modes
  def test_prior_sample(self):
    model = self._build_sts()
    ys, param_samples = model.prior_sample(
        num_timesteps=5, params_sample_shape=[2], trajectories_sample_shape=[3])

    self.assertAllEqual(ys.shape, [3, 2, 5, 1])
    for sampled, param in zip(param_samples, model.parameters):
      self.assertAllEqual(sampled.shape, [
          2,
      ] + param.prior.batch_shape.as_list() + param.prior.event_shape.as_list())

  @test_util.run_in_graph_and_eager_modes
  def test_default_priors_follow_batch_shapes(self):
    num_timesteps = 3
    time_series_sample_shape = [4, 2]
    observation_shape_full = time_series_sample_shape + [num_timesteps]
    dummy_observation = np.random.randn(
        *(observation_shape_full)).astype(np.float32)

    model = self._build_sts(observed_time_series=dummy_observation)

    # The model should construct a default parameter prior for *each* observed
    # time series, so the priors will have batch_shape equal to
    # `time_series_sample_shape`.
    for parameter in model.parameters:
      self.assertEqual(parameter.prior.batch_shape, time_series_sample_shape)

    # The initial state prior should also have the appropriate batch shape.
    # To test this, we build the ssm and test that it has a consistent
    # broadcast batch shape.
    param_samples = [p.prior.sample() for p in model.parameters]
    ssm = model.make_state_space_model(
        num_timesteps=num_timesteps, param_vals=param_samples)
    self.assertEqual(ssm.batch_shape, time_series_sample_shape)


class LocalLinearTrendTest(test.TestCase, _StsTestHarness):

  def _build_sts(self, observed_time_series=None):
    return LocalLinearTrend(observed_time_series=observed_time_series)


class SumTest(test.TestCase, _StsTestHarness):

  def _build_sts(self, observed_time_series=None):
    first_component = LocalLinearTrend(
        observed_time_series=observed_time_series, name='first_component')
    second_component = LocalLinearTrend(
        observed_time_series=observed_time_series, name='second_component')
    return Sum(
        components=[first_component, second_component],
        observed_time_series=observed_time_series)


if __name__ == '__main__':
  test.main()
