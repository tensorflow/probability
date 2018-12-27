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

from tensorflow_probability.python.sts import LinearRegression
from tensorflow_probability.python.sts import LocalLinearTrend
from tensorflow_probability.python.sts import Seasonal
from tensorflow_probability.python.sts import Sum
from tensorflow_probability.python.sts.internal import util as sts_util

tfd = tfp.distributions
tfb = tfp.bijectors
tfe = tf.contrib.eager


class _StructuralTimeSeriesTests(object):

  def test_broadcast_batch_shapes(self):

    batch_shape = [3, 1, 4]
    partial_batch_shape = [2, 1]
    expected_broadcast_batch_shape = [3, 2, 4]

    # Build a model where parameters have different batch shapes.
    partial_batch_loc = self._build_placeholder(
        np.random.randn(*partial_batch_shape))
    full_batch_loc = self._build_placeholder(
        np.random.randn(*batch_shape))

    partial_scale_prior = tfd.LogNormal(
        loc=partial_batch_loc, scale=tf.ones_like(partial_batch_loc))
    full_scale_prior = tfd.LogNormal(
        loc=full_batch_loc, scale=tf.ones_like(full_batch_loc))
    loc_prior = tfd.Normal(loc=partial_batch_loc,
                           scale=tf.ones_like(partial_batch_loc))

    linear_trend = LocalLinearTrend(level_scale_prior=full_scale_prior,
                                    slope_scale_prior=full_scale_prior,
                                    initial_level_prior=loc_prior,
                                    initial_slope_prior=loc_prior)
    seasonal = Seasonal(num_seasons=3,
                        drift_scale_prior=partial_scale_prior,
                        initial_effect_prior=loc_prior)
    model = Sum([linear_trend, seasonal],
                observation_noise_scale_prior=partial_scale_prior)
    param_samples = [p.prior.sample() for p in model.parameters]
    ssm = model.make_state_space_model(num_timesteps=2,
                                       param_vals=param_samples)

    # Test that the model's batch shape matches the SSM's batch shape,
    # and that they both match the expected broadcast shape.
    self.assertAllEqual(model.batch_shape, ssm.batch_shape)

    (model_batch_shape_tensor_,
     ssm_batch_shape_tensor_) = self.evaluate((model.batch_shape_tensor(),
                                               ssm.batch_shape_tensor()))
    self.assertAllEqual(model_batch_shape_tensor_, ssm_batch_shape_tensor_)
    self.assertAllEqual(model_batch_shape_tensor_,
                        expected_broadcast_batch_shape)

  def _build_placeholder(self, ndarray, dtype=None):
    """Convert a numpy array to a TF placeholder.

    Args:
      ndarray: any object convertible to a numpy array via `np.asarray()`.
      dtype: optional `dtype`; if not specified, defaults to `self.dtype`.

    Returns:
      placeholder: a TensorFlow `placeholder` with default value given by the
      provided `ndarray`, dtype given by `self.dtype`, and shape specified
      statically only if `self.use_static_shape` is `True`.
    """

    if dtype is None:
      dtype = self.dtype

    ndarray = np.asarray(ndarray).astype(dtype)
    return tf.placeholder_with_default(
        input=ndarray, shape=ndarray.shape if self.use_static_shape else None)


@tfe.run_all_tests_in_graph_and_eager_modes
class StructuralTimeSeriesTestsStaticShape32(
    _StructuralTimeSeriesTests, tf.test.TestCase):
  dtype = np.float32
  use_static_shape = True


@tfe.run_all_tests_in_graph_and_eager_modes
class StructuralTimeSeriesTestsDynamicShape32(
    _StructuralTimeSeriesTests, tf.test.TestCase):
  dtype = np.float32
  use_static_shape = False


@tfe.run_all_tests_in_graph_and_eager_modes
class StructuralTimeSeriesTestsStaticShape64(
    _StructuralTimeSeriesTests, tf.test.TestCase):
  dtype = np.float64
  use_static_shape = True


class _StsTestHarness(object):

  def setUp(self):
    np.random.seed(142)

  @tfe.run_test_in_graph_and_eager_modes
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

  @tfe.run_test_in_graph_and_eager_modes
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

    # We alternate full_batch_shape, partial_batch_shape in sequence so that in
    # a model with only one parameter, that parameter is constructed with full
    # batch shape.
    batch_shaped_parameters = [
        p.prior.sample(sample_shape=full_batch_shape if (i % 2 == 0)
                       else partial_batch_shape)
        for (i, p) in enumerate(model.parameters)]
    lp = self.evaluate(log_joint_fn(*batch_shaped_parameters))
    self.assertEqual(tf.TensorShape(full_batch_shape), lp.shape)

  @tfe.run_test_in_graph_and_eager_modes
  def test_prior_sample(self):
    model = self._build_sts()
    ys, param_samples = model.prior_sample(
        num_timesteps=5, params_sample_shape=[2], trajectories_sample_shape=[3])

    self.assertAllEqual(ys.shape, [3, 2, 5, 1])
    for sampled, param in zip(param_samples, model.parameters):
      self.assertAllEqual(sampled.shape, [
          2,
      ] + param.prior.batch_shape.as_list() + param.prior.event_shape.as_list())

  @tfe.run_test_in_graph_and_eager_modes
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


class LocalLinearTrendTest(tf.test.TestCase, _StsTestHarness):

  def _build_sts(self, observed_time_series=None):
    return LocalLinearTrend(observed_time_series=observed_time_series)


class SeasonalTest(tf.test.TestCase, _StsTestHarness):

  def _build_sts(self, observed_time_series=None):
    # Note that a Seasonal model with `num_steps_per_season > 1` would have
    # deterministic dependence between timesteps, so evaluating `log_prob` of an
    # arbitary time series leads to Cholesky decomposition errors unless the
    # model also includes an observation noise component (which it would in
    # practice, but this test harness attempts to test the component in
    # isolation). The `num_steps_per_season=1` case tested here will not suffer
    # from this issue.
    return Seasonal(num_seasons=7,
                    num_steps_per_season=1,
                    observed_time_series=observed_time_series)


class SeasonalWithMultipleStepsAndNoiseTest(tf.test.TestCase, _StsTestHarness):

  def _build_sts(self, observed_time_series=None):
    day_of_week = tfp.sts.Seasonal(num_seasons=7,
                                   num_steps_per_season=24,
                                   observed_time_series=observed_time_series,
                                   name='day_of_week')
    return tfp.sts.Sum(components=[day_of_week],
                       observed_time_series=observed_time_series)


class SumTest(tf.test.TestCase, _StsTestHarness):

  def _build_sts(self, observed_time_series=None):
    first_component = LocalLinearTrend(
        observed_time_series=observed_time_series, name='first_component')
    second_component = LocalLinearTrend(
        observed_time_series=observed_time_series, name='second_component')
    return Sum(
        components=[first_component, second_component],
        observed_time_series=observed_time_series)


class LinearRegressionTest(tf.test.TestCase, _StsTestHarness):

  def _build_sts(self, observed_time_series=None):
    max_timesteps = 100
    num_features = 3

    prior = tfd.Laplace(0., 1.)

    # LinearRegression components don't currently take an `observed_time_series`
    # argument, so they can't infer a prior batch shape. This means we have to
    # manually set the batch shape expected by the tests.
    if observed_time_series is not None:
      observed_time_series = sts_util.maybe_expand_trailing_dim(
          observed_time_series)
      batch_shape = observed_time_series.shape[:-2]
      prior = tfd.TransformedDistribution(prior, tfb.Identity(),
                                          event_shape=[num_features],
                                          batch_shape=batch_shape)

    regression = LinearRegression(
        design_matrix=tf.random_normal([max_timesteps, num_features]),
        weights_prior=prior)
    return Sum(components=[regression],
               observed_time_series=observed_time_series)

if __name__ == '__main__':
  tf.test.main()
