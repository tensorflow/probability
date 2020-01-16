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
"""Additive State Space Model Tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf

from tensorflow_probability import distributions as tfd
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.sts import AdditiveStateSpaceModel
from tensorflow_probability.python.sts import LocalLinearTrendStateSpaceModel


tfl = tf.linalg


class _AdditiveStateSpaceModelTest(test_util.TestCase):

  def test_identity(self):

    # Test that an additive SSM with a single component defines the same
    # distribution as the component model.

    y = self._build_placeholder([1.0, 2.5, 4.3, 6.1, 7.8])

    local_ssm = LocalLinearTrendStateSpaceModel(
        num_timesteps=5,
        level_scale=0.3,
        slope_scale=0.6,
        observation_noise_scale=0.1,
        initial_state_prior=tfd.MultivariateNormalDiag(
            scale_diag=self._build_placeholder([1., 1.])))

    additive_ssm = AdditiveStateSpaceModel([local_ssm])

    local_lp = local_ssm.log_prob(y[:, np.newaxis])
    additive_lp = additive_ssm.log_prob(y[:, np.newaxis])
    self.assertAllClose(self.evaluate(local_lp), self.evaluate(additive_lp))

  def test_nesting_additive_ssms(self):

    ssm1 = self._dummy_model(batch_shape=[1, 2])
    ssm2 = self._dummy_model(batch_shape=[3, 2])
    observation_noise_scale = 0.1

    additive_ssm = AdditiveStateSpaceModel(
        [ssm1, ssm2],
        observation_noise_scale=observation_noise_scale)

    nested_additive_ssm = AdditiveStateSpaceModel(
        [AdditiveStateSpaceModel([ssm1]),
         AdditiveStateSpaceModel([ssm2])],
        observation_noise_scale=observation_noise_scale)

    # Test that both models behave equivalently.
    y = self.evaluate(nested_additive_ssm.sample())

    additive_lp = additive_ssm.log_prob(y)
    nested_additive_lp = nested_additive_ssm.log_prob(y)
    self.assertAllClose(self.evaluate(additive_lp),
                        self.evaluate(nested_additive_lp))

    additive_mean = additive_ssm.mean()
    nested_additive_mean = nested_additive_ssm.mean()
    self.assertAllClose(
        self.evaluate(additive_mean),
        self.evaluate(nested_additive_mean))

    additive_variance = additive_ssm.variance()
    nested_additive_variance = nested_additive_ssm.variance()
    self.assertAllClose(
        self.evaluate(additive_variance),
        self.evaluate(nested_additive_variance))

  def test_sum_of_local_linear_trends(self):

    # We know analytically that the sum of two local linear trends is
    # another local linear trend, with means and variances scaled
    # accordingly, so the additive model should match this behavior.

    level_scale = 0.5
    slope_scale = 1.1
    initial_level = 3.
    initial_slope = -2.
    observation_noise_scale = 0.
    num_timesteps = 5
    y = self._build_placeholder([1.0, 2.5, 4.3, 6.1, 7.8])

    # Combine two local linear trend models, one a full model, the other
    # with just a moving mean (zero slope).
    local_ssm = LocalLinearTrendStateSpaceModel(
        num_timesteps=num_timesteps,
        level_scale=level_scale,
        slope_scale=slope_scale,
        initial_state_prior=tfd.MultivariateNormalDiag(
            loc=self._build_placeholder([initial_level, initial_slope]),
            scale_diag=self._build_placeholder([1., 1.])))

    second_level_scale = 0.3
    second_initial_level = 1.1
    moving_level_ssm = LocalLinearTrendStateSpaceModel(
        num_timesteps=num_timesteps,
        level_scale=second_level_scale,
        slope_scale=0.,
        initial_state_prior=tfd.MultivariateNormalDiag(
            loc=self._build_placeholder([second_initial_level, 0.]),
            scale_diag=self._build_placeholder([1., 0.])))

    additive_ssm = AdditiveStateSpaceModel(
        [local_ssm, moving_level_ssm],
        observation_noise_scale=observation_noise_scale)

    # Build the analytical sum of the two processes.
    target_ssm = LocalLinearTrendStateSpaceModel(
        num_timesteps=num_timesteps,
        level_scale=np.float32(np.sqrt(level_scale**2 + second_level_scale**2)),
        slope_scale=np.float32(slope_scale),
        observation_noise_scale=observation_noise_scale,
        initial_state_prior=tfd.MultivariateNormalDiag(
            loc=self._build_placeholder(
                [initial_level + second_initial_level, initial_slope + 0.]),
            scale_diag=self._build_placeholder(
                np.sqrt([2., 1.]))))

    # Test that both models behave equivalently.
    additive_mean = additive_ssm.mean()
    target_mean = target_ssm.mean()
    self.assertAllClose(
        self.evaluate(additive_mean), self.evaluate(target_mean))

    additive_variance = additive_ssm.variance()
    target_variance = target_ssm.variance()
    self.assertAllClose(
        self.evaluate(additive_variance), self.evaluate(target_variance))

    additive_lp = additive_ssm.log_prob(y[:, np.newaxis])
    target_lp = target_ssm.log_prob(y[:, np.newaxis])
    self.assertAllClose(self.evaluate(additive_lp), self.evaluate(target_lp))

  def test_batch_shape(self):
    batch_shape = [3, 2]

    ssm = self._dummy_model(batch_shape=batch_shape)
    additive_ssm = AdditiveStateSpaceModel([ssm, ssm])
    y = additive_ssm.sample()

    if self.use_static_shape:
      self.assertAllEqual(additive_ssm.batch_shape.as_list(), batch_shape)
      self.assertAllEqual(y.shape.as_list()[:-2], batch_shape)
    else:
      self.assertAllEqual(self.evaluate(additive_ssm.batch_shape_tensor()),
                          batch_shape)
      self.assertAllEqual(self.evaluate(tf.shape(y))[:-2], batch_shape)

  def test_multivariate_observations(self):

    # since STS components are scalar by design, we manually construct
    # a multivariate-output model to verify that the additive SSM handles
    # this case.
    num_timesteps = 5
    observation_size = 2
    multivariate_ssm = self._dummy_model(num_timesteps=num_timesteps,
                                         observation_size=observation_size)

    # Note it would not work to specify observation_noise_scale here;
    # multivariate observations need to derive the (multivariate)
    # observation noise distribution from their components.
    combined_ssm = AdditiveStateSpaceModel([multivariate_ssm,
                                            multivariate_ssm])

    y = combined_ssm.sample()
    expected_event_shape = [num_timesteps, observation_size]
    if self.use_static_shape:
      self.assertAllEqual(combined_ssm.event_shape.as_list(),
                          expected_event_shape)
      self.assertAllEqual(y.shape.as_list()[-2:], expected_event_shape)
    else:
      self.assertAllEqual(self.evaluate(combined_ssm.event_shape_tensor()),
                          expected_event_shape)
      self.assertAllEqual(
          self.evaluate(tf.shape(y))[-2:], expected_event_shape)

  def test_mismatched_num_timesteps_error(self):

    ssm1 = self._dummy_model(num_timesteps=10)
    ssm2 = self._dummy_model(num_timesteps=8)

    with self.assertRaisesWithPredicateMatch(
        ValueError, 'same number of timesteps'):

      # In the static case, the constructor should raise an exception.
      additive_ssm = AdditiveStateSpaceModel(
          component_ssms=[ssm1, ssm2])

      # In the dynamic case, the exception is raised at runtime.
      _ = self.evaluate(additive_ssm.sample())

  def test_broadcasting_batch_shape(self):

    # Build three SSMs with broadcast batch shape.
    ssm1 = self._dummy_model(batch_shape=[2])
    ssm2 = self._dummy_model(batch_shape=[3, 2])
    ssm3 = self._dummy_model(batch_shape=[1, 2])

    additive_ssm = AdditiveStateSpaceModel(
        component_ssms=[ssm1, ssm2, ssm3])
    y = additive_ssm.sample()

    broadcast_batch_shape = [3, 2]
    if self.use_static_shape:
      self.assertAllEqual(additive_ssm.batch_shape.as_list(),
                          broadcast_batch_shape)
      self.assertAllEqual(y.shape.as_list()[:-2],
                          broadcast_batch_shape)
    else:
      self.assertAllEqual(self.evaluate(additive_ssm.batch_shape_tensor()),
                          broadcast_batch_shape)
      self.assertAllEqual(
          self.evaluate(tf.shape(y))[:-2], broadcast_batch_shape)

  def test_broadcasting_correctness(self):

    # This test verifies that broadcasting of component parameters works as
    # expected. We construct a SSM with no batch shape, and test that when we
    # add it to another SSM of batch shape [3], we get the same model
    # as if we had explicitly broadcast the parameters of the first SSM before
    # adding.

    num_timesteps = 5
    transition_matrix = np.random.randn(2, 2)
    transition_noise_diag = np.exp(np.random.randn(2))
    observation_matrix = np.random.randn(1, 2)
    observation_noise_diag = np.exp(np.random.randn(1))
    initial_state_prior_diag = np.exp(np.random.randn(2))

    # First build the model in which we let AdditiveSSM do the broadcasting.
    batchless_ssm = tfd.LinearGaussianStateSpaceModel(
        num_timesteps=num_timesteps,
        transition_matrix=self._build_placeholder(transition_matrix),
        transition_noise=tfd.MultivariateNormalDiag(
            scale_diag=self._build_placeholder(transition_noise_diag)),
        observation_matrix=self._build_placeholder(observation_matrix),
        observation_noise=tfd.MultivariateNormalDiag(
            scale_diag=self._build_placeholder(observation_noise_diag)),
        initial_state_prior=tfd.MultivariateNormalDiag(
            scale_diag=self._build_placeholder(initial_state_prior_diag))
    )
    another_ssm = self._dummy_model(num_timesteps=num_timesteps,
                                    latent_size=4,
                                    batch_shape=[3])
    broadcast_additive_ssm = AdditiveStateSpaceModel(
        [batchless_ssm, another_ssm])

    # Next try doing our own broadcasting explicitly.
    broadcast_vector = np.ones([3, 1])
    broadcast_matrix = np.ones([3, 1, 1])
    batch_ssm = tfd.LinearGaussianStateSpaceModel(
        num_timesteps=num_timesteps,
        transition_matrix=self._build_placeholder(
            transition_matrix * broadcast_matrix),
        transition_noise=tfd.MultivariateNormalDiag(
            scale_diag=self._build_placeholder(
                transition_noise_diag * broadcast_vector)),
        observation_matrix=self._build_placeholder(
            observation_matrix * broadcast_matrix),
        observation_noise=tfd.MultivariateNormalDiag(
            scale_diag=self._build_placeholder(
                observation_noise_diag * broadcast_vector)),
        initial_state_prior=tfd.MultivariateNormalDiag(
            scale_diag=self._build_placeholder(
                initial_state_prior_diag * broadcast_vector)))
    manual_additive_ssm = AdditiveStateSpaceModel([batch_ssm, another_ssm])

    # Both additive SSMs define the same model, so they should give the same
    # log_probs.
    y = self.evaluate(broadcast_additive_ssm.sample(seed=42))
    self.assertAllClose(self.evaluate(broadcast_additive_ssm.log_prob(y)),
                        self.evaluate(manual_additive_ssm.log_prob(y)))

  def test_mismatched_observation_size_error(self):
    ssm1 = self._dummy_model(observation_size=1)
    ssm2 = self._dummy_model(observation_size=2)

    with self.assertRaisesWithPredicateMatch(Exception, ''):

      # In the static case, the constructor should raise an exception.
      additive_ssm = AdditiveStateSpaceModel(
          component_ssms=[ssm1, ssm2])

      # In the dynamic case, the exception is raised at runtime.
      _ = self.evaluate(additive_ssm.sample())

  def test_mismatched_dtype_error(self):
    ssm1 = self._dummy_model(dtype=self.dtype)
    ssm2 = self._dummy_model(dtype=np.float16)

    with self.assertRaisesRegexp(Exception, 'dtype'):
      _ = AdditiveStateSpaceModel(component_ssms=[ssm1, ssm2])

  def test_constant_offset(self):
    offset_ = 1.23456
    offset = self._build_placeholder(offset_)
    ssm = self._dummy_model()

    additive_ssm = AdditiveStateSpaceModel([ssm])
    additive_ssm_with_offset = AdditiveStateSpaceModel(
        [ssm], constant_offset=offset)
    additive_ssm_with_offset_and_explicit_scale = AdditiveStateSpaceModel(
        [ssm],
        constant_offset=offset,
        observation_noise_scale=(
            ssm.get_observation_noise_for_timestep(0).stddev()[..., 0]))

    mean_, offset_mean_, offset_with_scale_mean_ = self.evaluate(
        (additive_ssm.mean(),
         additive_ssm_with_offset.mean(),
         additive_ssm_with_offset_and_explicit_scale.mean()))
    print(mean_.shape, offset_mean_.shape, offset_with_scale_mean_.shape)
    self.assertAllClose(mean_, offset_mean_ - offset_)
    self.assertAllClose(mean_, offset_with_scale_mean_ - offset_)

    # Offset should not affect the stddev.
    stddev_, offset_stddev_, offset_with_scale_stddev_ = self.evaluate(
        (additive_ssm.stddev(),
         additive_ssm_with_offset.stddev(),
         additive_ssm_with_offset_and_explicit_scale.stddev()))
    self.assertAllClose(stddev_, offset_stddev_)
    self.assertAllClose(stddev_, offset_with_scale_stddev_)

  def test_batch_shape_ignores_component_state_priors(self):
    # If we pass an initial_state_prior directly to an AdditiveSSM, overriding
    # the initial state priors of component models, the overall batch shape
    # should no longer depend on the (overridden) component priors.
    # This ensures that we produce correct shapes in forecasting, where the
    # shapes may have changed to include dimensions corresponding to posterior
    # draws.

    # Create a component model with no batch shape *except* in the initial state
    # prior.
    latent_size = 2
    ssm = self._dummy_model(
        latent_size=latent_size,
        batch_shape=[],
        initial_state_prior_batch_shape=[5, 5])

    # If we override the initial state prior with an unbatched prior, the
    # resulting AdditiveSSM should not have batch dimensions.
    unbatched_initial_state_prior = tfd.MultivariateNormalDiag(
        scale_diag=self._build_placeholder(np.ones([latent_size])))
    additive_ssm = AdditiveStateSpaceModel(
        [ssm], initial_state_prior=unbatched_initial_state_prior)

    self.assertAllEqual(self.evaluate(additive_ssm.batch_shape_tensor()), [])

  def _build_placeholder(self, ndarray, dtype=None):
    """Convert a numpy array to a TF placeholder.

    Args:
      ndarray: any object convertible to a numpy array via `np.asarray()`.
      dtype: numpy `dtype` of the returned placeholder. If `None`, uses
        the default `self.dtype`.

    Returns:
      placeholder: a TensorFlow `placeholder` with default value given by the
      provided `ndarray`, dtype given by `self.dtype`, and shape specified
      statically only if `self.use_static_shape` is `True`.
    """
    dtype = dtype if dtype is not None else self.dtype
    ndarray = np.asarray(ndarray).astype(dtype)
    return tf1.placeholder_with_default(
        ndarray, shape=ndarray.shape if self.use_static_shape else None)

  def _dummy_model(self,
                   num_timesteps=5,
                   batch_shape=None,
                   initial_state_prior_batch_shape=None,
                   latent_size=2,
                   observation_size=1,
                   dtype=None):
    batch_shape = batch_shape if batch_shape is not None else []
    initial_state_prior_batch_shape = (
        initial_state_prior_batch_shape
        if initial_state_prior_batch_shape is not None else batch_shape)
    dtype = dtype if dtype is not None else self.dtype

    return tfd.LinearGaussianStateSpaceModel(
        num_timesteps=num_timesteps,
        transition_matrix=self._build_placeholder(np.eye(latent_size),
                                                  dtype=dtype),
        transition_noise=tfd.MultivariateNormalDiag(
            scale_diag=np.ones(batch_shape + [latent_size]).astype(dtype)),
        observation_matrix=self._build_placeholder(np.random.standard_normal(
            batch_shape + [observation_size, latent_size]), dtype=dtype),
        observation_noise=tfd.MultivariateNormalDiag(
            loc=self._build_placeholder(
                np.ones(batch_shape + [observation_size]), dtype=dtype),
            scale_diag=self._build_placeholder(
                np.ones(batch_shape + [observation_size]), dtype=dtype)),
        initial_state_prior=tfd.MultivariateNormalDiag(
            scale_diag=self._build_placeholder(
                np.ones(initial_state_prior_batch_shape + [latent_size]),
                dtype=dtype)))


@test_util.test_all_tf_execution_regimes
class AdditiveStateSpaceModelTestStaticShape32(_AdditiveStateSpaceModelTest):
  dtype = np.float32
  use_static_shape = True


class AdditiveStateSpaceModelTestDynamicShape32(_AdditiveStateSpaceModelTest):
  dtype = np.float32
  use_static_shape = False

  def test_dynamic_num_timesteps(self):
    # Verify that num_timesteps is set statically when at least one component
    # (not necessarily the first) has static num_timesteps.
    num_timesteps = 4
    dynamic_timesteps_component = self._dummy_model(
        num_timesteps=tf1.placeholder_with_default(num_timesteps, shape=None))
    static_timesteps_component = self._dummy_model(
        num_timesteps=num_timesteps)

    additive_ssm = AdditiveStateSpaceModel([dynamic_timesteps_component,
                                            dynamic_timesteps_component])
    self.assertEqual(self.evaluate(additive_ssm.num_timesteps), num_timesteps)

    additive_ssm = AdditiveStateSpaceModel([dynamic_timesteps_component,
                                            static_timesteps_component])
    self.assertEqual(num_timesteps, self.evaluate(additive_ssm.num_timesteps))


@test_util.test_all_tf_execution_regimes
class AdditiveStateSpaceModelTestStaticShape64(_AdditiveStateSpaceModelTest):
  dtype = np.float64
  use_static_shape = True

del _AdditiveStateSpaceModelTest  # Don't run tests for the base class.

if __name__ == '__main__':
  tf.test.main()
