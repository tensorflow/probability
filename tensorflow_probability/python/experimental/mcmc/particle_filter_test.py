# Copyright 2020 The TensorFlow Probability Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the _License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Tests for particle filtering."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.experimental.mcmc.particle_filter import SampleParticles
from tensorflow_probability.python.internal import prefer_static
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.internal import test_util


tfb = tfp.bijectors
tfd = tfp.distributions


def do_not_compile(f):
  """The identity function decorator."""
  return f


def xla_compile(f):
  """Decorator for XLA compilation."""
  return tf.function(f, autograph=False, experimental_compile=True)


@test_util.test_all_tf_execution_regimes
class SampleParticlesTest(test_util.TestCase):

  def test_sample_particles_works_with_joint_distributions(self):
    num_particles = 3
    jd = tfd.JointDistributionNamed({'x': tfd.Normal(0., 1.)})
    sp = SampleParticles(jd, num_particles=num_particles)

    # Check that SampleParticles has the correct shapes.
    self.assertAllEqualNested(jd.event_shape, sp.event_shape)
    self.assertAllEqualNested(
        *self.evaluate((jd.event_shape_tensor(), sp.event_shape_tensor())))
    self.assertAllEqualNested(
        tf.nest.map_structure(
            lambda x: np.concatenate([[num_particles], x], axis=0),
            jd.batch_shape),
        tf.nest.map_structure(tensorshape_util.as_list, sp.batch_shape))
    self.assertAllEqualNested(
        *self.evaluate(
            (tf.nest.map_structure(
                lambda x: tf.concat([[num_particles], x], axis=0),
                jd.batch_shape_tensor()),
             sp.batch_shape_tensor())))

    # Check that sample and log-prob work, and that we can take the log-prob
    # of a sample.
    x = self.evaluate(sp.sample())
    lp = self.evaluate(sp.log_prob(x))
    self.assertAllEqual(
        [part.shape for part in tf.nest.flatten(x)], [[num_particles]])
    self.assertAllEqual(
        [part.shape for part in tf.nest.flatten(lp)], [[num_particles]])

  def test_sample_particles_works_with_batch_and_event_shape(self):
    num_particles = 3
    d = tfd.MultivariateNormalDiag(loc=tf.zeros([2, 4]),
                                   scale_diag=tf.ones([2, 4]))
    sp = SampleParticles(d, num_particles=num_particles)

    # Check that SampleParticles has the correct shapes.
    self.assertAllEqual(sp.event_shape, d.event_shape)
    self.assertAllEqual(sp.batch_shape,
                        np.concatenate([[num_particles],
                                        d.batch_shape], axis=0))

    # Draw a sample, combining sample shape, batch shape, num_particles, *and*
    # event_shape, and check that it has the correct shape, and that we can
    # compute a log_prob with the correct shape.
    sample_shape = [5, 1]
    x = self.evaluate(sp.sample(sample_shape, seed=test_util.test_seed()))
    self.assertAllEqual(x.shape,  # [5, 3, 1, 2, 4]
                        np.concatenate([sample_shape,
                                        [num_particles],
                                        d.batch_shape,
                                        d.event_shape],
                                       axis=0))
    lp = self.evaluate(sp.log_prob(x))
    self.assertAllEqual(lp.shape,
                        np.concatenate([sample_shape,
                                        [num_particles],
                                        d.batch_shape],
                                       axis=0))


@test_util.test_all_tf_execution_regimes
class _ParticleFilterTest(test_util.TestCase):

  def test_random_walk(self):
    initial_state_prior = tfd.JointDistributionNamed({
        'position': tfd.Deterministic(0.)})

    # Biased random walk.
    def particle_dynamics(_, previous_state):
      state_shape = tf.shape(previous_state['position'])
      return tfd.JointDistributionNamed({
          'position': tfd.TransformedDistribution(
              tfd.Bernoulli(probs=tf.broadcast_to(0.75, state_shape),
                            dtype=self.dtype),
              tfb.Shift(previous_state['position']))})

    # Completely uninformative observations allowing a test
    # of the pure dynamics.
    def particle_observations(_, state):
      state_shape = tf.shape(state['position'])
      return tfd.Uniform(low=tf.broadcast_to(-100.0, state_shape),
                         high=tf.broadcast_to(100.0, state_shape))

    observations = tf.zeros((9,), dtype=self.dtype)
    trajectories, _ = self.evaluate(
        tfp.experimental.mcmc.infer_trajectories(
            observations=observations,
            initial_state_prior=initial_state_prior,
            transition_fn=particle_dynamics,
            observation_fn=particle_observations,
            num_particles=16384,
            seed=test_util.test_seed()))
    position = trajectories['position']

    # The trajectories have the following properties:
    # 1. they lie completely in the range [0, 8]
    self.assertAllInRange(position, 0., 8.)
    # 2. each step lies in the range [0, 1]
    self.assertAllInRange(position[1:] - position[:-1], 0., 1.)
    # 3. the expectation and variance of the final positions are 6 and 1.5.
    self.assertAllClose(tf.reduce_mean(position[-1]), 6., atol=0.1)
    self.assertAllClose(tf.math.reduce_variance(position[-1]), 1.5, atol=0.1)

  def test_batch_of_filters(self):

    batch_shape = [3, 2]
    num_particles = 1000
    num_timesteps = 40

    # Batch of priors on object 1D positions and velocities.
    initial_state_prior = tfd.JointDistributionNamed({
        'position': tfd.Normal(loc=0., scale=tf.ones(batch_shape)),
        'velocity': tfd.Normal(loc=0., scale=tf.ones(batch_shape) * 0.1)})

    def transition_fn(_, previous_state):
      return tfd.JointDistributionNamed({
          'position': tfd.Normal(
              loc=previous_state['position'] + previous_state['velocity'],
              scale=0.1),
          'velocity': tfd.Normal(loc=previous_state['velocity'], scale=0.01)})

    def observation_fn(_, state):
      return tfd.Normal(loc=state['position'], scale=0.1)

    # Batch of synthetic observations, .
    true_initial_positions = np.random.randn(*batch_shape).astype(self.dtype)
    true_velocities = 0.1 * np.random.randn(
        *batch_shape).astype(self.dtype)
    observed_positions = (
        true_velocities *
        np.arange(num_timesteps).astype(
            self.dtype)[..., tf.newaxis, tf.newaxis] +
        true_initial_positions)

    (particles,
     log_weights,
     parent_indices,
     incremental_log_marginal_likelihoods) = self.evaluate(
         tfp.experimental.mcmc.particle_filter(
             observations=observed_positions,
             initial_state_prior=initial_state_prior,
             transition_fn=transition_fn,
             observation_fn=observation_fn,
             num_particles=num_particles,
             seed=test_util.test_seed()))

    self.assertAllEqual(particles['position'].shape,
                        [num_timesteps, num_particles] + batch_shape)
    self.assertAllEqual(particles['velocity'].shape,
                        [num_timesteps, num_particles] + batch_shape)
    self.assertAllEqual(parent_indices.shape,
                        [num_timesteps, num_particles] + batch_shape)
    self.assertAllEqual(incremental_log_marginal_likelihoods.shape,
                        [num_timesteps] + batch_shape)

    self.assertAllClose(
        self.evaluate(
            tf.reduce_sum(tf.exp(log_weights) *
                          particles['position'], axis=1)),
        observed_positions,
        atol=0.1)

    velocity_means = tf.reduce_sum(tf.exp(log_weights) *
                                   particles['velocity'], axis=1)
    self.assertAllClose(
        self.evaluate(tf.reduce_mean(velocity_means, axis=0)),
        true_velocities, atol=0.05)

    # Uncertainty in velocity should decrease over time.
    velocity_stddev = self.evaluate(
        tf.math.reduce_std(particles['velocity'], axis=1))
    self.assertAllLess((velocity_stddev[-1] - velocity_stddev[0]), 0.)

    trajectories = self.evaluate(
        tfp.experimental.mcmc.reconstruct_trajectories(particles,
                                                       parent_indices))
    self.assertAllEqual([num_timesteps, num_particles] + batch_shape,
                        trajectories['position'].shape)
    self.assertAllEqual([num_timesteps, num_particles] + batch_shape,
                        trajectories['velocity'].shape)

    # Verify that `infer_trajectories` also works on batches.
    trajectories, incremental_log_marginal_likelihoods = self.evaluate(
        tfp.experimental.mcmc.infer_trajectories(
            observations=observed_positions,
            initial_state_prior=initial_state_prior,
            transition_fn=transition_fn,
            observation_fn=observation_fn,
            num_particles=num_particles,
            seed=test_util.test_seed()))
    self.assertAllEqual([num_timesteps, num_particles] + batch_shape,
                        trajectories['position'].shape)
    self.assertAllEqual([num_timesteps, num_particles] + batch_shape,
                        trajectories['velocity'].shape)
    self.assertAllEqual(incremental_log_marginal_likelihoods.shape,
                        [num_timesteps] + batch_shape)

  def test_reconstruct_trajectories_toy_example(self):
    particles = tf.convert_to_tensor([[1, 2, 3], [4, 5, 6,], [7, 8, 9]])
    # 1  --  4  -- 7
    # 2  \/  5  .- 8
    # 3  /\  6 /-- 9
    parent_indices = tf.convert_to_tensor([[0, 1, 2], [0, 2, 1], [0, 2, 2]])

    trajectories = self.evaluate(
        tfp.experimental.mcmc.reconstruct_trajectories(particles,
                                                       parent_indices))
    self.assertAllEqual(
        np.array([[1, 2, 2], [4, 6, 6], [7, 8, 9]]), trajectories)

  def test_epidemiological_model(self):
    # A toy, discrete version of an SIR (Susceptible, Infected, Recovered)
    # model (https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology)

    population_size = 1000
    infection_rate = tf.convert_to_tensor(1.1)
    infectious_period = tf.convert_to_tensor(8.0)

    initial_state_prior = tfd.JointDistributionNamed({
        'susceptible': tfd.Deterministic(999.),
        'infected': tfd.Deterministic(1.),
        'new_infections': tfd.Deterministic(1.),
        'new_recoveries': tfd.Deterministic(0.)})

    # Dynamics model: new infections and recoveries are given by the SIR
    # model with Poisson noise.
    def infection_dynamics(_, previous_state):
      new_infections = tfd.Poisson(
          infection_rate * previous_state['infected'] *
          previous_state['susceptible'] / population_size)
      new_recoveries = tfd.Poisson(previous_state['infected'] /
                                   infectious_period)

      def susceptible(new_infections):
        return tfd.Deterministic(
            prefer_static.maximum(
                0., previous_state['susceptible'] - new_infections))

      def infected(new_infections, new_recoveries):
        return tfd.Deterministic(
            prefer_static.maximum(
                0.,
                previous_state['infected'] + new_infections - new_recoveries))

      return tfd.JointDistributionNamed({
          'new_infections': new_infections,
          'new_recoveries': new_recoveries,
          'susceptible': susceptible,
          'infected': infected})

    # Observation model: each day we detect new cases, noisily.
    def infection_observations(_, state):
      return tfd.Poisson(state['infected'])

    # pylint: disable=bad-whitespace
    observations = tf.convert_to_tensor([
        0.,     4.,   1.,   5.,  23.,  27.,  75., 127., 248., 384., 540., 683.,
        714., 611., 561., 493., 385., 348., 300., 277., 249., 219., 216., 174.,
        132., 122., 115.,  99.,  76.,  84.,  77.,  56.,  42.,  56.,  46.,  38.,
        34.,   44.,  25.,  27.])
    # pylint: enable=bad-whitespace

    trajectories, _ = self.evaluate(
        tfp.experimental.mcmc.infer_trajectories(
            observations=observations,
            initial_state_prior=initial_state_prior,
            transition_fn=infection_dynamics,
            observation_fn=infection_observations,
            num_particles=100,
            seed=test_util.test_seed()))

    # The susceptible population should decrease over time.
    self.assertAllLessEqual(
        trajectories['susceptible'][1:, ...] -
        trajectories['susceptible'][:-1, ...],
        0.0)

  def test_data_driven_proposal(self):

    num_particles = 100
    observations = tf.convert_to_tensor([60., -179.2, 1337.42])

    # Define a system constrained primarily by observations, where proposing
    # from the dynamics would be a bad fit.
    initial_state_prior = tfd.Normal(loc=0., scale=1e6)
    transition_fn = (
        lambda _, previous_state: tfd.Normal(loc=previous_state, scale=1e6))
    observation_fn = lambda _, state: tfd.Normal(loc=state, scale=0.1)
    initial_state_proposal = tfd.Normal(loc=observations[0], scale=0.1)
    proposal_fn = (lambda step, state: tfd.Normal(  # pylint: disable=g-long-lambda
        loc=tf.ones_like(state) * observations[step + 1], scale=1.0))

    trajectories, _ = self.evaluate(
        tfp.experimental.mcmc.infer_trajectories(
            observations=observations,
            initial_state_prior=initial_state_prior,
            transition_fn=transition_fn,
            observation_fn=observation_fn,
            num_particles=num_particles,
            initial_state_proposal=initial_state_proposal,
            proposal_fn=proposal_fn,
            seed=test_util.test_seed()))
    self.assertAllClose(trajectories,
                        tf.convert_to_tensor(
                            tf.convert_to_tensor(
                                observations)[..., tf.newaxis] *
                            tf.ones([num_particles])), atol=1.0)

  def test_estimated_prob_approximates_true_prob(self):

    # Draw simulated data from a 2D linear Gaussian system.
    initial_state_prior = tfd.MultivariateNormalDiag(
        loc=0., scale_diag=(1., 1.))
    transition_matrix = tf.convert_to_tensor([[1., -0.5], [0.4, -1.]])
    transition_noise = tfd.MultivariateNormalTriL(
        loc=1., scale_tril=tf.convert_to_tensor([[0.3, 0], [-0.1, 0.2]]))
    observation_matrix = tf.convert_to_tensor([[0.1, 1.], [1., 0.2]])
    observation_noise = tfd.MultivariateNormalTriL(
        loc=-0.3, scale_tril=tf.convert_to_tensor([[0.5, 0], [0.1, 0.5]]))
    model = tfd.LinearGaussianStateSpaceModel(
        num_timesteps=20,
        initial_state_prior=initial_state_prior,
        transition_matrix=transition_matrix,
        transition_noise=transition_noise,
        observation_matrix=observation_matrix,
        observation_noise=observation_noise)
    observations = self.evaluate(
        model.sample(seed=test_util.test_seed()))
    (lps, filtered_means,
     _, _, _, _, _) = self.evaluate(model.forward_filter(observations))

    # Approximate the filtering means and marginal likelihood(s) using
    # the particle filter.
    # pylint: disable=g-long-lambda
    (particles, log_weights, _,
     estimated_incremental_log_marginal_likelihoods) = self.evaluate(
         tfp.experimental.mcmc.particle_filter(
             observations=observations,
             initial_state_prior=initial_state_prior,
             transition_fn=lambda _, previous_state: tfd.MultivariateNormalTriL(
                 loc=transition_noise.loc + tf.linalg.matvec(
                     transition_matrix, previous_state),
                 scale_tril=transition_noise.scale_tril),
             observation_fn=lambda _, state: tfd.MultivariateNormalTriL(
                 loc=observation_noise.loc + tf.linalg.matvec(
                     observation_matrix, state),
                 scale_tril=observation_noise.scale_tril),
             num_particles=1024,
             seed=test_util.test_seed()))
    # pylint: enable=g-long-lambda

    particle_means = np.sum(
        particles * np.exp(log_weights)[..., np.newaxis], axis=1)
    self.assertAllClose(filtered_means, particle_means, atol=0.1, rtol=0.1)

    self.assertAllClose(
        lps, estimated_incremental_log_marginal_likelihoods, atol=0.6)

  def test_proposal_weights_dont_affect_marginal_likelihood(self):
    observation = np.array([-1.3, 0.7]).astype(self.dtype)
    # This particle filter has proposals different from the dynamics,
    # so internally it will use proposal weights in addition to observation
    # weights. It should still get the observation likelihood correct.
    _, lps = self.evaluate(tfp.experimental.mcmc.infer_trajectories(
        observation,
        initial_state_prior=tfd.Normal(loc=0., scale=1.),
        transition_fn=lambda _, x: tfd.Normal(loc=x, scale=1.),
        observation_fn=lambda _, x: tfd.Normal(loc=x, scale=1.),
        initial_state_proposal=tfd.Normal(loc=0., scale=5.),
        proposal_fn=lambda _, x: tfd.Normal(loc=x, scale=5.),
        num_particles=1024,
        seed=test_util.test_seed()))

    # Compare marginal likelihood against that
    # from the true (jointly normal) marginal distribution.
    y1_marginal_dist = tfd.Normal(loc=0., scale=np.sqrt(1. + 1.))
    y2_conditional_dist = (
        lambda y1: tfd.Normal(loc=y1 / 2., scale=np.sqrt(5. / 2.)))
    true_lps = [y1_marginal_dist.log_prob(observation[0]),
                y2_conditional_dist(observation[0]).log_prob(observation[1])]
    # The following line passes at atol = 0.01 if num_particles = 32768.
    self.assertAllClose(true_lps, lps, atol=0.1)

  def test_can_step_dynamics_faster_than_observations(self):
    initial_state_prior = tfd.JointDistributionNamed({
        'position': tfd.Deterministic(1.),
        'velocity': tfd.Deterministic(0.)
    })

    # Use 100 steps between observations to integrate a simple harmonic
    # oscillator.
    dt = 0.01
    def simple_harmonic_motion_transition_fn(_, state):
      return tfd.JointDistributionNamed({
          'position': tfd.Normal(
              loc=state['position'] + dt * state['velocity'], scale=dt*0.01),
          'velocity': tfd.Normal(
              loc=state['velocity'] - dt * state['position'], scale=dt*0.01)
      })

    def observe_position(_, state):
      return tfd.Normal(loc=state['position'], scale=0.01)

    particles, _, _, lps = self.evaluate(tfp.experimental.mcmc.particle_filter(
        # 'Observing' the values we'd expect from a proper integrator should
        # give high likelihood if our discrete approximation is good.
        observations=tf.convert_to_tensor([tf.math.cos(0.),
                                           tf.math.cos(1.)]),
        initial_state_prior=initial_state_prior,
        transition_fn=simple_harmonic_motion_transition_fn,
        observation_fn=observe_position,
        num_particles=1024,
        num_transitions_per_observation=100,
        seed=test_util.test_seed()))

    self.assertLen(particles['position'], 101)
    self.assertAllClose(np.mean(particles['position'], axis=-1),
                        tf.math.cos(dt * np.arange(101)),
                        atol=0.04)
    self.assertLen(lps, 101)
    self.assertGreater(lps[0], 3.)
    self.assertGreater(lps[-1], 3.)

  def test_custom_trace_fn(self):

    def trace_fn(step_results):
      # Traces the mean and stddev of the particle population at each step.
      weights = tf.exp(step_results.log_weights)
      mean = tf.reduce_sum(weights * step_results.particles, axis=0)
      variance = tf.reduce_sum(
          weights * (step_results.particles - mean[tf.newaxis, ...])**2)
      return {'mean': mean,
              'stddev': tf.sqrt(variance),
              # In real usage we would likely not track the particles and
              # weights. We keep them here just so we can double-check the
              # stats, below.
              'particles': step_results.particles,
              'weights': weights}

    results = self.evaluate(
        tfp.experimental.mcmc.particle_filter(
            observations=tf.convert_to_tensor([1., 3., 5., 7., 9.]),
            initial_state_prior=tfd.Normal(0., 1.),
            transition_fn=lambda _, state: tfd.Normal(state, 1.),
            observation_fn=lambda _, state: tfd.Normal(state, 1.),
            num_particles=1024,
            trace_fn=trace_fn,
            seed=test_util.test_seed()))

    # Verify that posterior means are increasing.
    self.assertAllGreater(results['mean'][1:] - results['mean'][:-1], 0.)

    # Check that our traced means and scales match values computed
    # by averaging over particles after the fact.
    all_means = self.evaluate(tf.reduce_sum(
        results['weights'] * results['particles'], axis=1))
    all_variances = self.evaluate(
        tf.reduce_sum(
            results['weights'] *
            (results['particles'] - all_means[..., tf.newaxis])**2,
            axis=1))
    self.assertAllClose(results['mean'], all_means)
    self.assertAllClose(results['stddev'], np.sqrt(all_variances))

  def test_step_indices_to_trace(self):
    num_particles = 1024
    (particles_1_3,
     log_weights_1_3,
     parent_indices_1_3,
     incremental_log_marginal_likelihood_1_3) = self.evaluate(
         tfp.experimental.mcmc.particle_filter(
             observations=tf.convert_to_tensor([1., 3., 5., 7., 9.]),
             initial_state_prior=tfd.Normal(0., 1.),
             transition_fn=lambda _, state: tfd.Normal(state, 10.),
             observation_fn=lambda _, state: tfd.Normal(state, 0.1),
             num_particles=num_particles,
             step_indices_to_trace=[1, 3],
             seed=test_util.test_seed()))
    self.assertLen(particles_1_3, 2)
    self.assertLen(log_weights_1_3, 2)
    self.assertLen(parent_indices_1_3, 2)
    self.assertLen(incremental_log_marginal_likelihood_1_3, 2)
    means = np.sum(np.exp(log_weights_1_3) * particles_1_3, axis=1)
    self.assertAllClose(means, [3., 7.], atol=1.)

    (final_particles,
     final_log_weights,
     final_cumulative_lp) = self.evaluate(
         tfp.experimental.mcmc.particle_filter(
             observations=tf.convert_to_tensor([1., 3., 5., 7., 9.]),
             initial_state_prior=tfd.Normal(0., 1.),
             transition_fn=lambda _, state: tfd.Normal(state, 10.),
             observation_fn=lambda _, state: tfd.Normal(state, 0.1),
             num_particles=num_particles,
             trace_fn=lambda r: (r.particles,  # pylint: disable=g-long-lambda
                                 r.log_weights,
                                 r.accumulated_log_marginal_likelihood),
             step_indices_to_trace=-1,
             seed=test_util.test_seed()))
    self.assertLen(final_particles, num_particles)
    self.assertLen(final_log_weights, num_particles)
    self.assertEqual(final_cumulative_lp.shape, ())
    means = np.sum(np.exp(final_log_weights) * final_particles)
    self.assertAllClose(means, 9., atol=1.5)


class ParticleFilterTestFloat32(_ParticleFilterTest):
  dtype = np.float32


del _ParticleFilterTest


if __name__ == '__main__':
  tf.test.main()
