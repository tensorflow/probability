# Copyright 2020 The TensorFlow Probability Authors.
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
"""Particle filtering."""

import collections
import functools

import tensorflow.compat.v2 as tf
from tensorflow_probability.python import math as tfp_math
from tensorflow_probability.python.distributions import categorical
from tensorflow_probability.python.distributions import distribution as distribution_lib
from tensorflow_probability.python.internal import distribution_util as dist_util
from tensorflow_probability.python.internal import prefer_static
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.internal import tensorshape_util

from tensorflow_probability.python.util import SeedStream

__all__ = [
    'particle_filter',
    'reconstruct_trajectories'
]


# TODO(davmre): add unit tests for SampleParticles.
class SampleParticles(distribution_lib.Distribution):
  """Like tfd.Sample, but inserts new rightmost batch (vs event) dim."""

  def __init__(self, distribution, num_particles, name=None):
    parameters = dict(locals())
    with tf.name_scope(name or 'SampleParticles') as name:
      self._distribution = distribution
      self._num_particles = tensor_util.convert_nonref_to_tensor(
          num_particles, dtype_hint=tf.int32, name='num_particles')
      super(SampleParticles, self).__init__(
          dtype=distribution.dtype,
          reparameterization_type=distribution.reparameterization_type,
          validate_args=distribution.validate_args,
          allow_nan_stats=distribution.allow_nan_stats,
          name=name)
      self._parameters = self._no_dependency(parameters)

  @property
  def num_particles(self):
    return self._num_particles

  @property
  def distribution(self):
    return self._distribution

  def _event_shape(self):
    return self.distribution.event_shape

  def _event_shape_tensor(self, **kwargs):
    return self.distribution.event_shape_tensor(**kwargs)

  def _batch_shape(self):
    return tf.nest.map_structure(
        lambda b: tensorshape_util.concatenate(b, [self.num_particles]),
        self.distribution.batch_shape)

  def _batch_shape_tensor(self, **kwargs):
    return tf.nest.map_structure(
        lambda b: prefer_static.concat([b, [self.num_particles]], axis=0),
        self.distribution.batch_shape_tensor(**kwargs))

  def _log_prob(self, x, **kwargs):
    return self.distribution.log_prob(x, **kwargs)

  def _log_cdf(self, x, **kwargs):
    return self.distribution.log_cdf(x, **kwargs)

  def _log_sf(self, x, **kwargs):
    return self.distribution.log_sf(x, **kwargs)

  # TODO(b/152797117): Override _sample_n, once it supports joint distributions.
  def sample(self, sample_shape=(), seed=None, name=None):
    with tf.name_scope(name or 'sample_particles'):
      sample_shape = prefer_static.concat([
          [self.num_particles],
          dist_util.expand_to_vector(sample_shape)], axis=0)
      x = self.distribution.sample(sample_shape, seed=seed)

      def move_particles_to_rightmost_batch_dim(x, event_shape):
        ndims = prefer_static.rank_from_shape(prefer_static.shape(x))
        event_ndims = prefer_static.rank_from_shape(event_shape)
        return dist_util.move_dimension(x, 0, ndims - event_ndims - 1)
      return tf.nest.map_structure(
          move_particles_to_rightmost_batch_dim,
          x, self.distribution.event_shape_tensor())


def _gather_history(structure, step, num_steps):
  """Gather up to `num_steps` of history from a nested structure."""
  initial_step = prefer_static.maximum(0, step - num_steps)
  return tf.nest.map_structure(
      lambda x: tf.gather(x, prefer_static.range(initial_step, step)),
      structure)


ParticleFilterAccumulatedQuantities = collections.namedtuple(
    'ParticleFilterAccumulatedQuantities',
    ['all_resampled_particles',
     'all_parent_indices',
     'all_step_log_marginal_likelihoods',
     'state_history'  # Set to `tf.zeros([0])` if not tracked.
    ])


def particle_filter(observations,
                    initial_state_prior,
                    transition_fn,
                    observation_fn,
                    num_particles,
                    initial_state_proposal=None,
                    proposal_fn=None,
                    rejuvenation_kernel_fn=None,  # TODO(davmre): not yet supported. pylint: disable=unused-argument
                    num_steps_state_history_to_pass=None,
                    num_steps_observation_history_to_pass=None,
                    seed=None,
                    name=None):
  """Samples a series of particles representing filtered latent states.

  Each latent state is a `Tensor` or nested structure of `Tensor`s, as defined
  by the `initial_state_prior`.

  Each of the `transition_fn`, `observation_fn`, and `proposal_fn` args,
  if specified, takes arguments `(step, state)`, where `state` represents
  the latent state at timestep `step`. These `fn`s may also, optionally, take
  additional keyword arguments `state_history` and `observation_history`, which
  will be passed if and only if the corresponding
  `num_steps_state_history_to_pass` or `num_steps_observation_history_to_pass`
  arguments are provided to this method. These are described further below.

  Args:
    observations: a (structure of) Tensors, each of shape
      `concat([[num_timesteps, b1, ..., bN], event_shape])` with optional
      batch dimensions `b1, ..., bN`.
    initial_state_prior: a (joint) distribution over the initial latent state,
      with optional batch shape `[b1, ..., bN]`.
    transition_fn: callable returning a (joint) distribution over the next
      latent state.
    observation_fn: callable returning a (joint) distribution over the current
      observation.
    num_particles: `int` `Tensor` number of particles.
    initial_state_proposal: a (joint) distribution over the initial latent
      state, with optional batch shape `[b1, ..., bN]`. If `None`, the initial
      particles are proposed from the `initial_state_prior`.
      Default value: `None`.
    proposal_fn: callable returning a (joint) proposal distribution over the
      next latent state. If `None`, the dynamics model is used (
      `proposal_fn == transition_fn`).
      Default value: `None`.
    rejuvenation_kernel_fn: optional Python `callable` with signature
      `transition_kernel = rejuvenation_kernel_fn(target_log_prob_fn)`
      where `target_log_prob_fn` is a provided callable evaluating
      `p(x[t] | y[t], x[t-1])` at each step `t`, and `transition_kernel`
      should be an instance of `tfp.mcmc.TransitionKernel`.
      Default value: `None`.  # TODO(davmre): not yet supported.
    num_steps_state_history_to_pass: Python `int` number of steps to
      include in the optional `state_history` argument to `transition_fn`,
      `observation_fn`, and `proposal_fn`. If `None`, this argument
      will not be passed.
      Default value: `None`.
    num_steps_observation_history_to_pass: Python `int` number of steps to
      include in the optional `observation_history` argument to `transition_fn`,
      `observation_fn`, and `proposal_fn`. If `None`, this argument
      will not be passed.
      Default value: `None`.
    seed: Python `int` seed for random ops.
    name: Python `str` name for ops created by this method.
      Default value: `None` (i.e., `'particle_filter'`).
  Returns:
    particles: a (structure of) Tensor(s) matching the latent state, each
      of shape
      `concat([[num_timesteps, b1, ..., bN, num_particles], event_shape])`,
      representing unbiased samples from the series of (filtering) distributions
      `p(latent_states[t] | observations[:t])`.
    parent_indices: `int` `Tensor` of shape
      `[num_timesteps, b1, ..., bN, num_particles]`,
      such that `parent_indices[t, k]` gives the index of the particle at
      time `t - 1` that the `k`th particle at time `t` is immediately descended
      from. See also
      `tfp.experimental.mcmc.reconstruct_trajectories`.
    step_log_marginal_likelihoods: float `Tensor` of shape
      `[num_timesteps, b1, ..., bN]`,
      giving the natural logarithm of an unbiased estimate of
      `p(observations[t] | observations[:t])` at each timestep `t`. Note that (
      by [Jensen's inequality](
      https://en.wikipedia.org/wiki/Jensen%27s_inequality))
      this is *smaller* in expectation than the true
      `log p(observations[t] | observations[:t])`.

  #### Non-Markovian models (state and observation history).

  Models that do not follow the [Markov property](
  https://en.wikipedia.org/wiki/Markov_property), which requires that the
  current state contains all information relevent to the future of the system,
  are supported by specifying `num_steps_state_history_to_pass` and/or
  `num_steps_observation_history_to_pass`. If these are specified, additional
  keyword arguments `state_history` and/or `observation_history` (respectively)
  will be passed to each of `transition_fn`, `observation_fn`, and
  `proposal_fn`.

  The `state_history`, if requested, is a structure of `Tensor`s like
  the initial state, but with a batch dimension prefixed to every Tensor,
  of size `num_steps_state_history_to_pass` , so that
  `state_history[-1]` represents the previous state
  (for `transition_fn` and `proposal_fn`, this will equal the `state` arg),
  `state_history[-2]` the state before that, and so on.

  The `observation_history` is a structure like `observations`, but with leading
  dimension of `minimum(step, num_steps_observation_history_to_pass)`. At the
  initial step, `observation_history=None` will be passed and should be
  handled appropriately. At subsequent steps, `observation_history[-1]`
  refers to the observation at the previous timestep, and so on.

  #### Examples

  **Tracking unknown position and velocity**: Let's consider tracking an object
  moving in a one-dimensional space. We'll define a dynamical system
  by specifying an `initial_state_prior`, a `transition_fn`,
  and `observation_fn`.

  The structure of the latent state space is determined by the prior
  distribution. Here, we'll define a state space that includes the object's
  current position and velocity:

  ```python
  initial_state_prior = tfd.JointDistributionNamed({
      'position': tfd.Normal(loc=0., scale=1.),
      'velocity': tfd.Normal(loc=0., scale=0.1)})
  ```

  The `transition_fn` specifies the evolution of the system. It should
  return a distribution over latent states of the same structure as the prior.
  Here, we'll assume that the position evolves according to the velocity,
  with a small random drift, and the velocity also changes slowly, following
  a random drift:

  ```python
  def transition_fn(_, previous_state):
    return tfd.JointDistributionNamed({
        'position': tfd.Normal(
            loc=previous_state['position'] + previous_state['velocity'],
            scale=0.1),
        'velocity': tfd.Normal(loc=previous_state['velocity'], scale=0.01)})
  ```

  The `observation_fn` specifies the process by which the system is observed
  at each time step. Let's suppose we observe only a noisy version of the =
  current position.

  ```python
    def observation_fn(_, state):
      return tfd.Normal(loc=state['position'], scale=0.1)
  ```

  Now let's track our object. Suppose we've been given observations
  corresponding to an initial position of `0.4` and constant velocity of `0.01`:

  ```python
  # Generate simulated observations.
  observed_positions = tfd.Normal(loc=tf.linspace(0.4, 0.8, 0.01),
                                  scale=0.1).sample()

  # Run particle filtering.
  (particles,       # {'position': [40, 1000], 'velocity': [40, 1000]}
   parent_indices,  #  [40, 1000]
   _) = tfp.experimental.mcmc.particle_filter(
          observations=observed_positions,
          initial_state_prior=initial_state_prior,
          transition_fn=transition_fn,
          observation_fn=observation_fn,
          num_particles=1000)
  ```

  The particle filter samples from the "filtering" distribution over latent
  states: at each point in time, this is the distribution conditioned on all
  observations *up to that time*. For example,
  `particles['position'][t]` contains `num_particles` samples from the
  distribution `p(position[t] | observed_positions[:t])`. Because
  particles may be resampled, there is no relationship between a particle
  at time `t` and the particle with the same index at time `t + 1`.

  We may, however, trace back through the resampling steps to reconstruct
  samples of entire latent trajectories.

  ```python
  trajectories = tfp.experimental.mcmc.reconstruct_trajectories(
       particles, parent_indices)
  ```

  Here, `trajectories['position'][:, i]` contains the history of positions
  sampled for what became the `i`th particle at the final timestep. These
  are samples from the 'smoothed' posterior over trajectories, given all
  observations: `p(position[0:T] | observed_position[0:T])`.

  """
  seed = SeedStream(seed, 'particle_filter')
  with tf.name_scope(name or 'particle_filter'):
    num_timesteps = prefer_static.shape(
        tf.nest.flatten(observations)[0])[0]

    # Dress up the prior and prior proposal as a fake `transition_fn` and
    # `proposal_fn` respectively.
    prior_fn = lambda _1, _2: SampleParticles(  # pylint: disable=g-long-lambda
        initial_state_prior, num_particles)
    prior_proposal_fn = (
        None if initial_state_proposal is None
        else lambda _1, _2: SampleParticles(  # pylint: disable=g-long-lambda
            initial_state_proposal, num_particles))

    # Initialize from the prior, and incorporate the first observation.
    (initial_resampled_particles,
     initial_parent_indices,
     initial_step_log_marginal_likelihood) = _filter_one_step(
         step=0,
         # `previous_particles` at the first step is a dummy quantity, used only
         # to convey state structure and num_particles to an optional
         # proposal fn.
         previous_particles=prior_fn(0, []).sample(),
         observation=tf.nest.map_structure(
             lambda x: tf.gather(x, 0), observations),
         transition_fn=prior_fn,
         observation_fn=observation_fn,
         proposal_fn=prior_proposal_fn,
         seed=seed())

    loop_vars = _initialize_accumulated_quantities(
        initial_resampled_particles,
        initial_parent_indices,
        initial_step_log_marginal_likelihood,
        num_steps_state_history_to_pass,
        num_timesteps)

    def _loop_body(step, resampled_particles, accumulated_quantities):
      """Take one step in dynamics and accumulate marginal likelihood."""

      current_observation = tf.nest.map_structure(
          lambda x, step=step: tf.gather(x, step), observations)

      history_to_pass_into_fns = {}
      if num_steps_observation_history_to_pass:
        history_to_pass_into_fns['observation_history'] = _gather_history(
            observations, step, num_steps_observation_history_to_pass)
      if num_steps_state_history_to_pass:
        history_to_pass_into_fns['state_history'] = (
            accumulated_quantities.state_history)

      (resampled_particles,
       parent_indices,
       step_log_marginal_likelihood) = _filter_one_step(
           step=step,
           previous_particles=resampled_particles,
           observation=current_observation,
           transition_fn=functools.partial(
               transition_fn, **history_to_pass_into_fns),
           observation_fn=functools.partial(
               observation_fn, **history_to_pass_into_fns),
           proposal_fn=(
               None if proposal_fn is None else
               functools.partial(proposal_fn, **history_to_pass_into_fns)),
           seed=seed())

      new_accumulated_quantities = _write_accumulated_quantities(
          step,
          accumulated_quantities,
          resampled_particles,
          parent_indices,
          step_log_marginal_likelihood)

      return step + 1, resampled_particles, new_accumulated_quantities

    (_,
     _,
     loop_results) = tf.while_loop(
         cond=lambda step, *_: step < num_timesteps,
         body=_loop_body,
         loop_vars=(1, initial_resampled_particles, loop_vars))

    return (tf.nest.map_structure(lambda ta: ta.stack(),
                                  loop_results.all_resampled_particles),
            loop_results.all_parent_indices.stack(),
            loop_results.all_step_log_marginal_likelihoods.stack())


def _initialize_accumulated_quantities(initial_resampled_particles,
                                       initial_parent_indices,
                                       initial_step_log_marginal_likelihood,
                                       num_steps_state_history_to_pass,
                                       num_timesteps):
  """Initialize arrays and other quantities passed through the filter loop."""

  # Create arrays to store particles, indices, and likelihoods, and write
  # their initial values.
  all_resampled_particles = tf.nest.map_structure(
      lambda x: tf.TensorArray(dtype=x.dtype, size=num_timesteps).write(0, x),
      initial_resampled_particles)
  all_parent_indices = tf.TensorArray(
      dtype=tf.int32, size=num_timesteps).write(0, initial_parent_indices)
  all_step_log_marginal_likelihoods = tf.TensorArray(
      dtype=initial_step_log_marginal_likelihood.dtype,
      size=num_timesteps).write(0, initial_step_log_marginal_likelihood)

  # Because `while_loop` requires Tensor values, we'll represent the lack of
  # state history by a static-shape empty Tensor.
  # This can be detected elsewhere by branching on
  #  `tf.is_tensor(state_history) and state_history.shape[0] == 0`.
  state_history = tf.zeros([0])
  if num_steps_state_history_to_pass:
    # Repeat the initial state, so that `state_history` always has length
    # `num_steps_state_history_to_pass`.
    state_history = tf.nest.map_structure(
        lambda x: tf.broadcast_to(  # pylint: disable=g-long-lambda
            x[tf.newaxis, ...],
            prefer_static.concat([[num_steps_state_history_to_pass],
                                  prefer_static.shape(x)], axis=0)),
        initial_resampled_particles)

  return ParticleFilterAccumulatedQuantities(
      all_resampled_particles=all_resampled_particles,
      all_parent_indices=all_parent_indices,
      all_step_log_marginal_likelihoods=all_step_log_marginal_likelihoods,
      state_history=state_history)


def _write_accumulated_quantities(step,
                                  accumulated_quantities,
                                  resampled_particles,
                                  parent_indices,
                                  step_log_marginal_likelihood):
  """Update the loop state to reflect a step of filtering."""

  # Create a new dict instead of modifying in place, for cleanliness.

  # Write particles, indices, and likelihoods to their respective arrays.
  all_resampled_particles = tf.nest.map_structure(
      lambda x, y: x.write(step, y),
      accumulated_quantities.all_resampled_particles,
      resampled_particles)
  all_parent_indices = accumulated_quantities.all_parent_indices.write(
      step, parent_indices)
  all_step_log_marginal_likelihoods = (
      accumulated_quantities.all_step_log_marginal_likelihoods.write(
          step, step_log_marginal_likelihood))

  state_history = accumulated_quantities.state_history
  history_is_empty = (tf.is_tensor(state_history) and
                      state_history.shape[0] == 0)
  if not history_is_empty:
    batch_shape = prefer_static.shape(parent_indices)[1:-1]
    batch_rank = prefer_static.rank_from_shape(batch_shape)

    # Permute the particles from previous steps to match the current resampled
    # indices, so that the state history reflects coherent trajectories.
    def update_state_history_to_use_current_indices(x):
      return tf.gather(x[-1:],
                       parent_indices,
                       axis=batch_rank + 1,
                       batch_dims=batch_rank)
    resampled_state_history = tf.nest.map_structure(
        update_state_history_to_use_current_indices,
        accumulated_quantities.state_history)

    # Update the history by concat'ing the carried-forward elements with the
    # most recent state.
    state_history = tf.nest.map_structure(
        lambda h, s: tf.concat([h, s[tf.newaxis, ...]], axis=0),
        resampled_state_history, resampled_particles)

  return ParticleFilterAccumulatedQuantities(
      all_resampled_particles=all_resampled_particles,
      all_parent_indices=all_parent_indices,
      all_step_log_marginal_likelihoods=all_step_log_marginal_likelihoods,
      state_history=state_history)


def _filter_one_step(step,
                     observation,
                     previous_particles,
                     transition_fn,
                     observation_fn,
                     proposal_fn,
                     seed=None):
  """Advances the particle filter by a single time step."""
  with tf.name_scope('filter_one_step'):
    seed = SeedStream(seed, 'filter_one_step')

    proposed_particles, proposal_log_weights = _propose_with_log_weights(
        step=step - 1,
        particles=previous_particles,
        transition_fn=transition_fn,
        proposal_fn=proposal_fn,
        seed=seed())

    observation_log_weights = _compute_observation_log_weights(
        step, proposed_particles, observation, observation_fn)
    log_weights = proposal_log_weights + observation_log_weights

    resampled_particles, resample_indices = _resample(
        proposed_particles, log_weights, seed=seed())

    step_log_marginal_likelihood = tfp_math.reduce_logmeanexp(
        log_weights, axis=-1)

  return resampled_particles, resample_indices, step_log_marginal_likelihood


def _propose_with_log_weights(step,
                              particles,
                              transition_fn,
                              proposal_fn=None,
                              seed=None):
  """Proposes a new batch of particles with importance weights.

  Args:
    step: int `Tensor` current step.
    particles: Nested structure of `Tensor`s each of shape
      `[b1, ..., bN, num_particles, latent_part_event_shape]`, where
      `b1, ..., bN` are optional batch dimensions.
    transition_fn: callable, producing a distribution over `particles`
      at the next step.
    proposal_fn: callable, producing a distribution over `particles`
      at the next step.
      Default value: `None`.
    seed: Python `int` random seed.
      Default value: `None`.
  Returns:
    proposed_particles: Nested structure of `Tensor`s, matching `particles`.
    proposal_log_weights:  `Tensor` of shape
      `concat([[b1, ..., bN], [num_particles]])`.
  """
  with tf.name_scope('propose_with_log_weights'):
    transition_dist = transition_fn(step, particles)
    # If no proposal was specified, use the dynamics.
    if proposal_fn is None:
      return transition_dist.sample(seed=seed), 0.0

    proposal_dist = proposal_fn(step, particles)
    proposed_particles = proposal_dist.sample(seed=seed)
    proposal_log_weights = (
        transition_dist.log_prob(proposed_particles) -
        proposal_dist.log_prob(proposed_particles))
    return proposed_particles, proposal_log_weights


def _compute_observation_log_weights(step,
                                     particles,
                                     observation,
                                     observation_fn):
  """Computes particle importance weights from an observation step.

  Args:
    step: int `Tensor` current step.
    particles: Nested structure of `Tensor`s, each of shape
      `concat([[b1, ..., bN], [num_particles], latent_part_event_shape])`, where
      `b1, ..., bN` are optional batch dimensions.
    observation: Nested structure of `Tensor`s, each of shape
      `concat([[b1, ..., bN], observation_part_event_shape])` where
      `b1, ..., bN` are optional batch dimensions.
    observation_fn: callable, producing a distribution over `observation`s.
  Returns:
    log_weights: `Tensor` of shape `concat([[b1, ..., bN], [num_particles]])`.
  """
  with tf.name_scope('compute_observation_log_weights'):
    observation_dist = observation_fn(step, particles)
    def _add_right_batch_dim(obs, event_shape):
      ndims = prefer_static.rank_from_shape(prefer_static.shape(obs))
      event_ndims = prefer_static.rank_from_shape(event_shape)
      return tf.expand_dims(obs, ndims - event_ndims)
    observation_broadcast_over_particles = tf.nest.map_structure(
        _add_right_batch_dim,
        observation,
        observation_dist.event_shape_tensor())
    return observation_dist.log_prob(observation_broadcast_over_particles)


def _resample(particles, log_weights, seed=None):
  """Resamples the current particles according to provided weights.

  Args:
    particles: Nested structure of `Tensor`s each of shape
      `[b1, ..., bN, num_particles, ...]`, where
      `b1, ..., bN` are optional batch dimensions.
    log_weights: float `Tensor` of shape `[b1, ..., bN, num_particles]`, where
      `b1, ..., bN` are optional batch dimensions.
    seed: Python `int` random seed.
  Returns:
    resampled_particles: Nested structure of `Tensor`s, matching `particles`.
  """
  with tf.name_scope('resample'):
    weights_shape = prefer_static.shape(log_weights)
    batch_shape, num_particles = weights_shape[:-1], weights_shape[-1]
    batch_rank = prefer_static.rank_from_shape(batch_shape)

    resample_indices = dist_util.move_dimension(
        categorical.Categorical(log_weights).sample(num_particles, seed=seed),
        0, -1)
    resampled_particles = tf.nest.map_structure(
        lambda x: tf.gather(  # pylint: disable=g-long-lambda
            x, resample_indices, axis=batch_rank, batch_dims=batch_rank),
        particles)
  return resampled_particles, resample_indices


def reconstruct_trajectories(particles, parent_indices, name=None):
  """Reconstructs the ancestor trajectory that generated each final particle."""
  with tf.name_scope(name or 'reconstruct_trajectories'):
    indices_shape = prefer_static.shape(parent_indices)
    batch_shape, num_trajectories = indices_shape[1:-1], indices_shape[-1]
    batch_rank = prefer_static.rank_from_shape(batch_shape)

    # Walk backwards to compute the ancestor of each final particle at time t.
    final_indices = tf.broadcast_to(
        tf.range(0, num_trajectories), indices_shape[1:])
    ancestor_indices = tf.scan(
        fn=lambda ancestor, parent: tf.gather(  # pylint: disable=g-long-lambda
            parent, ancestor, axis=batch_rank, batch_dims=batch_rank),
        elems=parent_indices[1:],
        initializer=final_indices,
        reverse=True)
    ancestor_indices = tf.concat([ancestor_indices, [final_indices]], axis=0)

  return tf.nest.map_structure(
      lambda part: tf.gather(part, ancestor_indices,  # pylint: disable=g-long-lambda
                             axis=batch_rank + 1, batch_dims=batch_rank + 1),
      particles)
