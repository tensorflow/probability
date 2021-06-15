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

from __future__ import print_function

import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow_probability.python.experimental.mcmc import sequential_monte_carlo_kernel as smc_kernel
from tensorflow_probability.python.experimental.mcmc import weighted_resampling
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import docstring_util
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.mcmc.internal import util as mcmc_util

__all__ = [
    'infer_trajectories',
    'particle_filter',
    'reconstruct_trajectories',
]


# Default trace criterion.
_always_trace = lambda *_: True


def _default_trace_fn(state, kernel_results):
  return (state.particles,
          state.log_weights,
          kernel_results.parent_indices,
          kernel_results.incremental_log_marginal_likelihood)


particle_filter_arg_str = """\
Each latent state is a `Tensor` or nested structure of `Tensor`s, as defined
by the `initial_state_prior`.

The `transition_fn` and `proposal_fn` args, if specified, have signature
`next_state_dist = fn(step, state)`, where `step` is an `int` `Tensor` index
of the current time step (beginning at zero), and `state` represents
the latent state at time `step`. The return value is a `tfd.Distribution`
instance over the state at time `step + 1`.

Similarly, the `observation_fn` has signature
`observation_dist = observation_fn(step, state)`, where the return value
is a distribution over the value(s) observed at time `step`.

Args:
  observations: a (structure of) Tensors, each of shape
    `concat([[num_observation_steps, b1, ..., bN], event_shape])` with
    optional batch dimensions `b1, ..., bN`.
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
  resample_fn: Python `callable` to generate the indices of resampled
    particles, given their weights. Generally, one of
    `tfp.experimental.mcmc.resample_independent` or
    `tfp.experimental.mcmc.resample_systematic`, or any function
    with the same signature, `resampled_indices = f(log_probs, event_size, '
    'sample_shape, seed)`.
    Default: `tfp.experimental.mcmc.resample_systematic`.
  resample_criterion_fn: optional Python `callable` with signature
    `do_resample = resample_criterion_fn(log_weights)`,
    where `log_weights` is a float `Tensor` of shape
    `[b1, ..., bN, num_particles]` containing log (unnormalized) weights for
    all particles at the current step. The return value `do_resample`
    determines whether particles are resampled at the current step. In the
    case `resample_criterion_fn==None`, particles are resampled at every step.
    The default behavior resamples particles when the current effective
    sample size falls below half the total number of particles.
    Default value: `tfp.experimental.mcmc.ess_below_threshold`.
  rejuvenation_kernel_fn: optional Python `callable` with signature
    `transition_kernel = rejuvenation_kernel_fn(target_log_prob_fn)`
    where `target_log_prob_fn` is a provided callable evaluating
    `p(x[t] | y[t], x[t-1])` at each step `t`, and `transition_kernel`
    should be an instance of `tfp.mcmc.TransitionKernel`.
    Default value: `None`.  # TODO(davmre): not yet supported.
  num_transitions_per_observation: scalar Tensor positive `int` number of
    state transitions between regular observation points. A value of `1`
    indicates that there is an observation at every timestep,
    `2` that every other step is observed, and so on. Values greater than `1`
    may be used with an appropriately-chosen transition function to
    approximate continuous-time dynamics. The initial and final steps
    (steps `0` and `num_timesteps - 1`) are always observed.
    Default value: `None`.
"""


@docstring_util.expand_docstring(
    particle_filter_arg_str=particle_filter_arg_str)
def infer_trajectories(observations,
                       initial_state_prior,
                       transition_fn,
                       observation_fn,
                       num_particles,
                       initial_state_proposal=None,
                       proposal_fn=None,
                       resample_fn=weighted_resampling.resample_systematic,
                       resample_criterion_fn=smc_kernel.ess_below_threshold,
                       rejuvenation_kernel_fn=None,
                       num_transitions_per_observation=1,
                       seed=None,
                       name=None):  # pylint: disable=g-doc-args
  """Use particle filtering to sample from the posterior over trajectories.

  ${particle_filter_arg_str}
    seed: Optional seed, for reproducible results.
    name: Python `str` name for ops created by this method.
      Default value: `None` (i.e., `'infer_trajectories'`).
  Returns:
    trajectories: a (structure of) Tensor(s) matching the latent state, each
      of shape
      `concat([[num_timesteps, num_particles, b1, ..., bN], event_shape])`,
      representing unbiased samples from the posterior distribution
      `p(latent_states | observations)`.
    incremental_log_marginal_likelihoods: float `Tensor` of shape
      `[num_observation_steps, b1, ..., bN]`,
      giving the natural logarithm of an unbiased estimate of
      `p(observations[t] | observations[:t])` at each timestep `t`. Note that
      (by [Jensen's inequality](
      https://en.wikipedia.org/wiki/Jensen%27s_inequality))
      this is *smaller* in expectation than the true
      `log p(observations[t] | observations[:t])`.

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

  # Run particle filtering to sample plausible trajectories.
  (trajectories,  # {'position': [40, 1000], 'velocity': [40, 1000]}
   lps) = tfp.experimental.mcmc.infer_trajectories(
            observations=observed_positions,
            initial_state_prior=initial_state_prior,
            transition_fn=transition_fn,
            observation_fn=observation_fn,
            num_particles=1000)
  ```

  For all `i`, `trajectories['position'][:, i]` is a sample from the
  posterior over position sequences, given the observations:
  `p(state[0:T] | observations[0:T])`. Often, the sampled trajectories
  will be highly redundant in their earlier timesteps, because most
  of the initial particles have been discarded through resampling
  (this problem is known as 'particle degeneracy'; see section 3.5 of
  [Doucet and Johansen][1]).
  In such cases it may be useful to also consider the series of *filtering*
  distributions `p(state[t] | observations[:t])`, in which each latent state
  is inferred conditioned only on observations up to that point in time; these
  may be computed using `tfp.mcmc.experimental.particle_filter`.

  #### References

  [1] Arnaud Doucet and Adam M. Johansen. A tutorial on particle
      filtering and smoothing: Fifteen years later.
      _Handbook of nonlinear filtering_, 12(656-704), 2009.
      https://www.stats.ox.ac.uk/~doucet/doucet_johansen_tutorialPF2011.pdf

  """
  with tf.name_scope(name or 'infer_trajectories') as name:
    pf_seed, resample_seed = samplers.split_seed(
        seed, salt='infer_trajectories')
    (particles,
     log_weights,
     parent_indices,
     incremental_log_marginal_likelihoods) = particle_filter(
         observations=observations,
         initial_state_prior=initial_state_prior,
         transition_fn=transition_fn,
         observation_fn=observation_fn,
         num_particles=num_particles,
         initial_state_proposal=initial_state_proposal,
         proposal_fn=proposal_fn,
         resample_fn=resample_fn,
         resample_criterion_fn=resample_criterion_fn,
         rejuvenation_kernel_fn=rejuvenation_kernel_fn,
         num_transitions_per_observation=num_transitions_per_observation,
         trace_fn=_default_trace_fn,
         trace_criterion_fn=lambda *_: True,
         seed=pf_seed,
         name=name)
    weighted_trajectories = reconstruct_trajectories(particles, parent_indices)

    # Resample all steps of the trajectories using the final weights.
    resample_indices = resample_fn(log_probs=log_weights[-1],
                                   event_size=num_particles,
                                   sample_shape=(),
                                   seed=resample_seed)
    trajectories = tf.nest.map_structure(
        lambda x: mcmc_util.index_remapping_gather(x,  # pylint: disable=g-long-lambda
                                                   resample_indices,
                                                   axis=1),
        weighted_trajectories)

    return trajectories, incremental_log_marginal_likelihoods


@docstring_util.expand_docstring(
    particle_filter_arg_str=particle_filter_arg_str)
def particle_filter(observations,
                    initial_state_prior,
                    transition_fn,
                    observation_fn,
                    num_particles,
                    initial_state_proposal=None,
                    proposal_fn=None,
                    resample_fn=weighted_resampling.resample_systematic,
                    resample_criterion_fn=smc_kernel.ess_below_threshold,
                    rejuvenation_kernel_fn=None,  # TODO(davmre): not yet supported. pylint: disable=unused-argument
                    num_transitions_per_observation=1,
                    trace_fn=_default_trace_fn,
                    trace_criterion_fn=_always_trace,
                    static_trace_allocation_size=None,
                    parallel_iterations=1,
                    seed=None,
                    name=None):  # pylint: disable=g-doc-args
  """Samples a series of particles representing filtered latent states.

  The particle filter samples from the sequence of "filtering" distributions
  `p(state[t] | observations[:t])` over latent
  states: at each point in time, this is the distribution conditioned on all
  observations *up to that time*. Because particles may be resampled, a particle
  at time `t` may be different from the particle with the same index at time
  `t + 1`. To reconstruct trajectories by tracing back through the resampling
  process, see `tfp.mcmc.experimental.reconstruct_trajectories`.

  ${particle_filter_arg_str}
    trace_fn: Python `callable` defining the values to be traced at each step,
      with signature `traced_values = trace_fn(weighted_particles, results)`
      in which the first argument is an instance of
      `tfp.experimental.mcmc.WeightedParticles` and the second an instance of
      `SequentialMonteCarloResults` tuple, and the return value is a structure
      of `Tensor`s.
      Default value: `lambda s, r: (s.particles, s.log_weights,
      r.parent_indices, r.incremental_log_marginal_likelihood)`
    trace_criterion_fn: optional Python `callable` with signature
      `trace_this_step = trace_criterion_fn(weighted_particles, results)` taking
      the same arguments as `trace_fn` and returning a boolean `Tensor`. If
      `None`, only values from the final step are returned.
      Default value: `lambda *_: True` (trace every step).
    static_trace_allocation_size: Optional Python `int` size of trace to
      allocate statically. This should be an upper bound on the number of steps
      traced and is used only when the length cannot be
      statically inferred (for example, if a `trace_criterion_fn` is specified).
      It is primarily intended for contexts where static shapes are required,
      such as in XLA-compiled code.
      Default value: `None`.
    parallel_iterations: Passed to the internal `tf.while_loop`.
      Default value: `1`.
    seed: Optional seed for reproducible results.
    name: Python `str` name for ops created by this method.
      Default value: `None` (i.e., `'particle_filter'`).
  Returns:
    traced_results: A structure of Tensors as returned by `trace_fn`. If
      `trace_criterion_fn==None`, this is computed from the final step;
      otherwise, each Tensor will have initial dimension `num_steps_traced`
      and stacks the traced results across all steps.
  """

  init_seed, loop_seed = samplers.split_seed(seed, salt='particle_filter')
  with tf.name_scope(name or 'particle_filter'):
    num_observation_steps = ps.size0(tf.nest.flatten(observations)[0])
    num_timesteps = (
        1 + num_transitions_per_observation * (num_observation_steps - 1))

    # If trace criterion is `None`, we'll return only the final results.
    never_trace = lambda *_: False
    if trace_criterion_fn is None:
      static_trace_allocation_size = 0
      trace_criterion_fn = never_trace

    initial_weighted_particles = _particle_filter_initial_weighted_particles(
        observations=observations,
        observation_fn=observation_fn,
        initial_state_prior=initial_state_prior,
        initial_state_proposal=initial_state_proposal,
        num_particles=num_particles,
        seed=init_seed)
    propose_and_update_log_weights_fn = (
        _particle_filter_propose_and_update_log_weights_fn(
            observations=observations,
            transition_fn=transition_fn,
            proposal_fn=proposal_fn,
            observation_fn=observation_fn,
            num_transitions_per_observation=num_transitions_per_observation))

    kernel = smc_kernel.SequentialMonteCarlo(
        propose_and_update_log_weights_fn=propose_and_update_log_weights_fn,
        resample_fn=resample_fn,
        resample_criterion_fn=resample_criterion_fn)

    # Use `trace_scan` rather than `sample_chain` directly because the latter
    # would force us to trace the state history (with or without thinning),
    # which is not always appropriate.
    def seeded_one_step(seed_state_results, _):
      seed, state, results = seed_state_results
      one_step_seed, next_seed = samplers.split_seed(seed)
      next_state, next_results = kernel.one_step(
          state, results, seed=one_step_seed)
      return next_seed, next_state, next_results

    final_seed_state_result, traced_results = mcmc_util.trace_scan(
        loop_fn=seeded_one_step,
        initial_state=(loop_seed,
                       initial_weighted_particles,
                       kernel.bootstrap_results(initial_weighted_particles)),
        elems=tf.ones([num_timesteps]),
        trace_fn=lambda seed_state_results: trace_fn(*seed_state_results[1:]),
        trace_criterion_fn=(
            lambda seed_state_results: trace_criterion_fn(  # pylint: disable=g-long-lambda
                *seed_state_results[1:])),
        static_trace_allocation_size=static_trace_allocation_size,
        parallel_iterations=parallel_iterations)

    if trace_criterion_fn is never_trace:
      # Return results from just the final step.
      traced_results = trace_fn(*final_seed_state_result[1:])

    return traced_results


def _particle_filter_initial_weighted_particles(observations,
                                                observation_fn,
                                                initial_state_prior,
                                                initial_state_proposal,
                                                num_particles,
                                                seed=None):
  """Initialize a set of weighted particles including the first observation."""
  # Propose an initial state.
  if initial_state_proposal is None:
    initial_state = initial_state_prior.sample(num_particles, seed=seed)
    initial_log_weights = ps.zeros_like(
        initial_state_prior.log_prob(initial_state))
  else:
    initial_state = initial_state_proposal.sample(num_particles, seed=seed)
    initial_log_weights = (initial_state_prior.log_prob(initial_state) -
                           initial_state_proposal.log_prob(initial_state))
  # Normalize the initial weights. If we used a proposal, the weights are
  # normalized in expectation, but actually normalizing them reduces variance.
  initial_log_weights = tf.nn.log_softmax(initial_log_weights, axis=0)

  # Return particles weighted by the initial observation.
  return smc_kernel.WeightedParticles(
      particles=initial_state,
      log_weights=initial_log_weights + _compute_observation_log_weights(
          step=0,
          particles=initial_state,
          observations=observations,
          observation_fn=observation_fn))


def _particle_filter_propose_and_update_log_weights_fn(
    observations,
    transition_fn,
    proposal_fn,
    observation_fn,
    num_transitions_per_observation=1):
  """Build a function specifying a particle filter update step."""
  def propose_and_update_log_weights_fn(step, state, seed=None):
    particles, log_weights = state.particles, state.log_weights
    transition_dist = transition_fn(step, particles)
    assertions = _assert_batch_shape_matches_weights(
        distribution=transition_dist,
        weights_shape=ps.shape(log_weights),
        diststr='transition')

    if proposal_fn:
      proposal_dist = proposal_fn(step, particles)
      assertions += _assert_batch_shape_matches_weights(
          distribution=proposal_dist,
          weights_shape=ps.shape(log_weights),
          diststr='proposal')
      proposed_particles = proposal_dist.sample(seed=seed)

      log_weights += (transition_dist.log_prob(proposed_particles) -
                      proposal_dist.log_prob(proposed_particles))
      # The normalizing constant E~q[p(x)/q(x)] is 1 in expectation,
      # so we reduce variance by dividing it out. Intuitively: the marginal
      # likelihood of a model with no observations is constant
      # (equal to 1.), so the transition and proposal distributions shouldn't
      # affect it.
      log_weights = tf.nn.log_softmax(log_weights, axis=0)
    else:
      proposed_particles = transition_dist.sample(seed=seed)

    with tf.control_dependencies(assertions):
      return smc_kernel.WeightedParticles(
          particles=proposed_particles,
          log_weights=log_weights + _compute_observation_log_weights(
              step + 1, proposed_particles, observations, observation_fn,
              num_transitions_per_observation=num_transitions_per_observation))
  return propose_and_update_log_weights_fn


def _compute_observation_log_weights(step,
                                     particles,
                                     observations,
                                     observation_fn,
                                     num_transitions_per_observation=1):
  """Computes particle importance weights from an observation step.

  Args:
    step: int `Tensor` current step.
    particles: Nested structure of `Tensor`s, each of shape
      `concat([[num_particles, b1, ..., bN], event_shape])`, where
      `b1, ..., bN` are optional batch dimensions and `event_shape` may
      differ across `Tensor`s.
    observations: Nested structure of `Tensor`s, each of shape
      `concat([[num_observations, b1, ..., bN], event_shape])`
      where `b1, ..., bN` are optional batch dimensions and `event_shape` may
      differ across `Tensor`s.
    observation_fn: callable with signature
      `observation_dist = observation_fn(step, particles)`, producing
      a batch of distributions over the `observation` at the given `step`,
      one for each particle.
    num_transitions_per_observation: optional int `Tensor` number of times
      to apply the transition model between successive observation steps.
      Default value: `1`.
  Returns:
    log_weights: `Tensor` of shape `concat([num_particles, b1, ..., bN])`.
  """
  with tf.name_scope('compute_observation_log_weights'):

    step_has_observation = (
        # The second of these conditions subsumes the first, but both are
        # useful because the first can often be evaluated statically.
        ps.equal(num_transitions_per_observation, 1) |
        ps.equal(step % num_transitions_per_observation, 0))
    observation_idx = step // num_transitions_per_observation
    observation = tf.nest.map_structure(
        lambda x, step=step: tf.gather(x, observation_idx), observations)

    log_weights = observation_fn(step, particles).log_prob(observation)
    return ps.where(step_has_observation,
                    log_weights,
                    tf.zeros_like(log_weights))


def reconstruct_trajectories(particles, parent_indices, name=None):
  """Reconstructs the ancestor trajectory that generated each final particle."""
  with tf.name_scope(name or 'reconstruct_trajectories'):
    # Walk backwards to compute the ancestor of each final particle at time t.
    final_indices = smc_kernel._dummy_indices_like(parent_indices[-1])  # pylint: disable=protected-access
    ancestor_indices = tf.scan(
        fn=lambda ancestor, parent: mcmc_util.index_remapping_gather(  # pylint: disable=g-long-lambda
            parent, ancestor, axis=0),
        elems=parent_indices[1:],
        initializer=final_indices,
        reverse=True)
    ancestor_indices = tf.concat([ancestor_indices, [final_indices]], axis=0)

  return tf.nest.map_structure(
      lambda part: mcmc_util.index_remapping_gather(  # pylint: disable=g-long-lambda
          part, ancestor_indices, axis=1, indices_axis=1),
      particles)


def _assert_batch_shape_matches_weights(distribution, weights_shape, diststr):
  """Checks that all parts of a distribution have the expected batch shape."""
  shapes = [weights_shape] + tf.nest.flatten(distribution.batch_shape_tensor())
  static_shapes = [tf.get_static_value(ps.convert_to_shape_tensor(s))
                   for s in shapes]
  static_shapes_not_none = [s for s in static_shapes if s is not None]
  static_shapes_match = all([
      np.all(a == b)  # Also need to check for rank mismatch (below).
      for (a, b) in zip(static_shapes_not_none[1:],
                        static_shapes_not_none[:-1])])

  # Build a separate list of static ranks, since rank is often static even when
  # shape is not.
  ranks = [ps.rank_from_shape(s) for s in shapes]
  static_ranks = [int(r) for r in ranks if not tf.is_tensor(r)]
  static_ranks_match = all([a == b for (a, b) in zip(static_ranks[1:],
                                                     static_ranks[:-1])])

  msg = (
      "The {diststr} distribution's batch shape does not match the particle "
      "weights; a correct {diststr} distribution must return an independent "
      "log-density for each particle. You may be "
      "creating a joint distribution in which some parts do not depend on the "
      "previous particles, and/or you are creating an autobatched joint "
      "distribution without setting `batch_ndims`.".format(
          diststr=diststr))
  if not (static_ranks_match and static_shapes_match):
    raise ValueError(msg + ' ' +
                     'Weights have shape {}, but the distribution has batch '
                     'shape {}.'.format(
                         weights_shape, distribution.batch_shape))

  assertions = []
  if distribution.validate_args and any([s is None for s in static_shapes]):
    assertions = [assert_util.assert_equal(a, b, message=msg)
                  for a, b in zip(shapes[1:], shapes[:-1])]
  return assertions
