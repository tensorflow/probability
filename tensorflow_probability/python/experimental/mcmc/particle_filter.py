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

import collections
import functools

import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow_probability.python.distributions import distribution as distribution_lib
from tensorflow_probability.python.experimental.mcmc import weighted_resampling
from tensorflow_probability.python.internal import distribution_util as dist_util
from tensorflow_probability.python.internal import docstring_util
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.mcmc.internal import util as mcmc_util
from tensorflow_probability.python.util import SeedStream

__all__ = [
    'ess_below_threshold',
    'infer_trajectories',
    'particle_filter',
    'reconstruct_trajectories',
]


# TODO(b/153467570): Move SampleParticles into `tfp.distributions`.
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
        lambda b: tensorshape_util.concatenate(  # pylint: disable=g-long-lambda
            [tf.get_static_value(self.num_particles)], b),
        self.distribution.batch_shape)

  def _batch_shape_tensor(self, **kwargs):
    return tf.nest.map_structure(
        lambda b: ps.concat([[self.num_particles], b], axis=0),
        self.distribution.batch_shape_tensor(**kwargs))

  def _log_prob(self, x, **kwargs):
    return self._call_log_measure('_log_prob', x, kwargs)

  def _log_cdf(self, x, **kwargs):
    return self._call_log_measure('_log_cdf', x, kwargs)

  def _log_sf(self, x, **kwargs):
    return self._call_log_measure('_log_sf', x, kwargs)

  def _call_log_measure(self, attr, x, kwargs):
    return getattr(self.distribution, attr)(x, **kwargs)

  # TODO(b/152797117): Override _sample_n, once it supports joint distributions.
  def sample(self, sample_shape=(), seed=None, name=None):
    with tf.name_scope(name or 'sample_particles'):
      sample_shape = ps.concat([
          dist_util.expand_to_vector(sample_shape),
          [self.num_particles]], axis=0)
      return self.distribution.sample(sample_shape, seed=seed)


def _dummy_indices_like(indices):
  """Returns dummy indices ([0, 1, 2, ...]) with batch shape like `indices`."""
  indices_shape = ps.shape(indices)
  num_particles = indices_shape[0]
  return tf.broadcast_to(
      ps.reshape(
          ps.range(num_particles),
          ps.concat([[num_particles],
                     ps.ones([ps.rank_from_shape(indices_shape) - 1],
                             dtype=np.int32)],
                    axis=0)),
      indices_shape)


def ess_below_threshold(unnormalized_log_weights, threshold=0.5):
  """Determines if the effective sample size is much less than num_particles."""
  with tf.name_scope('ess_below_threshold'):
    num_particles = ps.size0(unnormalized_log_weights)
    log_weights = tf.math.log_softmax(unnormalized_log_weights, axis=0)
    log_ess = -tf.math.reduce_logsumexp(2 * log_weights, axis=0)
    return log_ess < (ps.log(num_particles) +
                      ps.log(threshold))


ParticleFilterStepResults = collections.namedtuple(
    'ParticleFilterStepResults',
    ['particles',
     'log_weights',
     'parent_indices',
     'incremental_log_marginal_likelihood',
     # Track both incremental and accumulated likelihoods because they're cheap,
     # and this allows users to get the accumulated likelihood without needing
     # to trace every step.
     'accumulated_log_marginal_likelihood'
    ])


def _default_trace_fn(results):
  return (results.particles,
          results.log_weights,
          results.parent_indices,
          results.incremental_log_marginal_likelihood)

ParticleFilterLoopVariables = collections.namedtuple(
    'ParticleFilterLoopVariables',
    ['step',
     'previous_step_results',
     'accumulated_traced_results',
     'num_steps_traced'
    ])


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
    sample size falls below half the total number of particles. Note that
    the resampling criterion is not used at the final step---there, particles
    are always resampled, so that we return unweighted values.
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
                       resample_criterion_fn=ess_below_threshold,
                       rejuvenation_kernel_fn=None,
                       num_transitions_per_observation=1,
                       seed=None,
                       name=None):  # pylint: disable=g-doc-args
  """Use particle filtering to sample from the posterior over trajectories.

  ${particle_filter_arg_str}
    seed: Python `int` seed for random ops.
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
    seed = SeedStream(seed, 'infer_trajectories')
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
         seed=seed,
         name=name)
    weighted_trajectories = reconstruct_trajectories(particles, parent_indices)

    # Resample all steps of the trajectories using the final weights.
    resample_indices = resample_fn(log_probs=log_weights[-1],
                                   event_size=num_particles,
                                   sample_shape=(),
                                   seed=seed)
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
                    resample_criterion_fn=ess_below_threshold,
                    rejuvenation_kernel_fn=None,  # TODO(davmre): not yet supported. pylint: disable=unused-argument
                    num_transitions_per_observation=1,
                    trace_fn=_default_trace_fn,
                    step_indices_to_trace=None,
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
    trace_fn: Python `callable` defining the values to be traced at each step.
      It takes a `ParticleFilterStepResults` tuple and returns a structure of
      `Tensor`s. The default function returns
      `(particles, log_weights, parent_indices, step_log_likelihood)`.
    step_indices_to_trace: optional `int` `Tensor` listing, in increasing order,
      the indices of steps at which to record the values traced by `trace_fn`.
      If `None`, the default behavior is to trace at every timestep,
      equivalent to specifying `step_indices_to_trace=tf.range(num_timsteps)`.
    seed: Python `int` seed for random ops.
    name: Python `str` name for ops created by this method.
      Default value: `None` (i.e., `'particle_filter'`).
  Returns:
    particles: a (structure of) Tensor(s) matching the latent state, each
      of shape
      `concat([[num_timesteps, num_particles, b1, ..., bN], event_shape])`,
      representing (possibly weighted) samples from the series of filtering
      distributions `p(latent_states[t] | observations[:t])`.
    log_weights: `float` `Tensor` of shape
      `[num_timesteps, num_particles, b1, ..., bN]`, such that
      `log_weights[t, :]` are the logarithms of normalized importance weights
      (such that `exp(reduce_logsumexp(log_weights), axis=-1) == 1.`) of
      the particles at time `t`. These may be used in conjunction with
      `particles` to compute expectations under the series of filtering
      distributions.
    parent_indices: `int` `Tensor` of shape
      `[num_timesteps, num_particles, b1, ..., bN]`,
      such that `parent_indices[t, k]` gives the index of the particle at
      time `t - 1` that the `k`th particle at time `t` is immediately descended
      from. See also
      `tfp.experimental.mcmc.reconstruct_trajectories`.
    incremental_log_marginal_likelihoods: float `Tensor` of shape
      `[num_observation_steps, b1, ..., bN]`,
      giving the natural logarithm of an unbiased estimate of
      `p(observations[t] | observations[:t])` at each observed timestep `t`.
      Note that (by [Jensen's inequality](
      https://en.wikipedia.org/wiki/Jensen%27s_inequality))
      this is *smaller* in expectation than the true
      `log p(observations[t] | observations[:t])`.

  """
  seed = SeedStream(seed, 'particle_filter')
  with tf.name_scope(name or 'particle_filter'):
    num_observation_steps = ps.size0(tf.nest.flatten(observations)[0])
    num_timesteps = (
        1 + num_transitions_per_observation * (num_observation_steps - 1))

    # If no criterion is specified, default is to resample at every step.
    if not resample_criterion_fn:
      resample_criterion_fn = lambda _: True

    # Canonicalize the list of steps to trace as a rank-1 tensor of (sorted)
    # positive integers. E.g., `3` -> `[3]`, `[-2, -1]` -> `[N - 2, N - 1]`.
    if step_indices_to_trace is not None:
      (step_indices_to_trace,
       traced_steps_have_rank_zero) = _canonicalize_steps_to_trace(
           step_indices_to_trace, num_timesteps)

    # Dress up the prior and prior proposal as a fake `transition_fn` and
    # `proposal_fn` respectively.
    prior_fn = lambda _1, _2: SampleParticles(  # pylint: disable=g-long-lambda
        initial_state_prior, num_particles)
    prior_proposal_fn = (
        None if initial_state_proposal is None
        else lambda _1, _2: SampleParticles(  # pylint: disable=g-long-lambda
            initial_state_proposal, num_particles))

    # Initially the particles all have the same weight, `1. / num_particles`.
    broadcast_batch_shape = tf.convert_to_tensor(
        functools.reduce(
            ps.broadcast_shape,
            tf.nest.flatten(initial_state_prior.batch_shape_tensor()),
            []), dtype=tf.int32)
    log_uniform_weights = ps.zeros(
        ps.concat([
            [num_particles],
            broadcast_batch_shape], axis=0),
        dtype=tf.float32) - ps.log(num_particles)

    # Initialize from the prior and incorporate the first observation.
    dummy_previous_step = ParticleFilterStepResults(
        particles=prior_fn(0, []).sample(),
        log_weights=log_uniform_weights,
        parent_indices=None,
        incremental_log_marginal_likelihood=0.,
        accumulated_log_marginal_likelihood=0.)
    initial_step_results = _filter_one_step(
        step=0,
        # `previous_particles` at the first step is a dummy quantity, used only
        # to convey state structure and num_particles to an optional
        # proposal fn.
        previous_step_results=dummy_previous_step,
        observation=tf.nest.map_structure(
            lambda x: tf.gather(x, 0), observations),
        transition_fn=prior_fn,
        observation_fn=observation_fn,
        proposal_fn=prior_proposal_fn,
        resample_fn=resample_fn,
        resample_criterion_fn=resample_criterion_fn,
        seed=seed)

    def _loop_body(step,
                   previous_step_results,
                   accumulated_traced_results,
                   num_steps_traced):
      """Take one step in dynamics and accumulate marginal likelihood."""

      step_has_observation = (
          # The second of these conditions subsumes the first, but both are
          # useful because the first can often be evaluated statically.
          ps.equal(num_transitions_per_observation, 1) |
          ps.equal(step % num_transitions_per_observation, 0))
      observation_idx = step // num_transitions_per_observation
      current_observation = tf.nest.map_structure(
          lambda x, step=step: tf.gather(x, observation_idx), observations)

      new_step_results = _filter_one_step(
          step=step,
          previous_step_results=previous_step_results,
          observation=current_observation,
          transition_fn=transition_fn,
          observation_fn=observation_fn,
          proposal_fn=proposal_fn,
          resample_criterion_fn=resample_criterion_fn,
          resample_fn=resample_fn,
          has_observation=step_has_observation,
          seed=seed)

      return _update_loop_variables(
          step=step,
          current_step_results=new_step_results,
          accumulated_traced_results=accumulated_traced_results,
          trace_fn=trace_fn,
          step_indices_to_trace=step_indices_to_trace,
          num_steps_traced=num_steps_traced)

    loop_results = tf.while_loop(
        cond=lambda step, *_: step < num_timesteps,
        body=_loop_body,
        loop_vars=_initialize_loop_variables(
            initial_step_results=initial_step_results,
            num_timesteps=num_timesteps,
            trace_fn=trace_fn,
            step_indices_to_trace=step_indices_to_trace))

    results = tf.nest.map_structure(lambda ta: ta.stack(),
                                    loop_results.accumulated_traced_results)
    if step_indices_to_trace is not None:
      # If we were passed a rank-0 (single scalar) step to trace, don't
      # return a time axis in the returned results.
      results = ps.cond(
          traced_steps_have_rank_zero,
          lambda: tf.nest.map_structure(lambda x: x[0, ...], results),
          lambda: results)

    return results


def _canonicalize_steps_to_trace(step_indices_to_trace, num_timesteps):
  """Canonicalizes `3` -> `[3]`, `[-2, -1]` -> `[N - 2, N - 1]`, etc."""
  step_indices_to_trace = tf.convert_to_tensor(
      step_indices_to_trace, dtype_hint=tf.int32)  # Warning: breaks gradients.
  traced_steps_have_rank_zero = ps.equal(
      ps.rank_from_shape(ps.shape(step_indices_to_trace)), 0)
  # Canonicalize negative step indices as positive.
  step_indices_to_trace = ps.where(step_indices_to_trace < 0,
                                   num_timesteps + step_indices_to_trace,
                                   step_indices_to_trace)
  # Canonicalize scalars as length-one vectors.
  return (ps.reshape(step_indices_to_trace, [ps.size(step_indices_to_trace)]),
          traced_steps_have_rank_zero)


def _initialize_loop_variables(initial_step_results,
                               num_timesteps,
                               trace_fn,
                               step_indices_to_trace):
  """Initialize arrays and other quantities passed through the filter loop."""

  # Create arrays to store traced values (particles, likelihoods, etc).
  num_steps_to_trace = (num_timesteps
                        if step_indices_to_trace is None
                        else ps.size0(step_indices_to_trace))
  traced_results = trace_fn(initial_step_results)
  trace_arrays = tf.nest.map_structure(
      lambda x: tf.TensorArray(dtype=x.dtype, size=num_steps_to_trace),
      traced_results)
  # If we are supposed to trace at step 0, write the traced values.
  num_steps_traced, trace_arrays = ps.cond(
      (True if step_indices_to_trace is None
       else ps.equal(step_indices_to_trace[0], 0)),
      lambda: (1,  # pylint: disable=g-long-lambda
               tf.nest.map_structure(
                   lambda ta, x: ta.write(0, x),
                   trace_arrays,
                   traced_results)),
      lambda: (0, trace_arrays))

  return ParticleFilterLoopVariables(
      step=1,
      previous_step_results=initial_step_results,
      accumulated_traced_results=trace_arrays,
      num_steps_traced=num_steps_traced)


def _update_loop_variables(step,
                           current_step_results,
                           accumulated_traced_results,
                           trace_fn,
                           step_indices_to_trace,
                           num_steps_traced):
  """Update the loop state to reflect a step of filtering."""

  # Write particles, indices, and likelihoods to their respective arrays.
  trace_this_step = True
  if step_indices_to_trace is not None:
    trace_this_step = ps.equal(
        step_indices_to_trace[ps.minimum(
            num_steps_traced,
            ps.cast(ps.size0(step_indices_to_trace) - 1, dtype=np.int32))],
        step)
  num_steps_traced, accumulated_traced_results = ps.cond(
      trace_this_step,
      lambda: (num_steps_traced + 1,  # pylint: disable=g-long-lambda
               tf.nest.map_structure(
                   lambda x, y: x.write(num_steps_traced, y),
                   accumulated_traced_results,
                   trace_fn(current_step_results))),
      lambda: (num_steps_traced, accumulated_traced_results))

  return ParticleFilterLoopVariables(
      step=step + 1,
      previous_step_results=current_step_results,
      accumulated_traced_results=accumulated_traced_results,
      num_steps_traced=num_steps_traced)


def _filter_one_step(step,
                     observation,
                     previous_step_results,
                     transition_fn,
                     observation_fn,
                     proposal_fn,
                     resample_fn,
                     resample_criterion_fn,
                     has_observation=True,
                     seed=None):
  """Advances the particle filter by a single time step."""
  with tf.name_scope('filter_one_step'):
    seed = SeedStream(seed, 'filter_one_step')
    num_particles = ps.size0(previous_step_results.log_weights)

    proposed_particles, proposal_log_weights = _propose_with_log_weights(
        step=step - 1,
        particles=previous_step_results.particles,
        transition_fn=transition_fn,
        proposal_fn=proposal_fn,
        seed=seed)
    log_weights = tf.nn.log_softmax(
        proposal_log_weights + previous_step_results.log_weights, axis=-1)

    # If this step has an observation, compute its weights and marginal
    # likelihood (and otherwise, leave weights unchanged).
    observation_log_weights = ps.cond(
        has_observation,
        lambda: ps.broadcast_to(  # pylint: disable=g-long-lambda
            _compute_observation_log_weights(
                step, proposed_particles, observation, observation_fn),
            ps.shape(log_weights)),
        lambda: tf.zeros_like(log_weights))

    unnormalized_log_weights = log_weights + observation_log_weights
    log_weights = tf.nn.log_softmax(unnormalized_log_weights, axis=0)
    # Every entry of `log_weights` differs from `unnormalized_log_weights`
    # by the same normalizing constant. We extract that constant by examining
    # an arbitrary entry.
    incremental_log_marginal_likelihood = (
        unnormalized_log_weights[0] - log_weights[0])

    # Adaptive resampling: resample particles iff the specified criterion.
    do_resample = resample_criterion_fn(unnormalized_log_weights)

    # Some batch elements may require resampling and others not, so
    # we first do the resampling for all elements, then select whether to use
    # the resampled values for each batch element according to
    # `do_resample`. If there were no batching, we might prefer to use
    # `tf.cond` to avoid the resampling computation on steps where it's not
    # needed---but we're ultimately interested in adaptive resampling
    # for statistical (not computational) purposes, so this isn't a dealbreaker.
    resampled_particles, resample_indices = weighted_resampling.resample(
        proposed_particles, log_weights, resample_fn, seed=seed)

    uniform_weights = (ps.zeros_like(log_weights) -
                       ps.log(num_particles))
    (resampled_particles,
     resample_indices,
     log_weights) = tf.nest.map_structure(
         lambda r, p: ps.where(do_resample, r, p),
         (resampled_particles, resample_indices, uniform_weights),
         (proposed_particles,
          _dummy_indices_like(resample_indices),
          log_weights))

  return ParticleFilterStepResults(
      particles=resampled_particles,
      log_weights=log_weights,
      parent_indices=resample_indices,
      incremental_log_marginal_likelihood=incremental_log_marginal_likelihood,
      accumulated_log_marginal_likelihood=(
          previous_step_results.accumulated_log_marginal_likelihood +
          incremental_log_marginal_likelihood))


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
      `concat([[num_particles, b1, ..., bN], latent_part_event_shape])`, where
      `b1, ..., bN` are optional batch dimensions.
    observation: Nested structure of `Tensor`s, each of shape
      `concat([[b1, ..., bN], observation_part_event_shape])` where
      `b1, ..., bN` are optional batch dimensions.
    observation_fn: callable, producing a distribution over `observation`s.
  Returns:
    log_weights: `Tensor` of shape `concat([num_particles, b1, ..., bN])`.
  """
  with tf.name_scope('compute_observation_log_weights'):
    observation_dist = observation_fn(step, particles)
    return observation_dist.log_prob(observation)


def reconstruct_trajectories(particles, parent_indices, name=None):
  """Reconstructs the ancestor trajectory that generated each final particle."""
  with tf.name_scope(name or 'reconstruct_trajectories'):
    # Walk backwards to compute the ancestor of each final particle at time t.
    final_indices = _dummy_indices_like(parent_indices[-1])
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
