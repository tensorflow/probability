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
import tensorflow_probability as tfp
from tensorflow_probability.python.distributions import categorical
from tensorflow_probability.python.distributions import distribution as distribution_lib
from tensorflow_probability.python.distributions import exponential
from tensorflow_probability.python.distributions import uniform
from tensorflow_probability.python.internal import distribution_util as dist_util
from tensorflow_probability.python.internal import docstring_util
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.internal import tensorshape_util

from tensorflow_probability.python.util import SeedStream

__all__ = [
    'ess_below_threshold',
    'infer_trajectories',
    'particle_filter',
    'reconstruct_trajectories',
    'resample_independent',
    'resample_stratified',
    'resample_systematic',
    'resample_deterministic_minimum_error'
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


# TODO(davmre): Replace this hack with a more efficient TF builtin.
def _batch_gather(params, indices, axis=0):
  """Gathers a batch of indices from `params` along the given axis.

  Args:
    params: `Tensor` of shape `[d[0], d[1], ..., d[N - 1]]`.
    indices: int `Tensor` of shape broadcastable to that of `params`.
    axis: int `Tensor` dimension of `params` (and of the broadcast indices) to
      gather over.
  Returns:
    result: `Tensor` of the same type and shape as `params`.
  """
  params_rank = ps.rank_from_shape(ps.shape(params))
  indices_rank = ps.rank_from_shape(ps.shape(indices))
  params_with_axis_on_right = dist_util.move_dimension(
      params, source_idx=axis, dest_idx=-1)
  indices_with_axis_on_right = ps.broadcast_to(
      dist_util.move_dimension(indices,
                               source_idx=axis - (params_rank - indices_rank),
                               dest_idx=-1),
      ps.shape(params_with_axis_on_right))

  result = tf.gather(params_with_axis_on_right,
                     indices_with_axis_on_right,
                     axis=params_rank - 1,
                     batch_dims=params_rank - 1)
  return dist_util.move_dimension(result, source_idx=-1, dest_idx=axis)


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


def _gather_history(structure, step, num_steps):
  """Gather up to `num_steps` of history from a nested structure."""
  initial_step = ps.maximum(0, step - num_steps)
  return tf.nest.map_structure(
      lambda x: tf.gather(x, ps.range(initial_step, step)),
      structure)


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
     'state_history',  # Set to `tf.zeros([0])` if not tracked.
     'num_steps_traced'
    ])


particle_filter_arg_str = """\
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
  num_steps_state_history_to_pass: scalar Python `int` number of steps to
    include in the optional `state_history` argument to `transition_fn`,
    `observation_fn`, and `proposal_fn`. If `None`, this argument
    will not be passed.
    Default value: `None`.
  num_steps_observation_history_to_pass: scalar Python `int` number of steps
    to include in the optional `observation_history` argument to
    `transition_fn`, `observation_fn`, and `proposal_fn`. If `None`, this
    argument will not be passed.
    Default value: `None`."""

non_markovian_specification_str = """\
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
refers to the observation at the previous timestep, and so on."""


@docstring_util.expand_docstring(
    particle_filter_arg_str=particle_filter_arg_str)
def infer_trajectories(observations,
                       initial_state_prior,
                       transition_fn,
                       observation_fn,
                       num_particles,
                       initial_state_proposal=None,
                       proposal_fn=None,
                       resample_criterion_fn=ess_below_threshold,
                       rejuvenation_kernel_fn=None,
                       num_transitions_per_observation=1,
                       num_steps_state_history_to_pass=None,
                       num_steps_observation_history_to_pass=None,
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

  ${non_markovian_specification_str}

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
         resample_criterion_fn=resample_criterion_fn,
         rejuvenation_kernel_fn=rejuvenation_kernel_fn,
         num_transitions_per_observation=num_transitions_per_observation,
         num_steps_state_history_to_pass=num_steps_state_history_to_pass,
         num_steps_observation_history_to_pass=(
             num_steps_observation_history_to_pass),
         trace_fn=_default_trace_fn,
         seed=seed,
         name=name)
    weighted_trajectories = reconstruct_trajectories(particles, parent_indices)

    # Resample all steps of the trajectories using the final weights.
    resample_indices = categorical.Categorical(
        dist_util.move_dimension(
            log_weights[-1, ...],
            source_idx=0,
            dest_idx=-1)).sample(num_particles, seed=seed)
    trajectories = tf.nest.map_structure(
        lambda x: _batch_gather(x, resample_indices, axis=1),
        weighted_trajectories)

    return trajectories, incremental_log_marginal_likelihoods


@docstring_util.expand_docstring(
    particle_filter_arg_str=particle_filter_arg_str,
    non_markovian_specification_str=non_markovian_specification_str)
def particle_filter(observations,
                    initial_state_prior,
                    transition_fn,
                    observation_fn,
                    num_particles,
                    initial_state_proposal=None,
                    proposal_fn=None,
                    resample_criterion_fn=ess_below_threshold,
                    rejuvenation_kernel_fn=None,  # TODO(davmre): not yet supported. pylint: disable=unused-argument
                    num_transitions_per_observation=1,
                    num_steps_state_history_to_pass=None,
                    num_steps_observation_history_to_pass=None,
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

  ${non_markovian_specification_str}
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
        resample_criterion_fn=resample_criterion_fn,
        seed=seed)

    def _loop_body(step,
                   previous_step_results,
                   accumulated_traced_results,
                   state_history,
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

      history_to_pass_into_fns = {}
      if num_steps_observation_history_to_pass:
        history_to_pass_into_fns['observation_history'] = _gather_history(
            observations,
            observation_idx,
            num_steps_observation_history_to_pass)
      if num_steps_state_history_to_pass:
        history_to_pass_into_fns['state_history'] = state_history

      new_step_results = _filter_one_step(
          step=step,
          previous_step_results=previous_step_results,
          observation=current_observation,
          transition_fn=functools.partial(
              transition_fn, **history_to_pass_into_fns),
          observation_fn=functools.partial(
              observation_fn, **history_to_pass_into_fns),
          proposal_fn=(
              None if proposal_fn is None else
              functools.partial(proposal_fn, **history_to_pass_into_fns)),
          resample_criterion_fn=resample_criterion_fn,
          has_observation=step_has_observation,
          seed=seed)

      return _update_loop_variables(
          step=step,
          current_step_results=new_step_results,
          accumulated_traced_results=accumulated_traced_results,
          state_history=state_history,
          trace_fn=trace_fn,
          step_indices_to_trace=step_indices_to_trace,
          num_steps_traced=num_steps_traced)

    loop_results = tf.while_loop(
        cond=lambda step, *_: step < num_timesteps,
        body=_loop_body,
        loop_vars=_initialize_loop_variables(
            initial_step_results=initial_step_results,
            num_timesteps=num_timesteps,
            num_steps_state_history_to_pass=num_steps_state_history_to_pass,
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
      step_indices_to_trace, dtype_hint=tf.int32)
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
                               num_steps_state_history_to_pass,
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
            ps.concat([[num_steps_state_history_to_pass],
                       ps.shape(x)], axis=0)),
        initial_step_results.particles)

  return ParticleFilterLoopVariables(
      step=1,
      previous_step_results=initial_step_results,
      accumulated_traced_results=trace_arrays,
      state_history=state_history,
      num_steps_traced=num_steps_traced)


def _update_loop_variables(step,
                           current_step_results,
                           accumulated_traced_results,
                           state_history,
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

  history_is_empty = (tf.is_tensor(state_history) and
                      state_history.shape[0] == 0)
  if not history_is_empty:
    # Permute the particles from previous steps to match the current resampled
    # indices, so that the state history reflects coherent trajectories.
    resampled_state_history = tf.nest.map_structure(
        lambda x: _batch_gather(x[1:],  # pylint: disable=g-long-lambda
                                current_step_results.parent_indices,
                                axis=1),
        state_history)

    # Update the history by concat'ing the carried-forward elements with the
    # most recent state.
    state_history = tf.nest.map_structure(
        lambda h, s: tf.concat([h, s[tf.newaxis, ...]], axis=0),
        resampled_state_history,
        current_step_results.particles)

  return ParticleFilterLoopVariables(
      step=step + 1,
      previous_step_results=current_step_results,
      accumulated_traced_results=accumulated_traced_results,
      state_history=state_history,
      num_steps_traced=num_steps_traced)


def _filter_one_step(step,
                     observation,
                     previous_step_results,
                     transition_fn,
                     observation_fn,
                     proposal_fn,
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
    resampled_particles, resample_indices = _resample(
        proposed_particles, log_weights, resample_independent, seed=seed)

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


def _resample(particles, log_weights, resample_fn, seed=None):
  """Resamples the current particles according to provided weights.

  Args:
    particles: Nested structure of `Tensor`s each of shape
      `[num_particles, b1, ..., bN, ...]`, where
      `b1, ..., bN` are optional batch dimensions.
    log_weights: float `Tensor` of shape `[num_particles, b1, ..., bN]`, where
      `b1, ..., bN` are optional batch dimensions.
    resample_fn: choose the function used for resampling.
      Use 'resample_systematic' for systematic resampling.
      Use 'resample_deterministic_minimum_error' for minimum error resampling.
      Use 'resample_independent' for independent resampling.
    seed: Python `int` random seed.

  Returns:
    resampled_particles: Nested structure of `Tensor`s, matching `particles`.
    resample_indices: int `Tensor` of shape `[num_particles, b1, ..., bN]`.
  """
  with tf.name_scope('resample'):
    weights_shape = ps.shape(log_weights)
    num_particles = weights_shape[0]
    log_probs = tf.math.log_softmax(log_weights, axis=0)
    resampled_indices = resample_fn(log_probs, num_particles, (), seed=seed)
    resampled_particles = tf.nest.map_structure(
        lambda x: _batch_gather(x, resampled_indices, axis=0),
        particles)
  return resampled_particles, resampled_indices


def reconstruct_trajectories(particles, parent_indices, name=None):
  """Reconstructs the ancestor trajectory that generated each final particle."""
  with tf.name_scope(name or 'reconstruct_trajectories'):
    # Walk backwards to compute the ancestor of each final particle at time t.
    final_indices = _dummy_indices_like(parent_indices[-1])
    ancestor_indices = tf.scan(
        fn=lambda ancestor, parent: _batch_gather(parent, ancestor, axis=0),
        elems=parent_indices[1:],
        initializer=final_indices,
        reverse=True)
    ancestor_indices = tf.concat([ancestor_indices, [final_indices]], axis=0)

  return tf.nest.map_structure(
      lambda part: _batch_gather(part, ancestor_indices, axis=1), particles)


# TODO(b/153689734): rewrite so as not to use `move_dimension`.
def _resample_using_log_points(log_probs, sample_shape, log_points, name=None):
  """Resample from `log_probs` using supplied points in interval `[0, 1]`."""

  # We divide up the unit interval [0, 1] according to the provided
  # probability distributions using `cumulative_logsumexp`.
  # At the end of each division we place a 'marker'.
  # We use points on the unit interval supplied by caller.
  # We sort the combination of points and markers. The number
  # of points between the markers defining a division gives the number
  # of samples we require in that division.
  # For example, suppose `probs` is `[0.2, 0.3, 0.5]`.
  # We divide up `[0, 1]` using 3 markers:
  #
  #     |     |          |
  # 0.  0.2   0.5        1.0  <- markers
  #
  # Suppose we are given four points: [0.1, 0.25, 0.9, 0.75]
  # After sorting the combination we get:
  #
  # 0.1  0.25     0.75 0.9    <- points
  #  *  | *   |    *    *|
  # 0.   0.2 0.5         1.0  <- markers
  #
  # We have one sample in the first category, one in the second and
  # two in the last.
  #
  # All of these computations are carried out in batched form.

  with tf.name_scope(name or 'resample_using_log_points') as name:
    points_shape = ps.shape(log_points)
    batch_shape, [num_markers] = ps.split(ps.shape(log_probs),
                                          num_or_size_splits=[-1, 1])

    # `working_shape` specifies the total number of events
    # we will be generating.
    working_shape = ps.concat([sample_shape, batch_shape], axis=0)
    # `markers_shape` is the shape of the markers we temporarily insert.
    markers_shape = ps.concat([working_shape, [num_markers]], axis=0)

    markers = ps.concat(
        [tf.ones(markers_shape, dtype=tf.int32),
         tf.zeros(points_shape, dtype=tf.int32)],
        axis=-1)
    log_marker_positions = tf.broadcast_to(
        tfp.math.log_cumsum_exp(log_probs, axis=-1),
        markers_shape)
    log_markers_and_points = ps.concat(
        [log_marker_positions, log_points], axis=-1)
    # Stable sort is used to ensure that no points get sorted between
    # markers that have zero distance between them. This ensures that
    # there will never be a sample drawn whose probability is intended
    # to be zero even when a point falls on the edge of the
    # corresponding zero-width bucket.
    indices = tf.argsort(log_markers_and_points, axis=-1, stable=True)
    sorted_markers = tf.gather_nd(
        markers,
        indices[..., tf.newaxis],
        batch_dims=(
            ps.rank_from_shape(sample_shape) +
            ps.rank_from_shape(batch_shape)))
    markers_and_samples = ps.cast(
        tf.cumsum(sorted_markers, axis=-1), dtype=tf.int32)
    markers_and_samples = tf.minimum(markers_and_samples, num_markers - 1)

    # Collect up samples, omitting markers.
    samples_mask = tf.equal(sorted_markers, 0)

    # The following block of code is equivalent to
    # `samples = markers_and_samples[samples_mask]` however boolean mask
    # indices are not supported by XLA.
    # Instead we use `argsort` to pick out the top `num_samples`
    # elements of `markers_and_samples` when sorted using `samples_mask`
    # as key.
    num_samples = points_shape[-1]
    sample_locations = tf.argsort(
        ps.cast(samples_mask, dtype=tf.int32),
        direction='DESCENDING',
        stable=True)
    samples = tf.gather_nd(
        markers_and_samples,
        sample_locations[..., :num_samples, tf.newaxis],
        batch_dims=(
            ps.rank_from_shape(sample_shape) +
            ps.rank_from_shape(batch_shape)))

    return tf.reshape(samples, points_shape)


# TODO(b/153689734): rewrite so as not to use `move_dimension`.
def resample_independent(log_probs, event_size, sample_shape,
                         seed=None, name=None):
  """Categorical resampler for sequential Monte Carlo.

  The return value from this function is similar to sampling with
  ```python
  expanded_sample_shape = tf.concat([[event_size], sample_shape]), axis=-1)
  tfd.Categorical(logits=log_probs).sample(expanded_sample_shape)`
  ```
  but with values sorted along the first axis. It can be considered to be
  sampling events made up of a length-`event_size` vector of draws from
  the `Categorical` distribution. For large input values this function should
  give better performance than using `Categorical`.
  The sortedness is an unintended side effect of the algorithm that is
  harmless in the context of simple SMC algorithms.

  This implementation is based on the algorithms in [Maskell et al. (2006)][1].
  It is also known as multinomial resampling as described in
  [Doucet et al. (2011)][2].

  Args:
    log_probs: A tensor-valued batch of discrete log probability distributions.
    event_size: the dimension of the vector considered a single draw.
    sample_shape: the `sample_shape` determining the number of draws.
    seed: Python '`int` used to seed calls to `tf.random.*`.
      Default value: None (i.e. no seed).
    name: Python `str` name for ops created by this method.
      Default value: `None` (i.e., `'resample_independent'`).

  Returns:
    resampled_indices: a tensor of samples.

  #### References

  [1]: S. Maskell, B. Alun-Jones and M. Macleod. A Single Instruction Multiple
       Data Particle Filter.
       In 2006 IEEE Nonlinear Statistical Signal Processing Workshop.
       http://people.ds.cam.ac.uk/fanf2/hermes/doc/antiforgery/stats.pdf
  [2]: A. Doucet & A. M. Johansen. Tutorial on Particle Filtering and
       Smoothing: Fifteen Years Later
       In 2011 The Oxford Handbook of Nonlinear Filtering
       https://www.stats.ox.ac.uk/~doucet/doucet_johansen_tutorialPF2011.pdf

  """
  with tf.name_scope(name or 'resample_independent') as name:
    log_probs = tf.convert_to_tensor(log_probs, dtype_hint=tf.float32)
    log_probs = dist_util.move_dimension(log_probs, source_idx=0, dest_idx=-1)
    points_shape = ps.concat([sample_shape,
                              ps.shape(log_probs)[:-1],
                              [event_size]], axis=0)
    log_points = -exponential.Exponential(
        rate=tf.constant(1.0, dtype=log_probs.dtype)).sample(
            points_shape, seed=seed)

    resampled = _resample_using_log_points(log_probs, sample_shape, log_points)
    return dist_util.move_dimension(resampled, source_idx=-1, dest_idx=0)


# TODO(b/153689734): rewrite so as not to use `move_dimension`.
def resample_systematic(log_probs, event_size, sample_shape,
                        seed=None, name=None):
  """A systematic resampler for sequential Monte Carlo.

  The value returned from this function is similar to sampling with
  ```python
  expanded_sample_shape = tf.concat([[event_size], sample_shape]), axis=-1)
  tfd.Categorical(logits=log_probs).sample(expanded_sample_shape)`
  ```
  but with values sorted along the first axis. It can be considered to be
  sampling events made up of a length-`event_size` vector of draws from
  the `Categorical` distribution. However, although the elements of
  this event have the appropriate marginal distribution, they are not
  independent of each other. Instead they are drawn using a stratified
  sampling method that in some sense reduces variance and is suitable for
  use with Sequential Monte Carlo algorithms as described in
  [Doucet et al. (2011)][2].
  The sortedness is an unintended side effect of the algorithm that is
  harmless in the context of simple SMC algorithms.

  This implementation is based on the algorithms in [Maskell et al. (2006)][1]
  where it is called minimum variance resampling.

  Args:
    log_probs: A tensor-valued batch of discrete log probability distributions.
    event_size: the dimension of the vector considered a single draw.
    sample_shape: the `sample_shape` determining the number of draws.
    seed: Python '`int` used to seed calls to `tf.random.*`.
      Default value: None (i.e. no seed).
    name: Python `str` name for ops created by this method.
      Default value: `None` (i.e., `'resample_systematic'`).

  Returns:
    resampled_indices: a tensor of samples.

  #### References
  [1]: S. Maskell, B. Alun-Jones and M. Macleod. A Single Instruction Multiple
       Data Particle Filter.
       In 2006 IEEE Nonlinear Statistical Signal Processing Workshop.
       http://people.ds.cam.ac.uk/fanf2/hermes/doc/antiforgery/stats.pdf
  [2]: A. Doucet & A. M. Johansen. Tutorial on Particle Filtering and
       Smoothing: Fifteen Years Later
       In 2011 The Oxford Handbook of Nonlinear Filtering
       https://www.stats.ox.ac.uk/~doucet/doucet_johansen_tutorialPF2011.pdf

  """
  with tf.name_scope(name or 'resample_systematic') as name:
    log_probs = tf.convert_to_tensor(log_probs, dtype_hint=tf.float32)
    log_probs = dist_util.move_dimension(log_probs, source_idx=0, dest_idx=-1)
    working_shape = ps.concat([sample_shape,
                               ps.shape(log_probs)[:-1]], axis=0)
    points_shape = ps.concat([working_shape, [event_size]], axis=0)
    # Draw a single offset for each event.
    interval_width = ps.cast(1. / event_size, dtype=log_probs.dtype)
    offsets = uniform.Uniform(
        low=ps.cast(0., dtype=log_probs.dtype),
        high=interval_width).sample(
            working_shape, seed=seed)[..., tf.newaxis]
    even_spacing = tf.linspace(
        start=ps.cast(0., dtype=log_probs.dtype),
        stop=1 - interval_width,
        num=event_size) + offsets
    log_points = tf.broadcast_to(tf.math.log(even_spacing), points_shape)

    resampled = _resample_using_log_points(log_probs, sample_shape, log_points)
    return dist_util.move_dimension(resampled, source_idx=-1, dest_idx=0)


# TODO(b/153689734): rewrite so as not to use `move_dimension`.
def resample_stratified(log_probs, event_size, sample_shape,
                        seed=None, name=None):
  """Stratified resampler for sequential Monte Carlo.

  The value returned from this algorithm is similar to sampling with
  ```python
  expanded_sample_shape = tf.concat([[event_size], sample_shape]), axis=-1)
  tfd.Categorical(logits=log_probs).sample(expanded_sample_shape)`
  ```
  but with values sorted along the first axis. It can be considered to be
  sampling events made up of a length-`event_size` vector of draws from
  the `Categorical` distribution. However, although the elements of
  this event have the appropriate marginal distribution, they are not
  independent of each other. Instead they are drawn using a low variance
  stratified sampling method suitable for use with Sequential Monte
  Carlo algorithms.
  The sortedness is an unintended side effect of the algorithm that is
  harmless in the context of simple SMC algorithms.

  This function is based on Algorithm #1 in the paper
  [Maskell et al. (2006)][1].

  Args:
    log_probs: A tensor-valued batch of discrete log probability distributions.
    event_size: the dimension of the vector considered a single draw.
    sample_shape: the `sample_shape` determining the number of draws.
    seed: Python '`int` used to seed calls to `tf.random.*`.
      Default value: None (i.e. no seed).
    name: Python `str` name for ops created by this method.
      Default value: `None` (i.e., `'resample_independent'`).

  Returns:
    resampled_indices: a tensor of samples.

  #### References

  [1]: S. Maskell, B. Alun-Jones and M. Macleod. A Single Instruction Multiple
       Data Particle Filter.
       In 2006 IEEE Nonlinear Statistical Signal Processing Workshop.
       http://people.ds.cam.ac.uk/fanf2/hermes/doc/antiforgery/stats.pdf

  """
  with tf.name_scope(name or 'resample_stratified') as name:
    log_probs = tf.convert_to_tensor(log_probs, dtype_hint=tf.float32)
    log_probs = dist_util.move_dimension(log_probs, source_idx=0, dest_idx=-1)
    points_shape = ps.concat([sample_shape,
                              ps.shape(log_probs)[:-1],
                              [event_size]], axis=0)
    # Draw an offset for every element of an event.
    interval_width = ps.cast(1. / event_size, dtype=log_probs.dtype)
    offsets = uniform.Uniform(low=ps.cast(0., dtype=log_probs.dtype),
                              high=interval_width).sample(
                                  points_shape, seed=seed)
    # The unit interval is divided into equal partitions and each point
    # is a random offset into a partition.
    even_spacing = tf.linspace(
        start=ps.cast(0., dtype=log_probs.dtype),
        stop=1 - interval_width,
        num=event_size) + offsets
    log_points = tf.math.log(even_spacing)

    resampled = _resample_using_log_points(log_probs, sample_shape, log_points)
    return dist_util.move_dimension(resampled, source_idx=-1, dest_idx=0)


# TODO(b/153199903): replace this function with `tf.scatter_nd` when
# it supports `batch_dims`.
def _scatter_nd_batch(indices, updates, shape, batch_dims=0):
  """A partial implementation of `scatter_nd` supporting `batch_dims`."""

  # `tf.scatter_nd` does not support a `batch_dims` argument.
  # Instead we use the gradient of `tf.gather_nd`.
  # From a purely mathematical perspective this works because
  # (if `tf.scatter_nd` supported `batch_dims`)
  # `gather_nd` and `scatter_nd` (with matching `indices`) are
  # adjoint linear operators and
  # the gradient w.r.t `x` of `dot(y, A(x))` is `adjoint(A)(y)`.
  #
  # Another perspective: back propagating through a "neural" network
  # containing a gather operation carries derivatives backwards through the
  # network, accumulating the derivatives in the locations that
  # were gathered from, ie. they are scattered.
  # If the network multiplies each gathered element by
  # some quantity, then the backwardly propagating derivatives are scaled
  # by this quantity before being scattered.
  # Combining this with the fact that`GradientTape.gradient`
  # starts back-propagation with derivatives equal to `1`, this allows us
  # to use the multipliers to determine the quantities scattered.
  #
  # However, derivatives are only supported for floating point types
  # so we 'tunnel' our types through the `float64` type.
  # So the implmentation is "partial" in the sense that it supports
  # data that can be losslessly converted to `tf.float64` and back.
  dtype = updates.dtype
  internal_dtype = tf.float64
  multipliers = ps.cast(updates, internal_dtype)
  with tf.GradientTape() as tape:
    zeros = tf.zeros(shape, dtype=internal_dtype)
    tape.watch(zeros)
    weighted_gathered = multipliers * tf.gather_nd(
        zeros,
        indices,
        batch_dims=batch_dims)
  grad = tape.gradient(weighted_gathered, zeros)
  return ps.cast(grad, dtype=dtype)


def _finite_differences(sums):
  """The inverse of `tf.cumsum` with `axis=-1`."""
  return ps.concat(
      [sums[..., :1], sums[..., 1:] - sums[..., :-1]], axis=-1)


def _samples_from_counts(values, counts, total_number):
  """Construct sequences of values from tabulated numbers of counts."""

  extended_result_shape = ps.concat(
      [ps.shape(counts)[:-1],
       [total_number + 1]], axis=0)
  padded_counts = ps.concat(
      [ps.zeros_like(counts[..., :1]),
       counts[..., :-1]], axis=-1)
  edge_positions = ps.cumsum(padded_counts, axis=-1)

  # We need to scatter `values` into an array according to
  # the given `counts`.
  # Because the final result typically consists of sequences of samples
  # that are constant in blocks, we can scatter just the finite
  # differences of the values (which become the 'edges' of the blocks)
  # and then cumulatively sum them back up
  # at the end. (Reminiscent of run length encoding.)
  # Eg. suppose we have values = `[0, 2, 1]`
  # and counts = `[2, 3, 4]`
  # Then the output we require is `[0, 0, 2, 2, 2, 1, 1, 1, 1]`.
  # The finite differences of the input are:
  #   `[0, 2, -1]`.
  # The finite differences of the output are:
  #   `[0, 0, 2, 0, 0, -1, 0, 0, 0]`.
  # The latter is the former scattered into a larger array.
  #
  # So the algorithm is essentially
  # compute finite differences -> scatter -> undo finite differences
  edge_heights = _finite_differences(values)
  edges = _scatter_nd_batch(
      edge_positions[..., tf.newaxis],
      edge_heights,
      extended_result_shape,
      batch_dims=ps.rank_from_shape(ps.shape(counts)) - 1)

  result = tf.cumsum(edges, axis=-1)[..., :-1]
  return result


# TODO(b/153689734): rewrite so as not to use `move_dimension`.
def resample_deterministic_minimum_error(
    log_probs, event_size, sample_shape,
    seed=None, name='resample_deterministic_minimum_error'):
  """Deterministic minimum error resampler for sequential Monte Carlo.

    The return value of this function is similar to sampling with
    ```python
    expanded_sample_shape = tf.concat([sample_shape, [event_size]]), axis=-1)
    tfd.Categorical(logits=log_probs).sample(expanded_sample_shape)`
    ```
    but with values chosen deterministically so that the empirical distribution
    is as close as possible to the specified distribution.
    (Note that the empirical distribution can only exactly equal the requested
    distribution if multiplying every probability by `event_size` gives
    an integer. So in general this is a biased "sampler".)
    It is intended to provide a good representative sample, suitable for use
    with some Sequential Monte Carlo algorithms.

  This function is based on Algorithm #3 in [Maskell et al. (2006)][1].

  Args:
    log_probs: a tensor-valued batch of discrete log probability distributions.
    event_size: the dimension of the vector considered a single draw.
    sample_shape: the `sample_shape` determining the number of draws. Because
      this resampler is deterministic it simply replicates the draw you
      would get for `sample_shape=[1]`.
    seed: This argument is unused but is present so that this function shares
      its interface with the other resampling functions.
      Default value: None
    name: Python `str` name for ops created by this method.
      Default value: `None` (i.e., `'resample_deterministic_minimum_error'`).

  Returns:
    resampled_indices: a tensor of samples.

  #### References
  [1]: S. Maskell, B. Alun-Jones and M. Macleod. A Single Instruction Multiple
       Data Particle Filter.
       In 2006 IEEE Nonlinear Statistical Signal Processing Workshop.
       http://people.ds.cam.ac.uk/fanf2/hermes/doc/antiforgery/stats.pdf

  """
  del seed

  with tf.name_scope(name or 'resample_deterministic_minimum_error'):
    sample_shape = tf.convert_to_tensor(sample_shape, dtype_hint=tf.int32)
    log_probs = dist_util.move_dimension(
        log_probs, source_idx=0, dest_idx=-1)
    probs = tf.math.exp(log_probs)
    prob_shape = ps.shape(probs)
    pdf_size = prob_shape[-1]
    # If we could draw fractional numbers of samples we would
    # choose `ideal_numbers` for the number of each element.
    ideal_numbers = event_size * probs
    # We approximate the ideal numbers by truncating to integers
    # and then repair the counts starting with the one with the
    # largest fractional error and working our way down.
    first_approximation = tf.floor(ideal_numbers)
    missing_fractions = ideal_numbers - first_approximation
    first_approximation = ps.cast(
        first_approximation, dtype=tf.int32)
    fraction_order = tf.argsort(missing_fractions, axis=-1)
    # We sort the integer parts and fractional parts together.
    batch_dims = ps.rank_from_shape(prob_shape) - 1
    first_approximation = tf.gather_nd(
        first_approximation,
        fraction_order[..., tf.newaxis],
        batch_dims=batch_dims)
    missing_fractions = tf.gather_nd(
        missing_fractions,
        fraction_order[..., tf.newaxis],
        batch_dims=batch_dims)
    sample_defect = event_size - tf.reduce_sum(
        first_approximation, axis=-1, keepdims=True)
    unpermuted = tf.broadcast_to(
        tf.range(pdf_size),
        prob_shape)
    increments = ps.cast(
        unpermuted >= pdf_size - sample_defect,
        dtype=first_approximation.dtype)
    counts = first_approximation + increments
    samples = _samples_from_counts(fraction_order, counts, event_size)
    result_shape = tf.concat([sample_shape,
                              prob_shape[:-1],
                              [event_size]], axis=0)
    # Replicate sample up to batch size.
    # TODO(dpiponi): rather than replicating, spread the "error" over
    # multiple samples with a minimum-discrepancy sequence.
    resampled = tf.broadcast_to(samples, result_shape)
    return dist_util.move_dimension(resampled, source_idx=-1, dest_idx=0)
