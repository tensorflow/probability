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
"""Experimental MCMC driver, `sample_sequential_monte_carlo`."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.math.generic import reduce_logmeanexp
from tensorflow_probability.python.mcmc import hmc
from tensorflow_probability.python.mcmc import random_walk_metropolis
from tensorflow_probability.python.mcmc import transformed_kernel
from tensorflow_probability.python.mcmc.internal import util as mcmc_util
from tensorflow_probability.python.mcmc.sample_annealed_importance import _find_inner_mh_results
from tensorflow_probability.python.util.seed_stream import SeedStream


__all__ = [
    'sample_sequential_monte_carlo',
]


PRINT_DEBUG = False
TUNE_STEPS = True

ParticleInfo = collections.namedtuple(
    'ParticleInfo',
    [
        'accept_prob',  # acceptance probability per particle
        'scalings',
        'tempered_log_prob',
        'likelihood_log_prob',
    ])

SMCResults = collections.namedtuple(
    'SMCResults',
    [
        'num_steps',
        'inverse_temperature',
        'log_marginal_likelihood',
        'particle_info',  # A namedtuple of ParticleInfo
    ])


def _make_tempered_target_log_prob_fn(
    prior_log_prob_fn, likelihood_log_prob_fn, temperatures):
  """Helper which creates inner kernel target_log_prob_fn."""
  def _tempered_target_log_prob(*args):
    priorlogprob = tf.identity(prior_log_prob_fn(*args),
                               name='prior_log_prob')
    loglike = tf.identity(likelihood_log_prob_fn(*args),
                          name='likelihood_log_prob')
    return tf.identity(priorlogprob + loglike * temperatures,
                       name='tempered_logp')
  return _tempered_target_log_prob


def make_rwmh_kernel_fn(target_log_prob_fn, init_state, scalings, seed=None):
  """Generate a Random Walk MH kernel."""
  with tf.name_scope('make_rwmh_kernel_fn'):
    seed = SeedStream(seed, salt='make_rwmh_kernel_fn')
    state_std = [
        tf.math.reduce_std(x, axis=0, keepdims=True) for x in init_state
    ]
    step_size = [
        s * ps.cast(  # pylint: disable=g-complex-comprehension
            mcmc_util.left_justified_expand_dims_like(scalings, s),
            s.dtype) for s in state_std
    ]
    return random_walk_metropolis.RandomWalkMetropolis(
        target_log_prob_fn,
        new_state_fn=random_walk_metropolis.random_walk_normal_fn(
            scale=step_size),
        seed=seed)


def compute_hmc_step_size(scalings, state_std, num_leapfrog_steps):
  return [
      s / ps.cast(num_leapfrog_steps, s.dtype) * ps.cast(  # pylint: disable=g-complex-comprehension
          mcmc_util.left_justified_expand_dims_like(scalings, s),
          s.dtype) for s in state_std
  ]


def gen_make_transform_hmc_kernel_fn(unconstraining_bijectors,
                                     num_leapfrog_steps=10):
  """Generate a transformed hmc kernel."""

  def make_transform_hmc_kernel_fn(
      target_log_prob_fn,
      init_state,
      scalings,
      seed=None):
    """Generate a transform hmc kernel."""

    with tf.name_scope('make_transformed_hmc_kernel_fn'):
      seed = SeedStream(seed, salt='make_transformed_hmc_kernel_fn')
      state_std = [
          bij.inverse(
              tf.math.reduce_std(bij.forward(x), axis=0, keepdims=True))
          for x, bij in zip(init_state, unconstraining_bijectors)
      ]
      step_size = compute_hmc_step_size(scalings, state_std, num_leapfrog_steps)
      return transformed_kernel.TransformedTransitionKernel(
          hmc.HamiltonianMonteCarlo(
              target_log_prob_fn=target_log_prob_fn,
              num_leapfrog_steps=num_leapfrog_steps,
              step_size=step_size,
              seed=seed),
          unconstraining_bijectors)

  return make_transform_hmc_kernel_fn


def gen_make_hmc_kernel_fn(num_leapfrog_steps=10):
  """Generate a transformed hmc kernel."""
  def make_hmc_kernel_fn(
      target_log_prob_fn,
      init_state,
      scalings,
      seed=None):
    """Generate a hmc without transformation kernel."""

    with tf.name_scope('make_hmc_kernel_fn'):
      seed = SeedStream(seed, salt='make_hmc_kernel_fn')
      state_std = [
          tf.math.reduce_std(x, axis=0, keepdims=True)
          for x in init_state
      ]
      step_size = compute_hmc_step_size(scalings, state_std, num_leapfrog_steps)
      return hmc.HamiltonianMonteCarlo(
          target_log_prob_fn=target_log_prob_fn,
          num_leapfrog_steps=num_leapfrog_steps,
          step_size=step_size,
          seed=seed)

  return make_hmc_kernel_fn

# Generate a default `make_hmc_kernel_fn`
default_make_hmc_kernel_fn = gen_make_hmc_kernel_fn()


# TODO(b/152412213) Experitment to improve recommendation on static parmaeters
def sample_sequential_monte_carlo(
    prior_log_prob_fn,
    likelihood_log_prob_fn,
    current_state,
    max_num_steps=25,
    max_stage=100,
    make_kernel_fn=make_rwmh_kernel_fn,
    optimal_accept=0.234,
    target_accept_prob=0.99,
    ess_threshold_ratio=0.5,
    parallel_iterations=10,
    seed=None,
    name=None):
  """Runs Sequential Monte Carlo to sample from the posterior distribution.

  This function uses an MCMC transition operator (e.g., Hamiltonian Monte Carlo)
  to sample from a series of distributions that slowly interpolates between
  an initial 'prior' distribution:

  `exp(prior_log_prob_fn(x))`

  and the target 'posterior' distribution:

  `exp(prior_log_prob_fn(x) + target_log_prob_fn(x))`,

  by mutating a collection of MC samples (i.e., particles). The approach is also
  known as Particle Filter in some literature.

  Args:
    prior_log_prob_fn: Python callable that returns the log density of the
      prior distribution.
    likelihood_log_prob_fn: Python callable which takes an argument like
      `current_state` (or `*current_state` if it's a list) and returns its
      (possibly unnormalized) log-density under the likelihood distribution.
    current_state: `Tensor` or Python `list` of `Tensor`s representing the
      current state(s) of the Markov chain(s). The first `r` dimensions index
      independent chains, `r = tf.rank(target_log_prob_fn(*current_state))`.
    max_num_steps: The maximum number of kernel transition steps in one mutation
      of the MC samples. Note that the actual number of steps in one mutation is
      tuned during sampling and likely lower than the max_num_steps.
    max_stage: Integer number of the stage for increasing the temperature
      from 0 to 1.
    make_kernel_fn: Python `callable` which returns a `TransitionKernel`-like
      object. Must take one argument representing the `TransitionKernel`'s
      `target_log_prob_fn`. The `target_log_prob_fn` argument represents the
      `TransitionKernel`'s target log distribution.  Note:
      `sample_annealed_importance_chain` creates a new `target_log_prob_fn`
      which is an interpolation between the supplied `target_log_prob_fn` and
      `proposal_log_prob_fn`; it is this interpolated function which is used as
      an argument to `make_kernel_fn`.
    optimal_accept: Optimal acceptance ratio for a Transitional Kernel. Default
      to 0.234 for Random Walk Metropolis kernel.
    target_accept_prob: Target acceptance probability at the end of one mutation
      step.
    ess_threshold_ratio: Target ratio for effective sample size.
    parallel_iterations: The number of iterations allowed to run in parallel.
        It must be a positive integer. See `tf.while_loop` for more details.
    seed: Python integer or TFP seedstream to seed the random number generator.
    name: Python `str` name prefixed to Ops created by this function.
      Default value: `None` (i.e., 'sample_annealed_importance_chain').

  Returns:
    n_stage: Number of the mutation stage SMC ran.
    final_state: `Tensor` or Python `list` of `Tensor`s representing the
      final state(s) of the Markov chain(s). The output are the posterior
      samples.
    final_kernel_results: `collections.namedtuple` of internal calculations used
      to advance the chain.

  """

  with tf.name_scope(name or 'sample_sequential_monte_carlo'):
    seed_stream = SeedStream(seed, salt='smc_seed')

    unwrap_state_list = not tf.nest.is_nested(current_state)
    if unwrap_state_list:
      current_state = [current_state]
    current_state = [
        tf.convert_to_tensor(s, dtype_hint=tf.float32) for s in current_state
    ]

    num_replica = ps.size0(current_state[0])
    effective_sample_size_threshold = tf.cast(
        num_replica * ess_threshold_ratio, tf.int32)

    def preprocess_state(init_state):
      """Initial preprocessing at Stage 0."""
      dimension = ps.reduce_sum([
          ps.reduce_prod(ps.shape(x)[1:]) for x in init_state])
      likelihood_log_prob = likelihood_log_prob_fn(*init_state)

      # Default to the optimal for normal distributed targets.
      # TODO(b/152412213): Revisit this tuning.
      scale_start = (
          tf.constant(2.38 ** 2, dtype=likelihood_log_prob.dtype) /
          tf.constant(dimension, dtype=likelihood_log_prob.dtype))
      # TODO(b/152412213): Enable batch of batches style by using non-scalar
      # inverse_temperature
      inverse_temperature = tf.zeros([], dtype=likelihood_log_prob.dtype)
      scalings = ps.ones_like(likelihood_log_prob) * ps.minimum(scale_start, 1.)
      kernel = make_kernel_fn(
          _make_tempered_target_log_prob_fn(
              prior_log_prob_fn,
              likelihood_log_prob_fn,
              inverse_temperature),
          init_state,
          scalings,
          seed=seed_stream())
      pkr = kernel.bootstrap_results(current_state)
      mh_results = _find_inner_mh_results(pkr)

      particle_info = ParticleInfo(
          accept_prob=ps.ones_like(likelihood_log_prob),
          scalings=scalings,
          tempered_log_prob=mh_results.accepted_results.target_log_prob,
          likelihood_log_prob=likelihood_log_prob,
      )

      return SMCResults(
          num_steps=tf.convert_to_tensor(
              max_num_steps, dtype=tf.int32, name='num_steps'),
          inverse_temperature=inverse_temperature,
          log_marginal_likelihood=tf.constant(
              0., dtype=likelihood_log_prob.dtype),
          particle_info=particle_info
      )

    def update_weights_temperature(inverse_temperature, likelihood_log_prob):
      """Calculate the next inverse temperature and update weights."""

      likelihood_diff = likelihood_log_prob - tf.reduce_max(likelihood_log_prob)

      def _body_fn(new_beta, upper_beta, lower_beta, eff_size, log_weights):
        """One iteration of the temperature and weight update."""
        new_beta = (lower_beta + upper_beta) / 2.0
        log_weights = (new_beta - inverse_temperature) * likelihood_diff
        log_weights_norm = (log_weights -
                            tf.math.reduce_logsumexp(log_weights))
        eff_size = tf.cast(
            tf.exp(-tf.math.reduce_logsumexp(2 * log_weights_norm)), tf.int32)
        upper_beta = tf.where(
            eff_size < effective_sample_size_threshold,
            new_beta, upper_beta)
        lower_beta = tf.where(
            eff_size < effective_sample_size_threshold,
            lower_beta, new_beta)
        return new_beta, upper_beta, lower_beta, eff_size, log_weights

      (new_beta, upper_beta, lower_beta, eff_size, log_weights) = tf.while_loop(  # pylint: disable=unused-variable
          cond=lambda new_beta, upper_beta, lower_beta, eff_size, *_:  # pylint: disable=g-long-lambda
          (upper_beta - lower_beta > 1e-6) &
          (eff_size != effective_sample_size_threshold),
          body=_body_fn,
          loop_vars=(
              tf.zeros_like(inverse_temperature),
              tf.cast(2.0, inverse_temperature.dtype),
              inverse_temperature,
              tf.cast(0, tf.int32),
              tf.zeros_like(likelihood_diff)),
          parallel_iterations=parallel_iterations
          )

      log_weights = tf.where(new_beta < 1.,
                             log_weights,
                             (1. - inverse_temperature) * likelihood_diff)
      marginal_loglike_ = reduce_logmeanexp(
          (new_beta - inverse_temperature) * likelihood_log_prob)

      return marginal_loglike_, tf.clip_by_value(new_beta, 0., 1.), log_weights

    def resample(log_weights, current_state, particle_info, seed=None):
      """Resample particles based on importance weights."""
      with tf.name_scope('resample_particles'):
        seed = SeedStream(seed, salt='resample_particles')
        resampling_indexes = tf.random.categorical(
            [log_weights], ps.reduce_prod(*ps.shape(log_weights)), seed=seed())
        next_state = tf.nest.map_structure(
            lambda x: tf.reshape(tf.gather(x, resampling_indexes), ps.shape(x)),
            current_state)
        next_particle_info = tf.nest.map_structure(
            lambda x: tf.reshape(tf.gather(x, resampling_indexes), ps.shape(x)),
            particle_info)

        return next_state, next_particle_info

    def tuning(num_steps, scalings, accept_prob):
      """Tune scaling and/or num_steps based on the acceptance rate."""
      num_proposed = num_replica * num_steps
      accept_prob = tf.cast(accept_prob, dtype=scalings.dtype)
      avg_scaling = tf.exp(tf.math.log(tf.reduce_mean(scalings))
                           + (tf.reduce_mean(accept_prob) - optimal_accept))
      scalings = 0.5 * (
          avg_scaling +
          tf.exp(tf.math.log(scalings) +
                 (accept_prob - optimal_accept))
      )

      if TUNE_STEPS:
        avg_accept = tf.math.maximum(
            1.0 / tf.cast(num_proposed, dtype=accept_prob.dtype),
            tf.reduce_mean(accept_prob))
        num_steps = tf.clip_by_value(
            tf.cast(
                tf.math.log1p(
                    -tf.cast(target_accept_prob, dtype=avg_accept.dtype)) /
                tf.math.log1p(-avg_accept),
                dtype=num_steps.dtype), 2, max_num_steps)

      return num_steps, scalings

    def mutate(
        current_state,
        scalings,
        num_steps,
        inverse_temperature):
      """Mutate the state using a Transition kernel."""
      with tf.name_scope('mutate_states'):
        kernel = make_kernel_fn(
            _make_tempered_target_log_prob_fn(
                prior_log_prob_fn,
                likelihood_log_prob_fn,
                inverse_temperature),
            current_state,
            scalings,
            seed=seed_stream())
        pkr = kernel.bootstrap_results(current_state)
        mh_results = _find_inner_mh_results(pkr)

        def mutate_onestep(i, state, pkr, accept_count):
          next_state, next_kernel_results = kernel.one_step(state, pkr)
          mh_results = _find_inner_mh_results(pkr)
          # TODO(b/152412213) Cumulate log_acceptance_ratio instead.
          accept_count += tf.cast(mh_results.is_accepted, accept_count.dtype)
          return i+1, next_state, next_kernel_results, accept_count

        (
            _,
            next_state,
            next_kernel_results,
            accept_count
        ) = tf.while_loop(
            cond=lambda i, *args: i < num_steps,
            body=mutate_onestep,
            loop_vars=(
                tf.zeros([], dtype=tf.int32),
                current_state,
                pkr,
                tf.zeros_like(mh_results.is_accepted, tf.float32)),
            parallel_iterations=parallel_iterations
            )
        next_mh_results = _find_inner_mh_results(next_kernel_results)

        return (next_state,
                accept_count / tf.cast(num_steps + 1, accept_count.dtype),
                next_mh_results.accepted_results.target_log_prob)

    pkr = preprocess_state(current_state)
    # Run once
    new_marginal, new_inv_temperature, log_weights = update_weights_temperature(
        pkr.inverse_temperature,
        pkr.particle_info.likelihood_log_prob)
    if PRINT_DEBUG:
      tf.print(
          'Stage:', 0,
          'Beta:', new_inv_temperature,
          'n_steps:', pkr.num_steps,
          'accept:', tf.reduce_mean(
              pkr.particle_info.accept_prob),
          'scaling:', tf.reduce_mean(pkr.particle_info.scalings)
          )
    resampled_state, resampled_particle_info = resample(
        log_weights, current_state, pkr.particle_info, seed_stream())
    next_state, acceptance_rate, tempered_log_prob = mutate(
        resampled_state,
        resampled_particle_info.scalings,
        pkr.num_steps,
        new_inv_temperature)
    next_pkr = SMCResults(
        num_steps=pkr.num_steps,
        inverse_temperature=new_inv_temperature,
        log_marginal_likelihood=pkr.log_marginal_likelihood + new_marginal,
        particle_info=ParticleInfo(
            accept_prob=acceptance_rate,
            scalings=resampled_particle_info.scalings,
            tempered_log_prob=tempered_log_prob,
            likelihood_log_prob=likelihood_log_prob_fn(*next_state),
        ))

    # Stage > 0
    def smc_body_fn(stage, state, smc_kernel_result):
      """Run one stage of SMC with constant temperature."""
      (
          new_marginal,
          new_inv_temperature,
          log_weights
      ) = update_weights_temperature(
          smc_kernel_result.inverse_temperature,
          smc_kernel_result.particle_info.likelihood_log_prob)
      # TODO(b/152412213) Use a tf.scan to better collect debug info.
      if PRINT_DEBUG:
        tf.print(
            'Stage:', stage,
            'Beta:', new_inv_temperature,
            'n_steps:', smc_kernel_result.num_steps,
            'accept:', tf.reduce_mean(
                smc_kernel_result.particle_info.accept_prob),
            'scaling:', tf.reduce_mean(smc_kernel_result.particle_info.scalings)
            )
      resampled_state, resampled_particle_info = resample(
          log_weights, state, smc_kernel_result.particle_info, seed_stream())
      num_steps, scalings = tuning(
          smc_kernel_result.num_steps,
          resampled_particle_info.scalings,
          resampled_particle_info.accept_prob)
      next_state, acceptance_rate, tempered_log_prob = mutate(
          resampled_state, scalings, num_steps, new_inv_temperature)
      next_pkr = SMCResults(
          num_steps=num_steps,
          inverse_temperature=new_inv_temperature,
          log_marginal_likelihood=(new_marginal +
                                   smc_kernel_result.log_marginal_likelihood),
          particle_info=ParticleInfo(
              accept_prob=acceptance_rate,
              scalings=scalings,
              tempered_log_prob=tempered_log_prob,
              likelihood_log_prob=likelihood_log_prob_fn(*next_state),
          ))
      return stage + 1, next_state, next_pkr

    (
        n_stage,
        final_state,
        final_kernel_results
    ) = tf.while_loop(
        cond=lambda i, state, pkr: (i < max_stage) & (  # pylint: disable=g-long-lambda
            pkr.inverse_temperature < 1.
        ),
        body=smc_body_fn,
        loop_vars=(
            tf.ones([], dtype=tf.int32),
            next_state,
            next_pkr),
        parallel_iterations=parallel_iterations
        )
    if unwrap_state_list:
      final_state = final_state[0]
    return n_stage, final_state, final_kernel_results
