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

import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.experimental.mcmc import weighted_resampling
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.math.generic import log1mexp
from tensorflow_probability.python.math.generic import log_add_exp
from tensorflow_probability.python.math.generic import reduce_logmeanexp
from tensorflow_probability.python.mcmc import hmc
from tensorflow_probability.python.mcmc import random_walk_metropolis
from tensorflow_probability.python.mcmc import transformed_kernel
from tensorflow_probability.python.mcmc.internal import util as mcmc_util
from tensorflow_probability.python.util.seed_stream import SeedStream


__all__ = [
    'default_make_hmc_kernel_fn',
    'gen_make_hmc_kernel_fn',
    'gen_make_transform_hmc_kernel_fn',
    'make_rwmh_kernel_fn',
    'sample_sequential_monte_carlo',
    'simple_heuristic_tuning',
]


PRINT_DEBUG = False

ParticleInfo = collections.namedtuple(
    'ParticleInfo',
    [
        'log_accept_prob',  # log acceptance probability per particle
        'log_scalings',
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


def gather_mh_like_result(results):
  """Gather log_accept_ratio and target_log_prob from kernel result."""
  # For MH kernel result.
  if (hasattr(results, 'proposed_results')
      and hasattr(results, 'accepted_results')):
    return results.log_accept_ratio, results.accepted_results.target_log_prob
  # For NUTS kernel result.
  if (hasattr(results, 'log_accept_ratio')
      and hasattr(results, 'target_log_prob')):
    return results.log_accept_ratio, results.target_log_prob
  # For TransformTransitionKernel Result.
  if hasattr(results, 'inner_results'):
    return gather_mh_like_result(results.inner_results)
  raise TypeError('Cannot find MH results.')


def default_make_tempered_target_log_prob_fn(
    prior_log_prob_fn, likelihood_log_prob_fn, inverse_temperatures):
  """Helper which creates inner kernel target_log_prob_fn."""
  def _tempered_target_log_prob(*args):
    priorlogprob = tf.identity(prior_log_prob_fn(*args),
                               name='prior_log_prob')
    loglike = tf.identity(likelihood_log_prob_fn(*args),
                          name='likelihood_log_prob')
    return tf.identity(priorlogprob + loglike * inverse_temperatures,
                       name='tempered_logp')
  return _tempered_target_log_prob


def make_rwmh_kernel_fn(target_log_prob_fn, init_state, scalings, seed=None):
  """Generate a Random Walk MH kernel."""
  with tf.name_scope('make_rwmh_kernel_fn'):
    seed = SeedStream(seed, salt='make_rwmh_kernel_fn')
    state_std = [
        tf.math.reduce_std(x, axis=0, keepdims=True)
        for x in init_state
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
      # TransformedTransitionKernel doesn't modify the input step size, thus we
      # need to pass the appropriate step size that are already in unconstrained
      # space
      state_std = [
          tf.math.reduce_std(bij.inverse(x), axis=0, keepdims=True)
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


def simple_heuristic_tuning(num_steps,
                            log_scalings,
                            log_accept_prob,
                            optimal_accept=0.234,
                            target_accept_prob=0.99,
                            name=None):
  """Tune the number of steps and scaling of one mutation.

  # TODO(b/152412213): Better explanation of the heuristic used here.

  This is a simple heuristic for tuning the number of steps of the next
  mutation, as well as the scaling of a transition kernel (e.g., step size in
  HMC, scale of a Normal proposal in RWMH) using the acceptance probability from
  the previous mutation stage in SMC.

  Args:
    num_steps: The initial number of steps for the next mutation, to be tune.
    log_scalings: The log of the scale of the proposal kernel
    log_accept_prob: The log of the acceptance ratio from the last mutation.
    optimal_accept: Optimal acceptance ratio for a Transitional Kernel. Default
      value is 0.234 (Optimal for Random Walk Metropolis kernel).
    target_accept_prob: Target acceptance probability at the end of one mutation
      step. Default value: 0.99
    name: Python `str` name prefixed to Ops created by this function.
      Default value: `None`.

  Returns:
    num_steps: The number of steps for the next mutation.
    new_log_scalings: The log of the scale of the proposal kernel for the next
      mutation.

  """
  with tf.name_scope(name or 'simple_heuristic_tuning'):
    optimal_accept = tf.constant(optimal_accept, dtype=log_accept_prob.dtype)
    target_accept_prob = tf.constant(
        target_accept_prob, dtype=log_accept_prob.dtype)
    log_half_constant = tf.constant(np.log(.5), dtype=log_scalings.dtype)

    avg_log_scalings = reduce_logmeanexp(log_scalings, axis=0)
    avg_log_accept_prob = reduce_logmeanexp(
        log_accept_prob, axis=0)

    avg_log_scaling_target = avg_log_scalings + (
        tf.exp(avg_log_accept_prob) - optimal_accept)
    new_log_scalings = log_half_constant + log_add_exp(
        avg_log_scaling_target,
        log_scalings + (tf.exp(log_accept_prob) - optimal_accept)
        )

    num_particles = ps.shape(log_accept_prob)[-1]
    num_proposed = tf.cast(
        num_particles * num_steps, dtype=avg_log_accept_prob.dtype)
    # max(1/num_proposed, average_accept_ratio)
    log_avg_accept = tf.math.maximum(-tf.math.log(num_proposed),
                                     avg_log_accept_prob)
    num_steps = tf.cast(
        tf.math.log1p(-target_accept_prob) / log1mexp(log_avg_accept),
        dtype=num_steps.dtype)
    # We choose the number of steps from the batch that takes the longest,
    # hence this is a reduce over all axes.
    max_step_across_batch = tf.reduce_max(num_steps)
    return max_step_across_batch, new_log_scalings


# TODO(b/152412213) Experitment to improve recommendation on static parmaeters
def sample_sequential_monte_carlo(
    prior_log_prob_fn,
    likelihood_log_prob_fn,
    current_state,
    min_num_steps=2,
    max_num_steps=25,
    max_stage=100,
    make_kernel_fn=make_rwmh_kernel_fn,
    tuning_fn=simple_heuristic_tuning,
    make_tempered_target_log_prob_fn=default_make_tempered_target_log_prob_fn,
    resample_fn=weighted_resampling.resample_systematic,
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
  known as Particle Filter in some literature. The current implemenetation is
  largely based on Del Moral et al [1], which adapts the tempering sequence
  adaptively (base on the effective sample size) and the scaling of the mutation
  kernel (base on the sample covariance of the particles) at each stage.

  Args:
    prior_log_prob_fn: Python callable that returns the log density of the
      prior distribution.
    likelihood_log_prob_fn: Python callable which takes an argument like
      `current_state` (or `*current_state` if it's a list) and returns its
      (possibly unnormalized) log-density under the likelihood distribution.
    current_state: Nested structure of `Tensor`s, each of shape
      `concat([[num_particles, b1, ..., bN], latent_part_event_shape])`, where
      `b1, ..., bN` are optional batch dimensions. Each batch represents an
      independent SMC run.
    min_num_steps: The minimal number of kernel transition steps in one mutation
      of the MC samples.
    max_num_steps: The maximum number of kernel transition steps in one mutation
      of the MC samples. Note that the actual number of steps in one mutation is
      tuned during sampling and likely lower than the max_num_step.
    max_stage: Integer number of the stage for increasing the temperature
      from 0 to 1.
    make_kernel_fn: Python `callable` which returns a `TransitionKernel`-like
      object. Must take one argument representing the `TransitionKernel`'s
      `target_log_prob_fn`. The `target_log_prob_fn` argument represents the
      `TransitionKernel`'s target log distribution.  Note:
      `sample_sequential_monte_carlo` creates a new `target_log_prob_fn`
      which is an interpolation between the supplied `target_log_prob_fn` and
      `proposal_log_prob_fn`; it is this interpolated function which is used as
      an argument to `make_kernel_fn`.
    tuning_fn: Python `callable` which takes the number of steps, the log
      scaling, and the log acceptance ratio from the last mutation and output
      the number of steps and log scaling for the next mutation.
    make_tempered_target_log_prob_fn: Python `callable` that takes the
      `prior_log_prob_fn`, `likelihood_log_prob_fn`, and `inverse_temperatures`
      and creates a `target_log_prob_fn` `callable` that pass to
      `make_kernel_fn`.
    resample_fn: Python `callable` to generate the indices of resampled
      particles, given their weights. Generally, one of
      `tfp.experimental.mcmc.resample_independent` or
      `tfp.experimental.mcmc.resample_systematic`, or any function
      with the same signature.
      Default value: `tfp.experimental.mcmc.resample_systematic`.
    ess_threshold_ratio: Target ratio for effective sample size.
    parallel_iterations: The number of iterations allowed to run in parallel.
        It must be a positive integer. See `tf.while_loop` for more details.
    seed: Python integer or TFP seedstream to seed the random number generator.
    name: Python `str` name prefixed to Ops created by this function.
      Default value: `None` (i.e., 'sample_sequential_monte_carlo').

  Returns:
    n_stage: Number of the mutation stage SMC ran.
    final_state: `Tensor` or Python `list` of `Tensor`s representing the
      final state(s) of the Markov chain(s). The output are the posterior
      samples.
    final_kernel_results: `collections.namedtuple` of internal calculations used
      to advance the chain.

  #### References

  [1] Del Moral, Pierre, Arnaud Doucet, and Ajay Jasra. An adaptive sequential
      Monte Carlo method for approximate Bayesian computation.
      _Statistics and Computing_, 22.5(1009-1020), 2012.

  """

  with tf.name_scope(name or 'sample_sequential_monte_carlo'):
    seed_stream = SeedStream(seed, salt='smc_seed')

    unwrap_state_list = not tf.nest.is_nested(current_state)
    if unwrap_state_list:
      current_state = [current_state]
    current_state = [
        tf.convert_to_tensor(s, dtype_hint=tf.float32) for s in current_state
    ]

    # Initial preprocessing at Stage 0
    likelihood_log_prob = likelihood_log_prob_fn(*current_state)

    likelihood_rank = ps.rank(likelihood_log_prob)
    dimension = ps.reduce_sum([
        ps.reduce_prod(ps.shape(x)[likelihood_rank:]) for x in current_state])

    # We infer the particle shapes from the resulting likelihood:
    # [num_particles, b1, ..., bN]
    particle_shape = ps.shape(likelihood_log_prob)
    num_particles, batch_shape = particle_shape[0], particle_shape[1:]
    effective_sample_size_threshold = tf.cast(
        num_particles * ess_threshold_ratio, tf.int32)

    # TODO(b/152412213): Revisit this default parameter.
    # Default to the optimal scaling of a random walk kernel for a d-dimensional
    # normal distributed targets: 2.38 ** 2 / d.
    # For more detail see:
    # Roberts GO, Gelman A, Gilks WR. Weak convergence and optimal scaling of
    # random walk Metropolis algorithms. _The annals of applied probability_.
    # 1997;7(1):110-20.
    scale_start = (
        tf.constant(2.38 ** 2, dtype=likelihood_log_prob.dtype) /
        tf.constant(dimension, dtype=likelihood_log_prob.dtype))

    inverse_temperature = tf.zeros(batch_shape, dtype=likelihood_log_prob.dtype)
    scalings = ps.ones_like(likelihood_log_prob) * ps.minimum(scale_start, 1.)
    kernel = make_kernel_fn(
        make_tempered_target_log_prob_fn(
            prior_log_prob_fn,
            likelihood_log_prob_fn,
            inverse_temperature),
        current_state,
        scalings,
        seed=seed_stream)
    pkr = kernel.bootstrap_results(current_state)
    _, kernel_target_log_prob = gather_mh_like_result(pkr)

    particle_info = ParticleInfo(
        log_accept_prob=ps.zeros_like(likelihood_log_prob),
        log_scalings=tf.math.log(scalings),
        tempered_log_prob=kernel_target_log_prob,
        likelihood_log_prob=likelihood_log_prob,
    )

    current_pkr = SMCResults(
        num_steps=tf.convert_to_tensor(
            max_num_steps, dtype=tf.int32, name='num_steps'),
        inverse_temperature=inverse_temperature,
        log_marginal_likelihood=tf.zeros_like(inverse_temperature),
        particle_info=particle_info
    )

    def update_weights_temperature(inverse_temperature, likelihood_log_prob):
      """Calculate the next inverse temperature and update weights."""
      likelihood_diff = likelihood_log_prob - tf.reduce_max(
          likelihood_log_prob, axis=0)

      def _body_fn(new_beta, upper_beta, lower_beta, eff_size, log_weights):
        """One iteration of the temperature and weight update."""
        new_beta = (lower_beta + upper_beta) / 2.0
        log_weights = (new_beta - inverse_temperature) * likelihood_diff
        log_weights_norm = tf.math.log_softmax(log_weights, axis=0)
        eff_size = tf.cast(
            tf.exp(-tf.math.reduce_logsumexp(2 * log_weights_norm, axis=0)),
            tf.int32)
        upper_beta = tf.where(
            eff_size < effective_sample_size_threshold,
            new_beta, upper_beta)
        lower_beta = tf.where(
            eff_size < effective_sample_size_threshold,
            lower_beta, new_beta)
        return new_beta, upper_beta, lower_beta, eff_size, log_weights

      def _cond_fn(new_beta, upper_beta, lower_beta, eff_size, *_):  # pylint: disable=unused-argument
        # TODO(junpenglao): revisit threshold below to be dtype specific.
        threshold = 1e-6
        return (
            tf.math.reduce_any(upper_beta - lower_beta > threshold) &
            tf.math.reduce_any(eff_size != effective_sample_size_threshold)
            )

      (new_beta, upper_beta, lower_beta, eff_size, log_weights) = tf.while_loop(  # pylint: disable=unused-variable
          cond=_cond_fn,
          body=_body_fn,
          loop_vars=(
              tf.zeros_like(inverse_temperature),
              tf.fill(
                  ps.shape(inverse_temperature),
                  tf.constant(2, inverse_temperature.dtype)),
              inverse_temperature,
              tf.zeros_like(inverse_temperature, dtype=tf.int32),
              tf.zeros_like(likelihood_diff)),
          parallel_iterations=parallel_iterations
          )

      log_weights = tf.where(new_beta < 1.,
                             log_weights,
                             (1. - inverse_temperature) * likelihood_diff)
      marginal_loglike_ = reduce_logmeanexp(
          (new_beta - inverse_temperature) * likelihood_log_prob, axis=0)
      new_inverse_temperature = tf.clip_by_value(new_beta, 0., 1.)

      return marginal_loglike_, new_inverse_temperature, log_weights

    def mutate(
        current_state,
        log_scalings,
        num_steps,
        inverse_temperature):
      """Mutate the state using a Transition kernel."""
      with tf.name_scope('mutate_states'):
        scalings = tf.exp(log_scalings)
        kernel = make_kernel_fn(
            make_tempered_target_log_prob_fn(
                prior_log_prob_fn,
                likelihood_log_prob_fn,
                inverse_temperature),
            current_state,
            scalings,
            seed=seed_stream)
        pkr = kernel.bootstrap_results(current_state)
        kernel_log_accept_ratio, _ = gather_mh_like_result(pkr)

        def mutate_onestep(i, state, pkr, log_accept_prob_sum):
          next_state, next_kernel_results = kernel.one_step(state, pkr)
          kernel_log_accept_ratio, _ = gather_mh_like_result(pkr)
          log_accept_prob = tf.minimum(kernel_log_accept_ratio, 0.)
          log_accept_prob_sum = log_add_exp(
              log_accept_prob_sum, log_accept_prob)
          return i + 1, next_state, next_kernel_results, log_accept_prob_sum

        (
            _,
            next_state,
            next_kernel_results,
            log_accept_prob_sum
        ) = tf.while_loop(
            cond=lambda i, *args: i < num_steps,
            body=mutate_onestep,
            loop_vars=(
                tf.zeros([], dtype=tf.int32),
                current_state,
                pkr,
                # we accumulate the acceptance probability in log space.
                tf.fill(
                    ps.shape(kernel_log_accept_ratio),
                    tf.constant(-np.inf, kernel_log_accept_ratio.dtype))
                ),
            parallel_iterations=parallel_iterations
            )
        _, kernel_target_log_prob = gather_mh_like_result(next_kernel_results)
        avg_log_accept_prob_per_particle = log_accept_prob_sum - tf.math.log(
            tf.cast(num_steps + 1, log_accept_prob_sum.dtype))
        return (next_state,
                avg_log_accept_prob_per_particle,
                kernel_target_log_prob)

    # One SMC steps.
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
            'accept:', tf.exp(reduce_logmeanexp(
                smc_kernel_result.particle_info.log_accept_prob, axis=0)),
            'scaling:', tf.exp(reduce_logmeanexp(
                smc_kernel_result.particle_info.log_scalings, axis=0))
            )
      (resampled_state,
       resampled_particle_info), _ = weighted_resampling.resample(
           particles=(state, smc_kernel_result.particle_info),
           log_weights=log_weights,
           resample_fn=resample_fn,
           seed=seed_stream)
      next_num_steps, next_log_scalings = tuning_fn(
          smc_kernel_result.num_steps,
          resampled_particle_info.log_scalings,
          resampled_particle_info.log_accept_prob)
      # Skip tuning at stage 0.
      next_num_steps = tf.where(stage == 0,
                                smc_kernel_result.num_steps,
                                next_num_steps)
      next_log_scalings = tf.where(stage == 0,
                                   resampled_particle_info.log_scalings,
                                   next_log_scalings)
      next_num_steps = tf.clip_by_value(
          next_num_steps, min_num_steps, max_num_steps)

      next_state, log_accept_prob, tempered_log_prob = mutate(
          resampled_state,
          next_log_scalings,
          next_num_steps,
          new_inv_temperature)
      next_pkr = SMCResults(
          num_steps=next_num_steps,
          inverse_temperature=new_inv_temperature,
          log_marginal_likelihood=(new_marginal +
                                   smc_kernel_result.log_marginal_likelihood),
          particle_info=ParticleInfo(
              log_accept_prob=log_accept_prob,
              log_scalings=next_log_scalings,
              tempered_log_prob=tempered_log_prob,
              likelihood_log_prob=likelihood_log_prob_fn(*next_state),
          ))
      return stage + 1, next_state, next_pkr

    (
        n_stage,
        final_state,
        final_kernel_results
    ) = tf.while_loop(
        cond=lambda i, state, pkr: (  # pylint: disable=g-long-lambda
            (i < max_stage) &
            tf.reduce_any(pkr.inverse_temperature < 1.)),
        body=smc_body_fn,
        loop_vars=(
            tf.zeros([], dtype=tf.int32),
            current_state,
            current_pkr),
        parallel_iterations=parallel_iterations
        )
    if unwrap_state_list:
      final_state = final_state[0]
    return n_stage, final_state, final_kernel_results
