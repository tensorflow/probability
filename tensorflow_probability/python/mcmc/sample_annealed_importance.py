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
"""Markov chain Monte Carlo driver, `sample_chain_annealed_importance_chain`."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
# Dependency imports
import numpy as np

import tensorflow as tf
from tensorflow_probability.python.mcmc.internal import util as mcmc_util


__all__ = [
    "sample_annealed_importance_chain",
]


AISResults = collections.namedtuple(
    "AISResults",
    [
        "proposal_log_prob",
        "target_log_prob",
        "inner_results",
    ])


def sample_annealed_importance_chain(
    num_steps,
    proposal_log_prob_fn,
    target_log_prob_fn,
    current_state,
    make_kernel_fn,
    parallel_iterations=10,
    name=None):
  """Runs annealed importance sampling (AIS) to estimate normalizing constants.

  This function uses an MCMC transition operator (e.g., Hamiltonian Monte Carlo)
  to sample from a series of distributions that slowly interpolates between
  an initial "proposal" distribution:

  `exp(proposal_log_prob_fn(x) - proposal_log_normalizer)`

  and the target distribution:

  `exp(target_log_prob_fn(x) - target_log_normalizer)`,

  accumulating importance weights along the way. The product of these
  importance weights gives an unbiased estimate of the ratio of the
  normalizing constants of the initial distribution and the target
  distribution:

  `E[exp(ais_weights)] = exp(target_log_normalizer - proposal_log_normalizer)`.

  Note: When running in graph mode, `proposal_log_prob_fn` and
  `target_log_prob_fn` are called exactly three times (although this may be
  reduced to two times in the future).

  Args:
    num_steps: Integer number of Markov chain updates to run. More
      iterations means more expense, but smoother annealing between q
      and p, which in turn means exponentially lower variance for the
      normalizing constant estimator.
    proposal_log_prob_fn: Python callable that returns the log density of the
      initial distribution.
    target_log_prob_fn: Python callable which takes an argument like
      `current_state` (or `*current_state` if it's a list) and returns its
      (possibly unnormalized) log-density under the target distribution.
    current_state: `Tensor` or Python `list` of `Tensor`s representing the
      current state(s) of the Markov chain(s). The first `r` dimensions index
      independent chains, `r = tf.rank(target_log_prob_fn(*current_state))`.
    make_kernel_fn: Python `callable` which returns a `TransitionKernel`-like
      object. Must take one argument representing the `TransitionKernel`'s
      `target_log_prob_fn`. The `target_log_prob_fn` argument represents the
      `TransitionKernel`'s target log distribution.  Note:
      `sample_annealed_importance_chain` creates a new `target_log_prob_fn`
      which is an interpolation between the supplied `target_log_prob_fn` and
      `proposal_log_prob_fn`; it is this interpolated function which is used as
      an argument to `make_kernel_fn`.
    parallel_iterations: The number of iterations allowed to run in parallel.
        It must be a positive integer. See `tf.while_loop` for more details.
    name: Python `str` name prefixed to Ops created by this function.
      Default value: `None` (i.e., "sample_annealed_importance_chain").

  Returns:
    next_state: `Tensor` or Python list of `Tensor`s representing the
      state(s) of the Markov chain(s) at the final iteration. Has same shape as
      input `current_state`.
    ais_weights: Tensor with the estimated weight(s). Has shape matching
      `target_log_prob_fn(current_state)`.
    kernel_results: `collections.namedtuple` of internal calculations used to
      advance the chain.

  #### Examples

  ##### Estimate the normalizing constant of a log-gamma distribution.

  ```python
  tfd = tfp.distributions

  # Run 100 AIS chains in parallel
  num_chains = 100
  dims = 20
  dtype = np.float32

  proposal = tfd.MultivariateNormalDiag(
     loc=tf.zeros([dims], dtype=dtype))

  target = tfd.TransformedDistribution(
    distribution=tfd.Gamma(concentration=dtype(2),
                           rate=dtype(3)),
    bijector=tfp.bijectors.Invert(tfp.bijectors.Exp()),
    event_shape=[dims])

  chains_state, ais_weights, kernels_results = (
      tfp.mcmc.sample_annealed_importance_chain(
          num_steps=1000,
          proposal_log_prob_fn=proposal.log_prob,
          target_log_prob_fn=target.log_prob,
          current_state=proposal.sample(num_chains),
          make_kernel_fn=lambda tlp_fn: tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=tlp_fn,
            step_size=0.2,
            num_leapfrog_steps=2)))

  log_estimated_normalizer = (tf.reduce_logsumexp(ais_weights)
                              - np.log(num_chains))
  log_true_normalizer = tf.lgamma(2.) - 2. * tf.log(3.)
  ```

  ##### Estimate marginal likelihood of a Bayesian regression model.

  ```python
  tfd = tfp.distributions

  def make_prior(dims, dtype):
    return tfd.MultivariateNormalDiag(
        loc=tf.zeros(dims, dtype))

  def make_likelihood(weights, x):
    return tfd.MultivariateNormalDiag(
        loc=tf.tensordot(weights, x, axes=[[0], [-1]]))

  # Run 100 AIS chains in parallel
  num_chains = 100
  dims = 10
  dtype = np.float32

  # Make training data.
  x = np.random.randn(num_chains, dims).astype(dtype)
  true_weights = np.random.randn(dims).astype(dtype)
  y = np.dot(x, true_weights) + np.random.randn(num_chains)

  # Setup model.
  prior = make_prior(dims, dtype)
  def target_log_prob_fn(weights):
    return prior.log_prob(weights) + make_likelihood(weights, x).log_prob(y)

  proposal = tfd.MultivariateNormalDiag(
      loc=tf.zeros(dims, dtype))

  weight_samples, ais_weights, kernel_results = (
      tfp.mcmc.sample_annealed_importance_chain(
        num_steps=1000,
        proposal_log_prob_fn=proposal.log_prob,
        target_log_prob_fn=target_log_prob_fn
        current_state=tf.zeros([num_chains, dims], dtype),
        make_kernel_fn=lambda tlp_fn: tfp.mcmc.HamiltonianMonteCarlo(
          target_log_prob_fn=tlp_fn,
          step_size=0.1,
          num_leapfrog_steps=2)))
  log_normalizer_estimate = (tf.reduce_logsumexp(ais_weights)
                             - np.log(num_chains))
  ```

  """
  with tf.compat.v1.name_scope(name, "sample_annealed_importance_chain",
                               [num_steps, current_state]):
    num_steps = tf.convert_to_tensor(
        value=num_steps, dtype=tf.int32, name="num_steps")
    if mcmc_util.is_list_like(current_state):
      current_state = [
          tf.convert_to_tensor(value=s, name="current_state")
          for s in current_state
      ]
    else:
      current_state = tf.convert_to_tensor(
          value=current_state, name="current_state")

    def _make_convex_combined_log_prob_fn(iter_):
      def _fn(*args):
        p = tf.identity(proposal_log_prob_fn(*args), name="proposal_log_prob")
        t = tf.identity(target_log_prob_fn(*args), name="target_log_prob")
        dtype = p.dtype.base_dtype
        beta = tf.cast(iter_ + 1, dtype) / tf.cast(num_steps, dtype)
        return tf.identity(beta * t + (1. - beta) * p,
                           name="convex_combined_log_prob")
      return _fn

    def _loop_body(iter_, ais_weights, current_state, kernel_results):
      """Closure which implements `tf.while_loop` body."""
      x = (current_state if mcmc_util.is_list_like(current_state)
           else [current_state])
      proposal_log_prob = proposal_log_prob_fn(*x)
      target_log_prob = target_log_prob_fn(*x)
      ais_weights += ((target_log_prob - proposal_log_prob) /
                      tf.cast(num_steps, ais_weights.dtype))
      kernel = make_kernel_fn(_make_convex_combined_log_prob_fn(iter_))
      next_state, inner_results = kernel.one_step(
          current_state, kernel_results.inner_results)
      kernel_results = AISResults(
          proposal_log_prob=proposal_log_prob,
          target_log_prob=target_log_prob,
          inner_results=inner_results,
      )
      return [iter_ + 1, ais_weights, next_state, kernel_results]

    def _bootstrap_results(init_state):
      """Creates first version of `previous_kernel_results`."""
      kernel = make_kernel_fn(_make_convex_combined_log_prob_fn(iter_=0))
      inner_results = kernel.bootstrap_results(init_state)

      convex_combined_log_prob = inner_results.accepted_results.target_log_prob
      dtype = convex_combined_log_prob.dtype.as_numpy_dtype
      shape = tf.shape(input=convex_combined_log_prob)
      proposal_log_prob = tf.fill(shape, dtype(np.nan),
                                  name="bootstrap_proposal_log_prob")
      target_log_prob = tf.fill(shape, dtype(np.nan),
                                name="target_target_log_prob")

      return AISResults(
          proposal_log_prob=proposal_log_prob,
          target_log_prob=target_log_prob,
          inner_results=inner_results,
      )

    previous_kernel_results = _bootstrap_results(current_state)
    inner_results = previous_kernel_results.inner_results

    ais_weights = tf.zeros(
        shape=tf.broadcast_dynamic_shape(
            tf.shape(input=inner_results.proposed_results.target_log_prob),
            tf.shape(input=inner_results.accepted_results.target_log_prob)),
        dtype=inner_results.proposed_results.target_log_prob.dtype.base_dtype)

    [_, ais_weights, current_state, kernel_results] = tf.while_loop(
        cond=lambda iter_, *args: iter_ < num_steps,
        body=_loop_body,
        loop_vars=[
            np.int32(0),  # iter_
            ais_weights,
            current_state,
            previous_kernel_results,
        ],
        parallel_iterations=parallel_iterations)

    return [current_state, ais_weights, kernel_results]
