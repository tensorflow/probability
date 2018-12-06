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
"""Markov chain Monte Carlo drivers.

@@sample_chain
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
# Dependency imports

import tensorflow as tf
from tensorflow_probability.python.mcmc import util as mcmc_util


__all__ = [
    "sample_chain",
]

# Cause all warnings to always be triggered.
# Not having this means subsequent calls wont trigger the warning.
warnings.simplefilter("always")


def sample_chain(
    num_results,
    current_state,
    previous_kernel_results=None,
    kernel=None,
    num_burnin_steps=0,
    num_steps_between_results=0,
    parallel_iterations=10,
    name=None):
  """Implements Markov chain Monte Carlo via repeated `TransitionKernel` steps.

  This function samples from an Markov chain at `current_state` and whose
  stationary distribution is governed by the supplied `TransitionKernel`
  instance (`kernel`).

  This function can sample from multiple chains, in parallel. (Whether or not
  there are multiple chains is dictated by the `kernel`.)

  The `current_state` can be represented as a single `Tensor` or a `list` of
  `Tensors` which collectively represent the current state.

  Since MCMC states are correlated, it is sometimes desirable to produce
  additional intermediate states, and then discard them, ending up with a set of
  states with decreased autocorrelation.  See [Owen (2017)][1]. Such "thinning"
  is made possible by setting `num_steps_between_results > 0`. The chain then
  takes `num_steps_between_results` extra steps between the steps that make it
  into the results. The extra steps are never materialized (in calls to
  `sess.run`), and thus do not increase memory requirements.

  Warning: when setting a `seed` in the `kernel`, ensure that `sample_chain`'s
  `parallel_iterations=1`, otherwise results will not be reproducible.

  Args:
    num_results: Integer number of Markov chain draws.
    current_state: `Tensor` or Python `list` of `Tensor`s representing the
      current state(s) of the Markov chain(s).
    previous_kernel_results: A (possibly nested) `tuple`, `namedtuple` or
      `list` of `Tensor`s representing internal calculations made within the
      previous call to this function (or as returned by `bootstrap_results`).
    kernel: An instance of `tfp.mcmc.TransitionKernel` which implements one step
      of the Markov chain.
    num_burnin_steps: Integer number of chain steps to take before starting to
      collect results.
      Default value: 0 (i.e., no burn-in).
    num_steps_between_results: Integer number of chain steps between collecting
      a result. Only one out of every `num_steps_between_samples + 1` steps is
      included in the returned results.  The number of returned chain states is
      still equal to `num_results`.  Default value: 0 (i.e., no thinning).
    parallel_iterations: The number of iterations allowed to run in parallel.
        It must be a positive integer. See `tf.while_loop` for more details.
    name: Python `str` name prefixed to Ops created by this function.
      Default value: `None` (i.e., "mcmc_sample_chain").

  Returns:
    next_states: Tensor or Python list of `Tensor`s representing the
      state(s) of the Markov chain(s) at each result step. Has same shape as
      input `current_state` but with a prepended `num_results`-size dimension.
    kernel_results: A (possibly nested) `tuple`, `namedtuple` or `list` of
      `Tensor`s representing internal calculations made within this function.

  #### Examples

  ##### Sample from a diagonal-variance Gaussian.

  ```python
  import tensorflow as tf
  import tensorflow_probability as tfp
  tfd = tfp.distributions

  def make_likelihood(true_variances):
    return tfd.MultivariateNormalDiag(
        scale_diag=tf.sqrt(true_variances))

  dims = 10
  dtype = np.float32
  true_variances = tf.linspace(dtype(1), dtype(3), dims)
  likelihood = make_likelihood(true_variances)

  states, kernel_results = tfp.mcmc.sample_chain(
      num_results=1000,
      current_state=tf.zeros(dims),
      kernel=tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=likelihood.log_prob,
        step_size=0.5,
        num_leapfrog_steps=2),
      num_burnin_steps=500)

  # Compute sample stats.
  sample_mean = tf.reduce_mean(states, axis=0)
  sample_var = tf.reduce_mean(
      tf.squared_difference(states, sample_mean),
      axis=0)
  ```

  ##### Sampling from factor-analysis posteriors with known factors.

  I.e.,

  ```none
  for i=1..n:
    w[i] ~ Normal(0, eye(d))            # prior
    x[i] ~ Normal(loc=matmul(w[i], F))  # likelihood
  ```

  where `F` denotes factors.

  ```python
  import tensorflow as tf
  import tensorflow_probability as tfp
  tfd = tfp.distributions

  def make_prior(dims, dtype):
    return tfd.MultivariateNormalDiag(
        loc=tf.zeros(dims, dtype))

  def make_likelihood(weights, factors):
    return tfd.MultivariateNormalDiag(
        loc=tf.tensordot(weights, factors, axes=[[0], [-1]]))

  # Setup data.
  num_weights = 10
  num_factors = 4
  num_chains = 100
  dtype = np.float32

  prior = make_prior(num_weights, dtype)
  weights = prior.sample(num_chains)
  factors = np.random.randn(num_factors, num_weights).astype(dtype)
  x = make_likelihood(weights, factors).sample(num_chains)

  def target_log_prob(w):
    # Target joint is: `f(w) = p(w, x | factors)`.
    return prior.log_prob(w) + make_likelihood(w, factors).log_prob(x)

  # Get `num_results` samples from `num_chains` independent chains.
  chains_states, kernels_results = tfp.mcmc.sample_chain(
      num_results=1000,
      current_state=tf.zeros([num_chains, dims], dtype),
      kernel=tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=target_log_prob,
        step_size=0.1,
        num_leapfrog_steps=2),
      num_burnin_steps=500)

  # Compute sample stats.
  sample_mean = tf.reduce_mean(chains_states, axis=[0, 1])
  sample_var = tf.reduce_mean(
      tf.squared_difference(chains_states, sample_mean),
      axis=[0, 1])
  ```

  #### References

  [1]: Art B. Owen. Statistically efficient thinning of a Markov chain sampler.
       _Technical Report_, 2017.
       http://statweb.stanford.edu/~owen/reports/bestthinning.pdf
  """
  if not kernel.is_calibrated:
    warnings.warn("Supplied `TransitionKernel` is not calibrated. Markov "
                  "chain may not converge to intended target distribution.")
  with tf.name_scope(
      name, "mcmc_sample_chain",
      [num_results, num_burnin_steps, num_steps_between_results]):
    num_results = tf.convert_to_tensor(
        num_results,
        dtype=tf.int32,
        name="num_results")
    num_burnin_steps = tf.convert_to_tensor(
        num_burnin_steps,
        dtype=tf.int64,
        name="num_burnin_steps")
    num_steps_between_results = tf.convert_to_tensor(
        num_steps_between_results,
        dtype=tf.int64,
        name="num_steps_between_results")

    if mcmc_util.is_list_like(current_state):
      current_state = [tf.convert_to_tensor(s, name="current_state")
                       for s in current_state]
    else:
      current_state = tf.convert_to_tensor(current_state, name="current_state")

    def _scan_body(args_list, num_steps):
      """Closure which implements `tf.scan` body."""
      next_state, current_kernel_results = mcmc_util.smart_for_loop(
          loop_num_iter=num_steps,
          body_fn=kernel.one_step,
          initial_loop_vars=args_list,
          parallel_iterations=parallel_iterations
      )
      return [next_state, current_kernel_results]

    if previous_kernel_results is None:
      previous_kernel_results = kernel.bootstrap_results(current_state)

    return tf.scan(
        fn=_scan_body,
        elems=tf.one_hot(indices=0,
                         depth=num_results,
                         on_value=1 + num_burnin_steps,
                         off_value=1 + num_steps_between_results,
                         dtype=tf.int64),  # num_steps
        initializer=[current_state, previous_kernel_results],
        parallel_iterations=parallel_iterations)
