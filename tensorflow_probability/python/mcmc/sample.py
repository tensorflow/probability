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

import collections
import warnings
# Dependency imports

import tensorflow as tf
from tensorflow_probability.python.mcmc import util as mcmc_util


__all__ = [
    "CheckpointableStatesAndTrace",
    "StatesAndTrace",
    "sample_chain",
]

# Cause all warnings to always be triggered.
# Not having this means subsequent calls wont trigger the warning.
warnings.filterwarnings("always",
                        module="tensorflow_probability.*sample",
                        append=True)  # Don't override user-set filters.


class StatesAndTrace(
    collections.namedtuple("StatesAndTrace", "all_states, trace")):
  """States and auxiliary trace of an MCMC chain.

  The first dimension of all the `Tensor`s in this structure is the same and
  represents the chain length.

  Attributes:
    all_states: A `Tensor` or a nested collection of `Tensor`s representing the
      MCMC chain state.
    trace: A `Tensor` or a nested collection of `Tensor`s representing the
      auxiliary values traced alongside the chain.
  """
  __slots__ = ()


class CheckpointableStatesAndTrace(
    collections.namedtuple("CheckpointableStatesAndTrace",
                           "all_states, trace, final_kernel_results")):
  """States and auxiliary trace of an MCMC chain.

  The first dimension of all the `Tensor`s in the `all_states` and `trace`
  attributes is the same and represents the chain length.

  Attributes:
    all_states: A `Tensor` or a nested collection of `Tensor`s representing the
      MCMC chain state.
    trace: A `Tensor` or a nested collection of `Tensor`s representing the
      auxiliary values traced alongside the chain.
    final_kernel_results: A `Tensor` or a nested collection of `Tensor`s
      representing the final value of the auxiliary state of the
      `TransitionKernel` that generated this chain.
  """
  __slots__ = ()


def sample_chain(
    num_results,
    current_state,
    previous_kernel_results=None,
    kernel=None,
    num_burnin_steps=0,
    num_steps_between_results=0,
    trace_fn=lambda current_state, kernel_results: kernel_results,
    return_final_kernel_results=False,
    parallel_iterations=10,
    name=None,
):
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

  In addition to returning the chain state, this function supports tracing of
  auxiliary variables used by the kernel. The traced values are selected by
  specifying `trace_fn`. By default, all kernel results are traced but in the
  future the default will be changed to no results being traced, so plan
  accordingly. See below for some examples of this feature.

  Args:
    num_results: Integer number of Markov chain draws.
    current_state: `Tensor` or Python `list` of `Tensor`s representing the
      current state(s) of the Markov chain(s).
    previous_kernel_results: A `Tensor` or a nested collection of `Tensor`s
      representing internal calculations made within the previous call to this
      function (or as returned by `bootstrap_results`).
    kernel: An instance of `tfp.mcmc.TransitionKernel` which implements one step
      of the Markov chain.
    num_burnin_steps: Integer number of chain steps to take before starting to
      collect results.
      Default value: 0 (i.e., no burn-in).
    num_steps_between_results: Integer number of chain steps between collecting
      a result. Only one out of every `num_steps_between_samples + 1` steps is
      included in the returned results.  The number of returned chain states is
      still equal to `num_results`.  Default value: 0 (i.e., no thinning).
    trace_fn: A callable that takes in the current chain state and the previous
      kernel results and return a `Tensor` or a nested collection of `Tensor`s
      that is then traced along with the chain state.
    return_final_kernel_results: If `True`, then the final kernel results are
      returned alongside the chain state and the trace specified by the
      `trace_fn`.
    parallel_iterations: The number of iterations allowed to run in parallel. It
      must be a positive integer. See `tf.while_loop` for more details.
    name: Python `str` name prefixed to Ops created by this function.
      Default value: `None` (i.e., "mcmc_sample_chain").

  Returns:
    checkpointable_states_and_trace: if `return_final_kernel_results` is
      `True`. The return value is an instance of
      `CheckpointableStatesAndTrace`.
    all_states: if `return_final_kernel_results` is `False` and `trace_fn` is
      `None`. The return value is a `Tensor` or Python list of `Tensor`s
      representing the state(s) of the Markov chain(s) at each result step. Has
      same shape as input `current_state` but with a prepended
      `num_results`-size dimension.
    states_and_trace: if `return_final_kernel_results` is `False` and
      `trace_fn` is not `None`. The return value is an instance of
      `StatesAndTrace`.

  #### Examples

  ##### Sample from a diagonal-variance Gaussian.

  I.e.,

  ```none
  for i=1..n:
    x[i] ~ MultivariateNormal(loc=0, scale=diag(true_stddev))  # likelihood
  ```

  ```python
  import tensorflow as tf
  import tensorflow_probability as tfp
  tfd = tfp.distributions

  dims = 10
  true_stddev = np.sqrt(np.linspace(1., 3., dims))
  likelihood = tfd.MultivariateNormalDiag(loc=0., scale_diag=true_stddev)

  states = tfp.mcmc.sample_chain(
      num_results=1000,
      num_burnin_steps=500,
      current_state=tf.zeros(dims),
      kernel=tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=likelihood.log_prob,
        step_size=0.5,
        num_leapfrog_steps=2),
      trace_fn=None)

  sample_mean = tf.reduce_mean(states, axis=0)
  # ==> approx all zeros

  sample_stddev = tf.sqrt(tf.reduce_mean(
      tf.squared_difference(states, sample_mean),
      axis=0))
  # ==> approx equal true_stddev
  ```

  ##### Sampling from factor-analysis posteriors with known factors.

  I.e.,

  ```none
  # prior
  w ~ MultivariateNormal(loc=0, scale=eye(d))
  for i=1..n:
    # likelihood
    x[i] ~ Normal(loc=w^T F[i], scale=1)
  ```

  where `F` denotes factors.

  ```python
  import tensorflow as tf
  import tensorflow_probability as tfp
  tfd = tfp.distributions

  # Specify model.
  def make_prior(dims):
    return tfd.MultivariateNormalDiag(
        loc=tf.zeros(dims))

  def make_likelihood(weights, factors):
    return tfd.MultivariateNormalDiag(
        loc=tf.matmul(weights, factors, adjoint_b=True))

  def joint_log_prob(num_weights, factors, x, w):
    return (make_prior(num_weights).log_prob(w) +
            make_likelihood(w, factors).log_prob(x))

  def unnormalized_log_posterior(w):
    # Posterior is proportional to: `p(W, X=x | factors)`.
    return joint_log_prob(num_weights, factors, x, w)

  # Setup data.
  num_weights = 10 # == d
  num_factors = 40 # == n
  num_chains = 100

  weights = make_prior(num_weights).sample(1)
  factors = tf.random_normal([num_factors, num_weights])
  x = make_likelihood(weights, factors).sample()

  # Sample from Hamiltonian Monte Carlo Markov Chain.

  # Get `num_results` samples from `num_chains` independent chains.
  chains_states, kernels_results = tfp.mcmc.sample_chain(
      num_results=1000,
      num_burnin_steps=500,
      current_state=tf.zeros([num_chains, num_weights], name='init_weights'),
      kernel=tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=unnormalized_log_posterior,
        step_size=0.1,
        num_leapfrog_steps=2))

  # Compute sample stats.
  sample_mean = tf.reduce_mean(chains_states, axis=[0, 1])
  # ==> approx equal to weights

  sample_var = tf.reduce_mean(
      tf.squared_difference(chains_states, sample_mean),
      axis=[0, 1])
  # ==> less than 1
  ```

  ##### Custom tracing functions.

  ```python
  import tensorflow as tf
  import tensorflow_probability as tfp
  tfd = tfp.distributions

  likelihood = tfd.Normal(loc=0., scale=1.)

  def sample_chain(trace_fn):
    return tfp.mcmc.sample_chain(
      num_results=1000,
      num_burnin_steps=500,
      current_state=0.,
      kernel=tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=likelihood.log_prob,
        step_size=0.5,
        num_leapfrog_steps=2),
      trace_fn=trace_fn)

  def trace_log_accept_ratio(states, previous_kernel_results):
    return previous_kernel_results.log_accept_ratio

  def trace_everything(states, previous_kernel_results):
    return previous_kernel_results

  _, log_accept_ratio = sample_chain(trace_fn=trace_log_accept_ratio)
  _, kernel_results = sample_chain(trace_fn=trace_everything)

  acceptance_prob = tf.exp(tf.minimum(log_accept_ratio_, 0.))
  # Equivalent to, but more efficient than:
  acceptance_prob = tf.exp(tf.minimum(kernel_results.log_accept_ratio_, 0.))
  ```

  #### References

  [1]: Art B. Owen. Statistically efficient thinning of a Markov chain sampler.
       _Technical Report_, 2017.
       http://statweb.stanford.edu/~owen/reports/bestthinning.pdf
  """
  if not kernel.is_calibrated:
    warnings.warn("supplied `TransitionKernel` is not calibrated. Markov "
                  "chain may not converge to intended target distribution.")
  with tf.compat.v1.name_scope(
      name, "mcmc_sample_chain",
      [num_results, num_burnin_steps, num_steps_between_results]):
    num_results = tf.convert_to_tensor(
        value=num_results, dtype=tf.int32, name="num_results")
    num_burnin_steps = tf.convert_to_tensor(
        value=num_burnin_steps, dtype=tf.int64, name="num_burnin_steps")
    num_steps_between_results = tf.convert_to_tensor(
        value=num_steps_between_results,
        dtype=tf.int64,
        name="num_steps_between_results")
    current_state = tf.nest.map_structure(
        lambda x: tf.convert_to_tensor(value=x, name="current_state"),
        current_state)
    if previous_kernel_results is None:
      previous_kernel_results = kernel.bootstrap_results(current_state)

    if trace_fn is None:
      # It simplifies the logic to use a dummy function here.
      trace_fn = lambda *args: ()
      no_trace = True
    else:
      no_trace = False
    if trace_fn is sample_chain.__defaults__[4]:
      warnings.warn("Tracing all kernel results by default is deprecated. Set "
                    "the `trace_fn` argument to None (the future default "
                    "value) or an explicit callback that traces the values "
                    "you are interested in.")

    def _trace_scan_fn(state_and_results, num_steps):
      next_state, current_kernel_results = mcmc_util.smart_for_loop(
          loop_num_iter=num_steps,
          body_fn=kernel.one_step,
          initial_loop_vars=list(state_and_results),
          parallel_iterations=parallel_iterations)
      return next_state, current_kernel_results

    (_, final_kernel_results), (all_states, trace) = mcmc_util.trace_scan(
        loop_fn=_trace_scan_fn,
        initial_state=(current_state, previous_kernel_results),
        elems=tf.one_hot(
            indices=0,
            depth=num_results,
            on_value=1 + num_burnin_steps,
            off_value=1 + num_steps_between_results,
            dtype=tf.int64),
        # pylint: disable=g-long-lambda
        trace_fn=lambda state_and_results: (state_and_results[0],
                                            trace_fn(*state_and_results)),
        # pylint: enable=g-long-lambda
        parallel_iterations=parallel_iterations)

    if return_final_kernel_results:
      return CheckpointableStatesAndTrace(
          all_states=all_states,
          trace=trace,
          final_kernel_results=final_kernel_results)
    else:
      if no_trace:
        return all_states
      else:
        return StatesAndTrace(all_states=all_states, trace=trace)
