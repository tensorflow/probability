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
"""Hamiltonian Monte Carlo, a gradient-based MCMC algorithm."""

import collections

import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import distribute_lib
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.mcmc import kernel as kernel_base
from tensorflow_probability.python.mcmc import metropolis_hastings
from tensorflow_probability.python.mcmc.internal import leapfrog_integrator as leapfrog_impl
from tensorflow_probability.python.mcmc.internal import util as mcmc_util
from tensorflow.python.util import deprecation  # pylint: disable=g-direct-tensorflow-import


__all__ = [
    'HamiltonianMonteCarlo',
    'UncalibratedHamiltonianMonteCarlo',
    'make_simple_step_size_update_policy',
]


class UncalibratedHamiltonianMonteCarloKernelResults(
    mcmc_util.PrettyNamedTupleMixin,
    collections.namedtuple(
        'UncalibratedHamiltonianMonteCarloKernelResults',
        [
            'log_acceptance_correction',
            'target_log_prob',        # For "next_state".
            'grads_target_log_prob',  # For "next_state".
            'initial_momentum',
            'final_momentum',
            'step_size',
            'num_leapfrog_steps',
            # Seed received by one_step, to reproduce divergent transitions etc.
            'seed',
        ])
    ):
  """Internal state and diagnostics for Uncalibrated HMC."""
  __slots__ = ()


class HamiltonianMonteCarloExtraKernelResults(
    mcmc_util.PrettyNamedTupleMixin,
    collections.namedtuple('HamiltonianMonteCarloExtraKernelResults',
                           ['step_size_assign',
                            ])):
  __slots__ = ()


@deprecation.deprecated('2019-05-22',
                        'Use tfp.mcmc.SimpleStepSizeAdaptation instead.')
def make_simple_step_size_update_policy(num_adaptation_steps,
                                        target_rate=0.75,
                                        decrement_multiplier=0.01,
                                        increment_multiplier=0.01,
                                        step_counter=None):
  """Create a function implementing a step-size update policy.

  The simple policy increases or decreases the `step_size_var` based on the
  average of `exp(minimum(0., log_accept_ratio))`. It is based on
  [Section 4.2 of Andrieu and Thoms (2008)](
  https://people.eecs.berkeley.edu/~jordan/sail/readings/andrieu-thoms.pdf).

  The `num_adaptation_steps` argument is set independently of any burnin
  for the overall chain. In general, adaptation prevents the chain from
  reaching a stationary distribution, so obtaining consistent samples requires
  `num_adaptation_steps` be set to a value [somewhat smaller](
  http://andrewgelman.com/2017/12/15/burn-vs-warm-iterative-simulation-algorithms/#comment-627745)
  than the number of burnin steps. However, it may sometimes be helpful to set
  `num_adaptation_steps` to a larger value during development in order to
  inspect the behavior of the chain during adaptation.

  Args:
    num_adaptation_steps: Scalar `int` `Tensor` number of initial steps to
      during which to adjust the step size. This may be greater, less than, or
      equal to the number of burnin steps. If `None`, the step size is adapted
      on every step (note this breaks stationarity of the chain!).
    target_rate: Scalar `Tensor` representing desired `accept_ratio`.
      Default value: `0.75` (i.e., [center of asymptotically optimal
      rate](https://arxiv.org/abs/1411.6669)).
    decrement_multiplier: `Tensor` representing amount to downscale current
      `step_size`.
      Default value: `0.01`.
    increment_multiplier: `Tensor` representing amount to upscale current
      `step_size`.
      Default value: `0.01`.
    step_counter: Scalar `int` `Variable` specifying the current step. The step
      size is adapted iff `step_counter < num_adaptation_steps`.
      Default value: if `None`, an internal variable
        `step_size_adaptation_step_counter` is created and initialized to `-1`.

  Returns:
    step_size_simple_update_fn: Callable that takes args
      `step_size_var, kernel_results` and returns updated step size(s).
  """
  if step_counter is None and num_adaptation_steps is not None:
    step_counter = tf1.get_variable(
        name='step_size_adaptation_step_counter',
        initializer=tf.constant(-1, dtype=tf.int32),
        # Specify the dtype for variable sharing to work correctly
        # (b/120599991).
        dtype=tf.int32,
        trainable=False,
        use_resource=True)

  def step_size_simple_update_fn(step_size_var, kernel_results):
    """Updates (list of) `step_size` using a standard adaptive MCMC procedure.

    Args:
      step_size_var: (List of) `tf.Variable`s representing the per `state_part`
        HMC `step_size`.
      kernel_results: `collections.namedtuple` containing `Tensor`s
        representing values from most recent call to `one_step`.

    Returns:
      step_size_assign: (List of) `Tensor`(s) representing updated
        `step_size_var`(s).
    """

    if kernel_results is None:
      if mcmc_util.is_list_like(step_size_var):
        return [tf.identity(ss) for ss in step_size_var]
      return tf.identity(step_size_var)
    log_n = tf.math.log(
        tf.cast(
            tf.size(kernel_results.log_accept_ratio),
            kernel_results.log_accept_ratio.dtype))
    log_mean_accept_ratio = tf.reduce_logsumexp(
        tf.minimum(kernel_results.log_accept_ratio, 0.)) - log_n
    adjustment = tf.where(
        log_mean_accept_ratio < tf.cast(
            tf.math.log(target_rate), log_mean_accept_ratio.dtype),
        -decrement_multiplier / (1. + decrement_multiplier),
        increment_multiplier)

    def build_assign_op():
      if mcmc_util.is_list_like(step_size_var):
        return [
            ss.assign_add(ss * tf.cast(adjustment, ss.dtype))
            for ss in step_size_var
        ]
      return step_size_var.assign_add(
          step_size_var * tf.cast(adjustment, step_size_var.dtype))

    if num_adaptation_steps is None:
      return build_assign_op()
    else:
      with tf.control_dependencies([step_counter.assign_add(1)]):
        return tf.cond(
            pred=step_counter < num_adaptation_steps,
            true_fn=build_assign_op,
            false_fn=lambda: step_size_var)

  return step_size_simple_update_fn


class HamiltonianMonteCarlo(kernel_base.TransitionKernel):
  """Runs one step of Hamiltonian Monte Carlo.

  Hamiltonian Monte Carlo (HMC) is a Markov chain Monte Carlo (MCMC) algorithm
  that takes a series of gradient-informed steps to produce a Metropolis
  proposal. This class implements one random HMC step from a given
  `current_state`. Mathematical details and derivations can be found in
  [Neal (2011)][1].

  The `one_step` function can update multiple chains in parallel. It assumes
  that all leftmost dimensions of `current_state` index independent chain states
  (and are therefore updated independently). The output of
  `target_log_prob_fn(*current_state)` should sum log-probabilities across all
  event dimensions. Slices along the rightmost dimensions may have different
  target distributions; for example, `current_state[0, :]` could have a
  different target distribution from `current_state[1, :]`. These semantics are
  governed by `target_log_prob_fn(*current_state)`. (The number of independent
  chains is `tf.size(target_log_prob_fn(*current_state))`.)

  #### Examples:

  ##### Simple chain with warm-up.

  In this example we sample from a standard univariate normal
  distribution using HMC with adaptive step size.

  ```python
  import tensorflow as tf
  import tensorflow_probability as tfp

  tf.enable_eager_execution()

  # Target distribution is proportional to: `exp(-x (1 + x))`.
  def unnormalized_log_prob(x):
    return -x - x**2.

  # Initialize the HMC transition kernel.
  num_results = int(10e3)
  num_burnin_steps = int(1e3)
  adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
      tfp.mcmc.HamiltonianMonteCarlo(
          target_log_prob_fn=unnormalized_log_prob,
          num_leapfrog_steps=3,
          step_size=1.),
      num_adaptation_steps=int(num_burnin_steps * 0.8))

  # Run the chain (with burn-in).
  @tf.function
  def run_chain():
    # Run the chain (with burn-in).
    samples, is_accepted = tfp.mcmc.sample_chain(
        num_results=num_results,
        num_burnin_steps=num_burnin_steps,
        current_state=1.,
        kernel=adaptive_hmc,
        trace_fn=lambda _, pkr: pkr.inner_results.is_accepted)

    sample_mean = tf.reduce_mean(samples)
    sample_stddev = tf.math.reduce_std(samples)
    is_accepted = tf.reduce_mean(tf.cast(is_accepted, dtype=tf.float32))
    return sample_mean, sample_stddev, is_accepted

  sample_mean, sample_stddev, is_accepted = run_chain()

  print('mean:{:.4f}  stddev:{:.4f}  acceptance:{:.4f}'.format(
      sample_mean.numpy(), sample_stddev.numpy(), is_accepted.numpy()))
  ```

  ##### Estimate parameters of a more complicated posterior.

  In this example, we'll use Monte-Carlo EM to find best-fit parameters. See
  [_Convergence of a stochastic approximation version of the EM algorithm_][2]
  for more details.

  More precisely, we use HMC to form a chain conditioned on parameter `sigma`
  and training data `{ (x[i], y[i]) : i=1...n }`. Then we use one gradient step
  of maximum-likelihood to improve the `sigma` estimate. Then repeat the process
  until convergence. (This procedure is a [Robbins--Monro algorithm](
  https://en.wikipedia.org/wiki/Stochastic_approximation).)

  The generative assumptions are:

  ```none
    W ~ MVN(loc=0, scale=sigma * eye(dims))
    for i=1...num_samples:
        X[i] ~ MVN(loc=0, scale=eye(dims))
      eps[i] ~ Normal(loc=0, scale=1)
        Y[i] = X[i].T * W + eps[i]
  ```

  We now implement a stochastic approximation of Expectation Maximization (SAEM)
  using `tensorflow_probability` intrinsics. [Bernard (1999)][2]

  ```python
  import tensorflow as tf
  import tensorflow_probability as tfp
  import numpy as np

  tf.enable_eager_execution()

  tfd = tfp.distributions

  def make_training_data(num_samples, dims, sigma):
    dt = np.asarray(sigma).dtype
    x = np.random.randn(dims, num_samples).astype(dt)
    w = sigma * np.random.randn(1, dims).astype(dt)
    noise = np.random.randn(num_samples).astype(dt)
    y = w.dot(x) + noise
    return y[0], x, w[0]

  def make_weights_prior(dims, log_sigma):
    return tfd.MultivariateNormalDiag(
        loc=tf.zeros([dims], dtype=log_sigma.dtype),
        scale_diag=tf.math.exp(log_sigma) *
                   tf.ones([dims], dtype=log_sigma.dtype))

  def make_response_likelihood(w, x):
    if w.shape.ndims == 1:
      y_bar = tf.matmul(w[tf.newaxis], x)[0]
    else:
      y_bar = tf.matmul(w, x)
    return tfd.Normal(loc=y_bar, scale=tf.ones_like(y_bar))  # [n]

  # Setup assumptions.
  dtype = np.float32
  num_samples = 500
  dims = 10
  tf.random.set_seed(10014)
  np.random.seed(10014)

  weights_prior_true_scale = np.array(0.3, dtype)
  y, x, _ = make_training_data(
      num_samples, dims, weights_prior_true_scale)

  log_sigma = tf.Variable(0., dtype=dtype, name='log_sigma')

  optimizer = tf.optimizers.SGD(learning_rate=0.01)

  @tf.function
  def mcem_iter(weights_chain_start, step_size):
    with tf.GradientTape() as tape:
      tape.watch(log_sigma)
      prior = make_weights_prior(dims, log_sigma)

      def unnormalized_posterior_log_prob(w):
        likelihood = make_response_likelihood(w, x)
        return (
            prior.log_prob(w) +
            tf.reduce_sum(likelihood.log_prob(y), axis=-1))  # [m]

      def trace_fn(_, pkr):
        return (
            pkr.inner_results.log_accept_ratio,
            pkr.inner_results.accepted_results.target_log_prob,
            pkr.inner_results.accepted_results.step_size)

      num_results = 2
      weights, (
          log_accept_ratio, target_log_prob, step_size) = tfp.mcmc.sample_chain(
          num_results=num_results,
          num_burnin_steps=0,
          current_state=weights_chain_start,
          kernel=tfp.mcmc.SimpleStepSizeAdaptation(
              tfp.mcmc.HamiltonianMonteCarlo(
                  target_log_prob_fn=unnormalized_posterior_log_prob,
                  num_leapfrog_steps=2,
                  step_size=step_size,
                  state_gradients_are_stopped=True,
              ),
              # Adapt for the entirety of the trajectory.
              num_adaptation_steps=2),
          trace_fn=trace_fn,
          seed=123)

      # We do an optimization step to propagate `log_sigma` after two HMC
      # steps to propagate `weights`.
      loss = -tf.reduce_mean(target_log_prob)

    avg_acceptance_ratio = tf.math.exp(
        tfp.math.reduce_logmeanexp(tf.minimum(log_accept_ratio, 0.)))

    optimizer.apply_gradients(
        [[tape.gradient(loss, log_sigma), log_sigma]])

    weights_prior_estimated_scale = tf.math.exp(log_sigma)
    return (weights_prior_estimated_scale, weights[-1], loss,
            step_size[-1], avg_acceptance_ratio)

  num_iters = int(40)

  weights_prior_estimated_scale_ = np.zeros(num_iters, dtype)
  weights_ = np.zeros([num_iters + 1, dims], dtype)
  loss_ = np.zeros([num_iters], dtype)
  weights_[0] = np.random.randn(dims).astype(dtype)
  step_size_ = 0.03

  for iter_ in range(num_iters):
    [
        weights_prior_estimated_scale_[iter_],
        weights_[iter_ + 1],
        loss_[iter_],
        step_size_,
        avg_acceptance_ratio_,
    ] = mcem_iter(weights_[iter_], step_size_)
    tf.compat.v1.logging.vlog(
        1, ('iter:{:>2}  loss:{: 9.3f}  scale:{:.3f}  '
            'step_size:{:.4f}  avg_acceptance_ratio:{:.4f}').format(
                iter_, loss_[iter_], weights_prior_estimated_scale_[iter_],
                step_size_, avg_acceptance_ratio_))

  # Should converge to ~0.22.
  import matplotlib.pyplot as plt
  plt.plot(weights_prior_estimated_scale_)
  plt.ylabel('weights_prior_estimated_scale')
  plt.xlabel('iteration')
  ```

  #### References

  [1]: Radford Neal. MCMC Using Hamiltonian Dynamics. _Handbook of Markov Chain
       Monte Carlo_, 2011. https://arxiv.org/abs/1206.1901

  [2]: Bernard Delyon, Marc Lavielle, Eric, Moulines. _Convergence of a
       stochastic approximation version of the EM algorithm_, Ann. Statist. 27
       (1999), no. 1, 94--128. https://projecteuclid.org/euclid.aos/1018031103
  """

  def __init__(self,
               target_log_prob_fn,
               step_size,
               num_leapfrog_steps,
               state_gradients_are_stopped=False,
               store_parameters_in_results=False,
               experimental_shard_axis_names=None,
               name=None):
    """Initializes this transition kernel.

    Args:
      target_log_prob_fn: Python callable which takes an argument like
        `current_state` (or `*current_state` if it's a list) and returns its
        (possibly unnormalized) log-density under the target distribution.
      step_size: `Tensor` or Python `list` of `Tensor`s representing the step
        size for the leapfrog integrator. Must broadcast with the shape of
        `current_state`. Larger step sizes lead to faster progress, but
        too-large step sizes make rejection exponentially more likely. When
        possible, it's often helpful to match per-variable step sizes to the
        standard deviations of the target distribution in each variable.
      num_leapfrog_steps: Integer number of steps to run the leapfrog integrator
        for. Total progress per HMC step is roughly proportional to
        `step_size * num_leapfrog_steps`.
      state_gradients_are_stopped: Python `bool` indicating that the proposed
        new state be run through `tf.stop_gradient`. This is particularly useful
        when combining optimization over samples from the HMC chain.
        Default value: `False` (i.e., do not apply `stop_gradient`).
      store_parameters_in_results: If `True`, then `step_size` and
        `num_leapfrog_steps` are written to and read from eponymous fields in
        the kernel results objects returned from `one_step` and
        `bootstrap_results`. This allows wrapper kernels to adjust those
        parameters on the fly.
      experimental_shard_axis_names: A structure of string names indicating how
        members of the state are sharded.
      name: Python `str` name prefixed to Ops created by this function.
        Default value: `None` (i.e., 'hmc_kernel').
    """
    self._impl = metropolis_hastings.MetropolisHastings(
        inner_kernel=UncalibratedHamiltonianMonteCarlo(
            target_log_prob_fn=target_log_prob_fn,
            step_size=step_size,
            num_leapfrog_steps=num_leapfrog_steps,
            state_gradients_are_stopped=state_gradients_are_stopped,
            name=name or 'hmc_kernel',
            store_parameters_in_results=store_parameters_in_results,
        )).experimental_with_shard_axes(experimental_shard_axis_names)
    self._parameters = self._impl.inner_kernel.parameters.copy()

  @property
  def target_log_prob_fn(self):
    return self._impl.inner_kernel.target_log_prob_fn

  @property
  def step_size(self):
    """Returns the step_size parameter.

    If `store_parameters_in_results` argument to the initializer was set to
    `True`, this only returns the value of the `step_size` placed in the kernel
    results by the `bootstrap_results` method. The actual step size in that
    situation is governed by the `previous_kernel_results` argument to
    `one_step` method.

    Returns:
      step_size: A floating point `Tensor` or a list of such `Tensors`.
    """
    return self._impl.inner_kernel.step_size

  @property
  def num_leapfrog_steps(self):
    """Returns the num_leapfrog_steps parameter.

    If `store_parameters_in_results` argument to the initializer was set to
    `True`, this only returns the value of the `num_leapfrog_steps` placed in
    the kernel results by the `bootstrap_results` method. The actual
    `num_leapfrog_steps` in that situation is governed by the
    `previous_kernel_results` argument to `one_step` method.

    Returns:
      num_leapfrog_steps: An integer `Tensor`.
    """
    return self._impl.inner_kernel.num_leapfrog_steps

  @property
  def state_gradients_are_stopped(self):
    return self._impl.inner_kernel.state_gradients_are_stopped

  @property
  def name(self):
    return self._impl.inner_kernel.name

  @property
  def parameters(self):
    """Return `dict` of ``__init__`` arguments and their values."""
    return self._parameters

  @property
  def is_calibrated(self):
    return True

  def one_step(self, current_state, previous_kernel_results, seed=None):
    """Runs one iteration of Hamiltonian Monte Carlo.

    Args:
      current_state: `Tensor` or Python `list` of `Tensor`s representing the
        current state(s) of the Markov chain(s). The first `r` dimensions index
        independent chains, `r = tf.rank(target_log_prob_fn(*current_state))`.
      previous_kernel_results: `collections.namedtuple` containing `Tensor`s
        representing values from previous calls to this function (or from the
        `bootstrap_results` function.)
      seed: PRNG seed; see `tfp.random.sanitize_seed` for details.

    Returns:
      next_state: Tensor or Python list of `Tensor`s representing the state(s)
        of the Markov chain(s) after taking exactly one step. Has same type and
        shape as `current_state`.
      kernel_results: `collections.namedtuple` of internal calculations used to
        advance the chain.

    Raises:
      ValueError: if there isn't one `step_size` or a list with same length as
        `current_state`.
    """
    previous_step_size_assign = []

    with tf.control_dependencies(previous_step_size_assign):
      next_state, kernel_results = self._impl.one_step(
          current_state, previous_kernel_results, seed=seed)
      return next_state, kernel_results

  def bootstrap_results(self, init_state):
    """Creates initial `previous_kernel_results` using a supplied `state`."""
    return self._impl.bootstrap_results(init_state)

  @property
  def experimental_shard_axis_names(self):
    return self._parameters['experimental_shard_axis_names']

  def experimental_with_shard_axes(self, shard_axis_names):
    return self.copy(experimental_shard_axis_names=shard_axis_names)


class UncalibratedHamiltonianMonteCarlo(kernel_base.TransitionKernel):
  """Runs one step of Uncalibrated Hamiltonian Monte Carlo.

  Warning: this kernel will not result in a chain which converges to the
  `target_log_prob`. To get a convergent MCMC, use `HamiltonianMonteCarlo(...)`
  or `MetropolisHastings(UncalibratedHamiltonianMonteCarlo(...))`.

  For more details on `UncalibratedHamiltonianMonteCarlo`, see
  `HamiltonianMonteCarlo`.
  """

  def __init__(self,
               target_log_prob_fn,
               step_size,
               num_leapfrog_steps,
               state_gradients_are_stopped=False,
               store_parameters_in_results=False,
               experimental_shard_axis_names=None,
               name=None):
    """Initializes this transition kernel.

    Args:
      target_log_prob_fn: Python callable which takes an argument like
        `current_state` (or `*current_state` if it's a list) and returns its
        (possibly unnormalized) log-density under the target distribution.
      step_size: `Tensor` or Python `list` of `Tensor`s representing the step
        size for the leapfrog integrator. Must broadcast with the shape of
        `current_state`. Larger step sizes lead to faster progress, but
        too-large step sizes make rejection exponentially more likely. When
        possible, it's often helpful to match per-variable step sizes to the
        standard deviations of the target distribution in each variable.
      num_leapfrog_steps: Integer number of steps to run the leapfrog integrator
        for. Total progress per HMC step is roughly proportional to
        `step_size * num_leapfrog_steps`.
      state_gradients_are_stopped: Python `bool` indicating that the proposed
        new state be run through `tf.stop_gradient`. This is particularly useful
        when combining optimization over samples from the HMC chain.
        Default value: `False` (i.e., do not apply `stop_gradient`).
      store_parameters_in_results: If `True`, then `step_size` and
        `num_leapfrog_steps` are written to and read from eponymous fields in
        the kernel results objects returned from `one_step` and
        `bootstrap_results`. This allows wrapper kernels to adjust those
        parameters on the fly.
      experimental_shard_axis_names: A structure of string names indicating how
        members of the state are sharded.
      name: Python `str` name prefixed to Ops created by this function.
        Default value: `None` (i.e., 'hmc_kernel').
    """
    if not store_parameters_in_results:
      mcmc_util.warn_if_parameters_are_not_simple_tensors(
          dict(step_size=step_size, num_leapfrog_steps=num_leapfrog_steps))
    self._parameters = dict(
        target_log_prob_fn=target_log_prob_fn,
        step_size=step_size,
        num_leapfrog_steps=num_leapfrog_steps,
        state_gradients_are_stopped=state_gradients_are_stopped,
        name=name,
        experimental_shard_axis_names=experimental_shard_axis_names,
        store_parameters_in_results=store_parameters_in_results,
    )
    self._momentum_dtype = None

  @property
  def target_log_prob_fn(self):
    return self._parameters['target_log_prob_fn']

  @property
  def step_size(self):
    """Returns the step_size parameter.

    If `store_parameters_in_results` argument to the initializer was set to
    `True`, this only returns the value of the `step_size` placed in the kernel
    results by the `bootstrap_results` method. The actual step size in that
    situation is governed by the `previous_kernel_results` argument to
    `one_step` method.

    Returns:
      step_size: A floating point `Tensor` or a list of such `Tensors`.
    """
    return self._parameters['step_size']

  @property
  def num_leapfrog_steps(self):
    """Returns the num_leapfrog_steps parameter.

    If `store_parameters_in_results` argument to the initializer was set to
    `True`, this only returns the value of the `num_leapfrog_steps` placed in
    the kernel results by the `bootstrap_results` method. The actual
    `num_leapfrog_steps` in that situation is governed by the
    `previous_kernel_results` argument to `one_step` method.

    Returns:
      num_leapfrog_steps: An integer `Tensor`.
    """
    return self._parameters['num_leapfrog_steps']

  @property
  def state_gradients_are_stopped(self):
    return self._parameters['state_gradients_are_stopped']

  @property
  def name(self):
    return self._parameters['name']

  @property
  def parameters(self):
    """Return `dict` of ``__init__`` arguments and their values."""
    return self._parameters

  @property
  def is_calibrated(self):
    return False

  @property
  def _store_parameters_in_results(self):
    return self._parameters['store_parameters_in_results']

  @mcmc_util.set_doc(HamiltonianMonteCarlo.one_step.__doc__)
  def one_step(self, current_state, previous_kernel_results, seed=None):
    with tf.name_scope(mcmc_util.make_name(self.name, 'hmc', 'one_step')):
      if self._store_parameters_in_results:
        step_size = previous_kernel_results.step_size
        num_leapfrog_steps = previous_kernel_results.num_leapfrog_steps
      else:
        step_size = self.step_size
        num_leapfrog_steps = self.num_leapfrog_steps

      [
          current_state_parts,
          step_sizes,
          current_target_log_prob,
          current_target_log_prob_grad_parts,
      ] = _prepare_args(
          self.target_log_prob_fn,
          current_state,
          step_size,
          previous_kernel_results.target_log_prob,
          previous_kernel_results.grads_target_log_prob,
          maybe_expand=True,
          state_gradients_are_stopped=self.state_gradients_are_stopped)

      seed = samplers.sanitize_seed(seed)  # Retain for diagnostics.
      seeds = list(samplers.split_seed(seed, n=len(current_state_parts)))
      seeds = distribute_lib.fold_in_axis_index(
          seeds, self.experimental_shard_axis_names)

      current_momentum_parts = []
      for part_seed, x in zip(seeds, current_state_parts):
        current_momentum_parts.append(
            samplers.normal(
                shape=ps.shape(x),
                dtype=self._momentum_dtype or dtype_util.base_dtype(x.dtype),
                seed=part_seed))

      integrator = leapfrog_impl.SimpleLeapfrogIntegrator(
          self.target_log_prob_fn, step_sizes, num_leapfrog_steps)

      [
          next_momentum_parts,
          next_state_parts,
          next_target_log_prob,
          next_target_log_prob_grad_parts,
      ] = integrator(current_momentum_parts,
                     current_state_parts,
                     current_target_log_prob,
                     current_target_log_prob_grad_parts)
      if self.state_gradients_are_stopped:
        next_state_parts = [tf.stop_gradient(x) for x in next_state_parts]

      def maybe_flatten(x):
        return x if mcmc_util.is_list_like(current_state) else x[0]

      independent_chain_ndims = ps.rank(current_target_log_prob)

      new_kernel_results = previous_kernel_results._replace(
          log_acceptance_correction=_compute_log_acceptance_correction(
              current_momentum_parts, next_momentum_parts,
              independent_chain_ndims,
              shard_axis_names=self.experimental_shard_axis_names),
          target_log_prob=next_target_log_prob,
          grads_target_log_prob=next_target_log_prob_grad_parts,
          initial_momentum=current_momentum_parts,
          final_momentum=next_momentum_parts,
          seed=seed,
      )

      return maybe_flatten(next_state_parts), new_kernel_results

  @mcmc_util.set_doc(HamiltonianMonteCarlo.bootstrap_results.__doc__)
  def bootstrap_results(self, init_state):
    with tf.name_scope(
        mcmc_util.make_name(self.name, 'hmc', 'bootstrap_results')):
      init_state, _ = mcmc_util.prepare_state_parts(init_state)
      if self.state_gradients_are_stopped:
        init_state = [tf.stop_gradient(x) for x in init_state]
      [
          init_target_log_prob,
          init_grads_target_log_prob,
      ] = mcmc_util.maybe_call_fn_and_grads(self.target_log_prob_fn, init_state)
      result = UncalibratedHamiltonianMonteCarloKernelResults(
          log_acceptance_correction=tf.zeros_like(init_target_log_prob),
          target_log_prob=init_target_log_prob,
          grads_target_log_prob=init_grads_target_log_prob,
          initial_momentum=tf.nest.map_structure(tf.zeros_like, init_state),
          final_momentum=tf.nest.map_structure(tf.zeros_like, init_state),
          step_size=[],
          num_leapfrog_steps=[],
          # Allow room for one_step's seed.
          seed=samplers.zeros_seed())
      if self._store_parameters_in_results:
        result = result._replace(
            # TODO(b/142590314): Try to use the following code once we commit to
            # a tensorization policy.
            # step_size=mcmc_util.prepare_state_parts(
            #    self.step_size,
            #    dtype=init_target_log_prob.dtype,
            #    name='step_size')[0],
            step_size=tf.nest.map_structure(
                lambda x: tf.convert_to_tensor(  # pylint: disable=g-long-lambda
                    x,
                    dtype=init_target_log_prob.dtype,
                    name='step_size'),
                self.step_size),
            num_leapfrog_steps=tf.convert_to_tensor(
                self.num_leapfrog_steps,
                dtype=tf.int32,
                name='num_leapfrog_steps'))
      return result

  @property
  def experimental_shard_axis_names(self):
    return self._parameters['experimental_shard_axis_names']

  def experimental_with_shard_axes(self, shard_axis_names):
    return self.copy(experimental_shard_axis_names=shard_axis_names)


def _compute_log_acceptance_correction(current_momentums,
                                       proposed_momentums,
                                       independent_chain_ndims,
                                       shard_axis_names=None,
                                       name=None):
  """Helper to `kernel` which computes the log acceptance-correction.

  A sufficient but not necessary condition for the existence of a stationary
  distribution, `p(x)`, is "detailed balance", i.e.:

  ```none
  p(x'|x) p(x) = p(x|x') p(x')
  ```

  In the Metropolis-Hastings algorithm, a state is proposed according to
  `g(x'|x)` and accepted according to `a(x'|x)`, hence
  `p(x'|x) = g(x'|x) a(x'|x)`.

  Inserting this into the detailed balance equation implies:

  ```none
      g(x'|x) a(x'|x) p(x) = g(x|x') a(x|x') p(x')
  ==> a(x'|x) / a(x|x') = p(x') / p(x) [g(x|x') / g(x'|x)]    (*)
  ```

  One definition of `a(x'|x)` which satisfies (*) is:

  ```none
  a(x'|x) = min(1, p(x') / p(x) [g(x|x') / g(x'|x)])
  ```

  (To see that this satisfies (*), notice that under this definition only at
  most one `a(x'|x)` and `a(x|x') can be other than one.)

  We call the bracketed term the "acceptance correction".

  In the case of UncalibratedHMC, the log acceptance-correction is not the log
  proposal-ratio. UncalibratedHMC augments the state-space with momentum, z.
  Assuming a standard Gaussian distribution for momentums, the chain eventually
  converges to:

  ```none
  p([x, z]) propto= target_prob(x) exp(-0.5 z**2)
  ```

  Relating this back to Metropolis-Hastings parlance, for HMC we have:

  ```none
  p([x, z]) propto= target_prob(x) exp(-0.5 z**2)
  g([x, z] | [x', z']) = g([x', z'] | [x, z])
  ```

  In other words, the MH bracketed term is `1`. However, because we desire to
  use a general MH framework, we can place the momentum probability ratio inside
  the metropolis-correction factor thus getting an acceptance probability:

  ```none
                       target_prob(x')
  accept_prob(x'|x) = -----------------  [exp(-0.5 z**2) / exp(-0.5 z'**2)]
                       target_prob(x)
  ```

  (Note: we actually need to handle the kinetic energy change at each leapfrog
  step, but this is the idea.)

  Args:
    current_momentums: `Tensor` representing the value(s) of the current
      momentum(s) of the state (parts).
    proposed_momentums: `Tensor` representing the value(s) of the proposed
      momentum(s) of the state (parts).
    independent_chain_ndims: Scalar `int` `Tensor` representing the number of
      leftmost `Tensor` dimensions which index independent chains.
    shard_axis_names: A structure of string names indicating how
      members of the state are sharded.
    name: Python `str` name prefixed to Ops created by this function.
      Default value: `None` (i.e., 'compute_log_acceptance_correction').

  Returns:
    log_acceptance_correction: `Tensor` representing the `log`
      acceptance-correction.  (See docstring for mathematical definition.)
  """
  with tf.name_scope(name or 'compute_log_acceptance_correction'):
    def compute_sum_sq(v, shard_axes):
      sum_sq = tf.reduce_sum(v ** 2., axis=ps.range(
          independent_chain_ndims, ps.rank(v)))
      if shard_axes is not None:
        sum_sq = distribute_lib.psum(sum_sq, shard_axes)
      return sum_sq
    shard_axis_names = (shard_axis_names or ([None] * len(current_momentums)))
    current_kinetic = tf.add_n([
        compute_sum_sq(v, axes) for v, axes
        in zip(current_momentums, shard_axis_names)])
    proposed_kinetic = tf.add_n([
        compute_sum_sq(v, axes) for v, axes
        in zip(proposed_momentums, shard_axis_names)])
    return 0.5 * mcmc_util.safe_sum([current_kinetic, -proposed_kinetic])


def _prepare_args(target_log_prob_fn,
                  state,
                  step_size,
                  target_log_prob=None,
                  grads_target_log_prob=None,
                  maybe_expand=False,
                  state_gradients_are_stopped=False):
  """Helper which processes input args to meet list-like assumptions."""
  state_parts, _ = mcmc_util.prepare_state_parts(state, name='current_state')
  if state_gradients_are_stopped:
    state_parts = [tf.stop_gradient(x) for x in state_parts]
  target_log_prob, grads_target_log_prob = mcmc_util.maybe_call_fn_and_grads(
      target_log_prob_fn, state_parts, target_log_prob, grads_target_log_prob)
  step_sizes, _ = mcmc_util.prepare_state_parts(
      step_size, dtype=target_log_prob.dtype, name='step_size')
  if len(step_sizes) == 1:
    step_sizes *= len(state_parts)
  if len(state_parts) != len(step_sizes):
    raise ValueError('There should be exactly one `step_size` or it should '
                     'have same length as `current_state`.')
  def maybe_flatten(x):
    return x if maybe_expand or mcmc_util.is_list_like(state) else x[0]
  return [
      maybe_flatten(state_parts),
      maybe_flatten(step_sizes),
      target_log_prob,
      grads_target_log_prob,
  ]
