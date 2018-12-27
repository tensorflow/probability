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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
# Dependency imports

import numpy as np
import tensorflow as tf

from tensorflow_probability.python import distributions
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.mcmc import kernel as kernel_base
from tensorflow_probability.python.mcmc import metropolis_hastings
from tensorflow_probability.python.mcmc import util as mcmc_util


__all__ = [
    'HamiltonianMonteCarlo',
    'UncalibratedHamiltonianMonteCarlo',
    'make_simple_step_size_update_policy',
]


UncalibratedHamiltonianMonteCarloKernelResults = collections.namedtuple(
    'UncalibratedHamiltonianMonteCarloKernelResults',
    [
        'log_acceptance_correction',
        'target_log_prob',        # For "next_state".
        'grads_target_log_prob',  # For "next_state".
    ])

HamiltonianMonteCarloExtraKernelResults = collections.namedtuple(
    'HamiltonianMonteCarloExtraKernelResults',
    [
        'step_size_assign',
    ])


def make_simple_step_size_update_policy(num_adaptation_steps,
                                        target_rate=0.75,
                                        decrement_multiplier=0.01,
                                        increment_multiplier=0.01,
                                        step_counter=None):
  """Create a function implementing a step-size update policy.

  The simple policy increases or decreases the `step_size_var` based on the
  average of `exp(minimum(0., log_accept_ratio))`. It is based on
  [Section 4.2 of Andrieu and Thoms (2008)](
  http://www4.ncsu.edu/~rsmith/MA797V_S12/Andrieu08_AdaptiveMCMC_Tutorial.pdf).

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
    step_counter = tf.get_variable(
        name='step_size_adaptation_step_counter',
        initializer=np.array(-1, dtype=np.int64),
        # Specify the dtype for variable sharing to work correctly
        # (b/120599991).
        dtype=tf.int64,
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
    log_n = tf.log(tf.cast(tf.size(kernel_results.log_accept_ratio),
                           kernel_results.log_accept_ratio.dtype))
    log_mean_accept_ratio = tf.reduce_logsumexp(
        tf.minimum(kernel_results.log_accept_ratio, 0.)) - log_n
    adjustment = tf.where(
        log_mean_accept_ratio < tf.cast(
            tf.log(target_rate), log_mean_accept_ratio.dtype),
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
        return tf.cond(step_counter < num_adaptation_steps,
                       build_assign_op,
                       lambda: step_size_var)

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

  # Target distribution is proportional to: `exp(-x (1 + x))`.
  def unnormalized_log_prob(x):
    return -x - x**2.

  # Create state to hold updated `step_size`.
  step_size = tf.get_variable(
      name='step_size',
      initializer=1.,
      use_resource=True,  # For TFE compatibility.
      trainable=False)

  # Initialize the HMC transition kernel.
  num_results = int(10e3)
  num_burnin_steps = int(1e3)
  hmc = tfp.mcmc.HamiltonianMonteCarlo(
      target_log_prob_fn=unnormalized_log_prob,
      num_leapfrog_steps=3,
      step_size=step_size,
      step_size_update_fn=tfp.mcmc.make_simple_step_size_update_policy(
        num_adaptation_steps=int(num_burnin_steps * 0.8)))

  # Run the chain (with burn-in).
  samples, kernel_results = tfp.mcmc.sample_chain(
      num_results=num_results,
      num_burnin_steps=num_burnin_steps,
      current_state=1.,
      kernel=hmc)

  # Initialize all constructed variables.
  init_op = tf.global_variables_initializer()

  with tf.Session() as sess:
    init_op.run()
    samples_, kernel_results_ = sess.run([samples, kernel_results])

  print('mean:{:.4f}  stddev:{:.4f}  acceptance:{:.4f}'.format(
      samples_.mean(), samples_.std(), kernel_results_.is_accepted.mean()))
  # mean:-0.5003  stddev:0.7711  acceptance:0.6240
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

  tfd = tfp.distributions

  def make_training_data(num_samples, dims, sigma):
    dt = np.asarray(sigma).dtype
    zeros = tf.zeros(dims, dtype=dt)
    x = tf.transpose(tfd.MultivariateNormalDiag(loc=zeros).sample(
        num_samples, seed=1))  # [d, n]
    w = tfd.MultivariateNormalDiag(
        loc=zeros,
        scale_identity_multiplier=sigma).sample([1], seed=2)  # [1, d]
    noise = tfd.Normal(loc=np.array(0, dt), scale=np.array(1, dt)).sample(
        num_samples, seed=3)  # [n]
    y = tf.matmul(w, x) + noise  # [1, n]
    return y[0], x, w[0]

  def make_weights_prior(dims, dtype):
    return tfd.MultivariateNormalDiag(
        loc=tf.zeros([dims], dtype=dtype),
        scale_identity_multiplier=tf.exp(tf.get_variable(
            name='log_sigma',
            initializer=np.array(0, dtype),
            use_resource=True)))

  def make_response_likelihood(w, x):
    w_shape = tf.pad(
        tf.shape(w),
        paddings=[[tf.where(tf.rank(w) > 1, 0, 1), 0]],
        constant_values=1)
    y_shape = tf.concat([tf.shape(w)[:-1], [tf.shape(x)[-1]]], axis=0)
    w_expand = tf.reshape(w, w_shape)
    return tfd.Normal(
        loc=tf.reshape(tf.matmul(w_expand, x), y_shape),
        scale=np.array(1, w.dtype.as_numpy_dtype))  # [n]

  # Setup assumptions.
  dtype = np.float32
  num_samples = 500
  dims = 10

  weights_prior_true_scale = np.array(0.3, dtype)
  with tf.Session() as sess:
    y, x, true_weights = sess.run(
        make_training_data(num_samples, dims, weights_prior_true_scale))

  prior = make_weights_prior(dims, dtype)
  def unnormalized_posterior_log_prob(w):
    likelihood = make_response_likelihood(w, x)
    return (prior.log_prob(w)
            + tf.reduce_sum(likelihood.log_prob(y), axis=-1))  # [m]

  weights_chain_start = tf.placeholder(dtype, shape=[dims])

  step_size = tf.get_variable(
      name='step_size',
      initializer=np.array(0.05, dtype),
      use_resource=True,
      trainable=False)

  num_results = 2
  weights, kernel_results = tfp.mcmc.sample_chain(
      num_results=num_results,
      num_burnin_steps=0,
      current_state=weights_chain_start,
      kernel=tfp.mcmc.HamiltonianMonteCarlo(
          target_log_prob_fn=unnormalized_posterior_log_prob,
          num_leapfrog_steps=2,
          step_size=step_size,
          step_size_update_fn=tfp.mcmc.make_simple_step_size_update_policy(
            num_adaptation_steps=None),
          state_gradients_are_stopped=True))

  avg_acceptance_ratio = tf.reduce_mean(
      tf.exp(tf.minimum(kernel_results.log_accept_ratio, 0.)))

  # We do an optimization step to propagate `log_sigma` after two HMC steps to
  # propagate `weights`.
  loss = -tf.reduce_mean(kernel_results.accepted_results.target_log_prob)
  optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
  train_op = optimizer.minimize(loss)

  with tf.variable_scope(tf.get_variable_scope(), reuse=True):
    weights_prior_estimated_scale = tf.exp(
        tf.get_variable(name='log_sigma', dtype=dtype))

  init_op = tf.global_variables_initializer()

  num_iters = int(40)

  weights_prior_estimated_scale_ = np.zeros(num_iters, dtype)
  weights_ = np.zeros([num_iters + 1, dims], dtype)
  weights_[0] = np.random.randn(dims).astype(dtype)

  with tf.Session() as sess:
    init_op.run()
    for iter_ in range(num_iters):
      [
          _,
          weights_prior_estimated_scale_[iter_],
          weights_[iter_ + 1],
          loss_,
          step_size_,
          avg_acceptance_ratio_,
      ] = sess.run([
          train_op,
          weights_prior_estimated_scale,
          weights[-1],
          loss,
          step_size,
          avg_acceptance_ratio,
      ], feed_dict={weights_chain_start: weights_[iter_]})
      print('iter:{:>2}  loss:{: 9.3f}  scale:{:.3f}  '
            'step_size:{:.4f}  avg_acceptance_ratio:{:.4f}').format(
                iter_, loss_, weights_prior_estimated_scale_[iter_],
                step_size_, avg_acceptance_ratio_))

  # Should converge to ~0.24.
  import matplotlib.pyplot as plt
  plot.plot(weights_prior_estimated_scale_)
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
               step_size_update_fn=None,
               seed=None,
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
      step_size_update_fn: Python `callable` taking current `step_size`
        (typically a `tf.Variable`) and `kernel_results` (typically
        `collections.namedtuple`) and returns updated step_size (`Tensor`s).
        Default value: `None` (i.e., do not update `step_size` automatically).
      seed: Python integer to seed the random number generator.
      name: Python `str` name prefixed to Ops created by this function.
        Default value: `None` (i.e., 'hmc_kernel').
    """
    impl = metropolis_hastings.MetropolisHastings(
        inner_kernel=UncalibratedHamiltonianMonteCarlo(
            target_log_prob_fn=target_log_prob_fn,
            step_size=step_size,
            num_leapfrog_steps=num_leapfrog_steps,
            state_gradients_are_stopped=state_gradients_are_stopped,
            seed=seed,
            name='hmc_kernel' if name is None else name),
        seed=seed)
    parameters = impl.inner_kernel.parameters.copy()
    parameters['step_size_update_fn'] = step_size_update_fn
    self._impl = impl
    self._parameters = parameters

  @property
  def target_log_prob_fn(self):
    return self._impl.inner_kernel.target_log_prob_fn

  @property
  def step_size(self):
    return self._impl.inner_kernel.step_size

  @property
  def num_leapfrog_steps(self):
    return self._impl.inner_kernel.num_leapfrog_steps

  @property
  def state_gradients_are_stopped(self):
    return self._impl.inner_kernel.state_gradients_are_stopped

  @property
  def step_size_update_fn(self):
    return self._parameters['step_size_update_fn']

  @property
  def seed(self):
    return self._impl.inner_kernel.seed

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

  def one_step(self, current_state, previous_kernel_results):
    """Runs one iteration of Hamiltonian Monte Carlo.

    Args:
      current_state: `Tensor` or Python `list` of `Tensor`s representing the
        current state(s) of the Markov chain(s). The first `r` dimensions index
        independent chains, `r = tf.rank(target_log_prob_fn(*current_state))`.
      previous_kernel_results: `collections.namedtuple` containing `Tensor`s
        representing values from previous calls to this function (or from the
        `bootstrap_results` function.)

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
    previous_step_size_assign = (
        [] if self.step_size_update_fn is None
        else (previous_kernel_results.extra.step_size_assign
              if mcmc_util.is_list_like(
                  previous_kernel_results.extra.step_size_assign)
              else [previous_kernel_results.extra.step_size_assign]))

    with tf.control_dependencies(previous_step_size_assign):
      next_state, kernel_results = self._impl.one_step(
          current_state, previous_kernel_results)
      if self.step_size_update_fn is not None:
        step_size_assign = self.step_size_update_fn(  # pylint: disable=not-callable
            self.step_size, kernel_results)
        kernel_results = kernel_results._replace(
            extra=HamiltonianMonteCarloExtraKernelResults(
                step_size_assign=step_size_assign))
      return next_state, kernel_results

  def bootstrap_results(self, init_state):
    """Creates initial `previous_kernel_results` using a supplied `state`."""
    kernel_results = self._impl.bootstrap_results(init_state)
    if self.step_size_update_fn is not None:
      step_size_assign = self.step_size_update_fn(self.step_size, None)  # pylint: disable=not-callable
      kernel_results = kernel_results._replace(
          extra=HamiltonianMonteCarloExtraKernelResults(
              step_size_assign=step_size_assign))
    return kernel_results


class UncalibratedHamiltonianMonteCarlo(kernel_base.TransitionKernel):
  """Runs one step of Uncalibrated Hamiltonian Monte Carlo.

  Warning: this kernel will not result in a chain which converges to the
  `target_log_prob`. To get a convergent MCMC, use `HamiltonianMonteCarlo(...)`
  or `MetropolisHastings(UncalibratedHamiltonianMonteCarlo(...))`.

  For more details on `UncalibratedHamiltonianMonteCarlo`, see
  `HamiltonianMonteCarlo`.
  """

  @mcmc_util.set_doc(HamiltonianMonteCarlo.__init__.__doc__)
  def __init__(self,
               target_log_prob_fn,
               step_size,
               num_leapfrog_steps,
               state_gradients_are_stopped=False,
               seed=None,
               name=None):
    if seed is not None and tf.executing_eagerly():
      # TODO(b/68017812): Re-enable once TFE supports `tf.random_shuffle` seed.
      raise NotImplementedError('Specifying a `seed` when running eagerly is '
                                'not currently supported. To run in Eager '
                                'mode with a seed, use `tf.set_random_seed`.')
    self._seed_stream = distributions.SeedStream(seed, 'hmc_one_step')
    self._parameters = dict(
        target_log_prob_fn=target_log_prob_fn,
        step_size=step_size,
        num_leapfrog_steps=num_leapfrog_steps,
        state_gradients_are_stopped=state_gradients_are_stopped,
        seed=seed,
        name=name)

  @property
  def target_log_prob_fn(self):
    return self._parameters['target_log_prob_fn']

  @property
  def step_size(self):
    return self._parameters['step_size']

  @property
  def num_leapfrog_steps(self):
    return self._parameters['num_leapfrog_steps']

  @property
  def state_gradients_are_stopped(self):
    return self._parameters['state_gradients_are_stopped']

  @property
  def seed(self):
    return self._parameters['seed']

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

  @mcmc_util.set_doc(HamiltonianMonteCarlo.one_step.__doc__)
  def one_step(self, current_state, previous_kernel_results):
    with tf.name_scope(
        name=mcmc_util.make_name(self.name, 'hmc', 'one_step'),
        values=[self.step_size,
                self.num_leapfrog_steps,
                current_state,
                previous_kernel_results.target_log_prob,
                previous_kernel_results.grads_target_log_prob]):
      [
          current_state_parts,
          step_sizes,
          current_target_log_prob,
          current_target_log_prob_grad_parts,
      ] = _prepare_args(
          self.target_log_prob_fn,
          current_state,
          self.step_size,
          previous_kernel_results.target_log_prob,
          previous_kernel_results.grads_target_log_prob,
          maybe_expand=True,
          state_gradients_are_stopped=self.state_gradients_are_stopped)

      independent_chain_ndims = distribution_util.prefer_static_rank(
          current_target_log_prob)

      current_momentum_parts = []
      for x in current_state_parts:
        current_momentum_parts.append(tf.random_normal(
            shape=tf.shape(x),
            dtype=x.dtype.base_dtype,
            seed=self._seed_stream()))

      def _leapfrog_one_step(*args):
        """Closure representing computation done during each leapfrog step."""
        return _leapfrog_integrator_one_step(
            target_log_prob_fn=self.target_log_prob_fn,
            independent_chain_ndims=independent_chain_ndims,
            step_sizes=step_sizes,
            current_momentum_parts=args[0],
            current_state_parts=args[1],
            current_target_log_prob=args[2],
            current_target_log_prob_grad_parts=args[3],
            state_gradients_are_stopped=self.state_gradients_are_stopped)

      num_leapfrog_steps = tf.convert_to_tensor(
          self.num_leapfrog_steps, dtype=tf.int64, name='num_leapfrog_steps')

      [
          next_momentum_parts,
          next_state_parts,
          next_target_log_prob,
          next_target_log_prob_grad_parts,

      ] = tf.while_loop(
          cond=lambda i, *args: i < num_leapfrog_steps,
          body=lambda i, *args: [i + 1] + list(_leapfrog_one_step(*args)),
          loop_vars=[
              tf.zeros([], tf.int64, name='iter'),
              current_momentum_parts,
              current_state_parts,
              current_target_log_prob,
              current_target_log_prob_grad_parts
          ])[1:]

      def maybe_flatten(x):
        return x if mcmc_util.is_list_like(current_state) else x[0]

      return [
          maybe_flatten(next_state_parts),
          UncalibratedHamiltonianMonteCarloKernelResults(
              log_acceptance_correction=_compute_log_acceptance_correction(
                  current_momentum_parts,
                  next_momentum_parts,
                  independent_chain_ndims),
              target_log_prob=next_target_log_prob,
              grads_target_log_prob=next_target_log_prob_grad_parts,
          ),
      ]

  @mcmc_util.set_doc(HamiltonianMonteCarlo.bootstrap_results.__doc__)
  def bootstrap_results(self, init_state):
    with tf.name_scope(
        name=mcmc_util.make_name(self.name, 'hmc', 'bootstrap_results'),
        values=[init_state]):
      if not mcmc_util.is_list_like(init_state):
        init_state = [init_state]
      if self.state_gradients_are_stopped:
        init_state = [tf.stop_gradient(x) for x in init_state]
      else:
        init_state = [tf.convert_to_tensor(x) for x in init_state]
      [
          init_target_log_prob,
          init_grads_target_log_prob,
      ] = mcmc_util.maybe_call_fn_and_grads(self.target_log_prob_fn, init_state)
      return UncalibratedHamiltonianMonteCarloKernelResults(
          log_acceptance_correction=tf.zeros_like(init_target_log_prob),
          target_log_prob=init_target_log_prob,
          grads_target_log_prob=init_grads_target_log_prob,
      )


def _leapfrog_integrator_one_step(
    target_log_prob_fn,
    independent_chain_ndims,
    step_sizes,
    current_momentum_parts,
    current_state_parts,
    current_target_log_prob,
    current_target_log_prob_grad_parts,
    state_gradients_are_stopped=False,
    name=None):
  """Applies `num_leapfrog_steps` of the leapfrog integrator.

  Assumes a simple quadratic kinetic energy function: `0.5 ||momentum||**2`.

  #### Examples:

  ##### Simple quadratic potential.

  ```python
  import matplotlib.pyplot as plt
  %matplotlib inline
  import numpy as np
  import tensorflow as tf
  from tensorflow_probability.python.mcmc.hmc import _leapfrog_integrator
  tfd = tfp.distributions

  dims = 10
  num_iter = int(1e3)
  dtype = np.float32

  position = tf.placeholder(np.float32)
  momentum = tf.placeholder(np.float32)

  target_log_prob_fn = tfd.MultivariateNormalDiag(
      loc=tf.zeros(dims, dtype)).log_prob

  def _leapfrog_one_step(*args):
    # Closure representing computation done during each leapfrog step.
    return _leapfrog_integrator_one_step(
        target_log_prob_fn=target_log_prob_fn,
        independent_chain_ndims=0,
        step_sizes=[0.1],
        current_momentum_parts=args[0],
        current_state_parts=args[1],
        current_target_log_prob=args[2],
        current_target_log_prob_grad_parts=args[3])

  # Do leapfrog integration.
  [
      [next_momentum],
      [next_position],
      next_target_log_prob,
      next_target_log_prob_grad_parts,
  ] = tf.while_loop(
      cond=lambda *args: True,
      body=_leapfrog_one_step,
      loop_vars=[
        [momentum],
        [position],
        target_log_prob_fn(position),
        tf.gradients(target_log_prob_fn(position), position),
      ],
      maximum_iterations=3)

  momentum_ = np.random.randn(dims).astype(dtype)
  position_ = np.random.randn(dims).astype(dtype)
  positions = np.zeros([num_iter, dims], dtype)

  with tf.Session() as sess:
    for i in xrange(num_iter):
      position_, momentum_ = sess.run(
          [next_momentum, next_position],
          feed_dict={position: position_, momentum: momentum_})
      positions[i] = position_

  plt.plot(positions[:, 0]);  # Sinusoidal.
  ```

  Args:
    target_log_prob_fn: Python callable which takes an argument like
      `*current_state_parts` and returns its (possibly unnormalized) log-density
      under the target distribution.
    independent_chain_ndims: Scalar `int` `Tensor` representing the number of
      leftmost `Tensor` dimensions which index independent chains.
    step_sizes: Python `list` of `Tensor`s representing the step size for the
      leapfrog integrator. Must broadcast with the shape of
      `current_state_parts`.  Larger step sizes lead to faster progress, but
      too-large step sizes make rejection exponentially more likely. When
      possible, it's often helpful to match per-variable step sizes to the
      standard deviations of the target distribution in each variable.
    current_momentum_parts: Tensor containing the value(s) of the momentum
      variable(s) to update.
    current_state_parts: Python `list` of `Tensor`s representing the current
      state(s) of the Markov chain(s). The first `independent_chain_ndims` of
      the `Tensor`(s) index different chains.
    current_target_log_prob: `Tensor` representing the value of
      `target_log_prob_fn(*current_state_parts)`. The only reason to specify
      this argument is to reduce TF graph size.
    current_target_log_prob_grad_parts: Python list of `Tensor`s representing
      gradient of `target_log_prob_fn(*current_state_parts`) wrt
      `current_state_parts`. Must have same shape as `current_state_parts`. The
      only reason to specify this argument is to reduce TF graph size.
    state_gradients_are_stopped: Python `bool` indicating that the proposed new
      state be run through `tf.stop_gradient`. This is particularly useful when
      combining optimization over samples from the HMC chain.
      Default value: `False` (i.e., do not apply `stop_gradient`).
    name: Python `str` name prefixed to Ops created by this function.
      Default value: `None` (i.e., 'hmc_leapfrog_integrator').

  Returns:
    proposed_momentum_parts: Updated value of the momentum.
    proposed_state_parts: Tensor or Python list of `Tensor`s representing the
      state(s) of the Markov chain(s) at each result step. Has same shape as
      input `current_state_parts`.
    proposed_target_log_prob: `Tensor` representing the value of
      `target_log_prob_fn` at `next_state`.
    proposed_target_log_prob_grad_parts: Gradient of `proposed_target_log_prob`
      wrt `next_state`.

  Raises:
    ValueError: if `len(momentum_parts) != len(state_parts)`.
    ValueError: if `len(state_parts) != len(step_sizes)`.
    ValueError: if `len(state_parts) != len(grads_target_log_prob)`.
    TypeError: if `not target_log_prob.dtype.is_floating`.
  """
  # Note on per-variable step sizes:
  #
  # Using per-variable step sizes is equivalent to using the same step
  # size for all variables and adding a diagonal mass matrix in the
  # kinetic energy term of the Hamiltonian being integrated. This is
  # hinted at by Neal (2011) but not derived in detail there.
  #
  # Let x and v be position and momentum variables respectively.
  # Let g(x) be the gradient of `target_log_prob_fn(x)`.
  # Let S be a diagonal matrix of per-variable step sizes.
  # Let the Hamiltonian H(x, v) = -target_log_prob_fn(x) + 0.5 * ||v||**2.
  #
  # Using per-variable step sizes gives the updates
  # v'  = v  + 0.5 * matmul(S, g(x))
  # x'' = x  + matmul(S, v')
  # v'' = v' + 0.5 * matmul(S, g(x''))
  #
  # Let u = matmul(inv(S), v).
  # Multiplying v by inv(S) in the updates above gives the transformed dynamics
  # u'  = matmul(inv(S), v')  = matmul(inv(S), v) + 0.5 * g(x)
  #                           = u + 0.5 * g(x)
  # x'' = x + matmul(S, v') = x + matmul(S**2, u')
  # u'' = matmul(inv(S), v'') = matmul(inv(S), v') + 0.5 * g(x'')
  #                           = u' + 0.5 * g(x'')
  #
  # These are exactly the leapfrog updates for the Hamiltonian
  # H'(x, u) = -target_log_prob_fn(x) + 0.5 * u^T S**2 u
  #          = -target_log_prob_fn(x) + 0.5 * ||v||**2 = H(x, v).
  #
  # To summarize:
  #
  # * Using per-variable step sizes implicitly simulates the dynamics
  #   of the Hamiltonian H' (which are energy-conserving in H'). We
  #   keep track of v instead of u, but the underlying dynamics are
  #   the same if we transform back.
  # * The value of the Hamiltonian H'(x, u) is the same as the value
  #   of the original Hamiltonian H(x, v) after we transform back from
  #   u to v.
  # * Sampling v ~ N(0, I) is equivalent to sampling u ~ N(0, S**-2).
  #
  # So using per-variable step sizes in HMC will give results that are
  # exactly identical to explicitly using a diagonal mass matrix.

  with tf.name_scope(
      name, 'hmc_leapfrog_integrator_one_step',
      [independent_chain_ndims, step_sizes,
       current_momentum_parts, current_state_parts,
       current_target_log_prob, current_target_log_prob_grad_parts]):

    # Step 1: Update momentum.
    proposed_momentum_parts = [
        v + 0.5 * tf.cast(eps, v.dtype) * g
        for v, eps, g
        in zip(current_momentum_parts,
               step_sizes,
               current_target_log_prob_grad_parts)]

    # Step 2: Update state.
    proposed_state_parts = [
        x + tf.cast(eps, v.dtype) * v
        for x, eps, v
        in zip(current_state_parts,
               step_sizes,
               proposed_momentum_parts)]

    if state_gradients_are_stopped:
      proposed_state_parts = [tf.stop_gradient(x) for x in proposed_state_parts]

    # Step 3a: Re-evaluate target-log-prob (and grad) at proposed state.
    [
        proposed_target_log_prob,
        proposed_target_log_prob_grad_parts,
    ] = mcmc_util.maybe_call_fn_and_grads(
        target_log_prob_fn,
        proposed_state_parts)

    if not proposed_target_log_prob.dtype.is_floating:
      raise TypeError('`target_log_prob_fn` must produce a `Tensor` '
                      'with `float` `dtype`.')

    if any(g is None for g in proposed_target_log_prob_grad_parts):
      raise ValueError(
          'Encountered `None` gradient. Does your target `target_log_prob_fn` '
          'access all `tf.Variable`s via `tf.get_variable`?\n'
          '  current_state_parts: {}\n'
          '  proposed_state_parts: {}\n'
          '  proposed_target_log_prob_grad_parts: {}'.format(
              current_state_parts,
              proposed_state_parts,
              proposed_target_log_prob_grad_parts))

    # Step 3b: Update momentum (again).
    proposed_momentum_parts = [
        v + 0.5 * tf.cast(eps, v.dtype) * g
        for v, eps, g
        in zip(proposed_momentum_parts,
               step_sizes,
               proposed_target_log_prob_grad_parts)]

    return [
        proposed_momentum_parts,
        proposed_state_parts,
        proposed_target_log_prob,
        proposed_target_log_prob_grad_parts,
    ]


def _compute_log_acceptance_correction(current_momentums,
                                       proposed_momentums,
                                       independent_chain_ndims,
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
    name: Python `str` name prefixed to Ops created by this function.
      Default value: `None` (i.e., 'compute_log_acceptance_correction').

  Returns:
    log_acceptance_correction: `Tensor` representing the `log`
      acceptance-correction.  (See docstring for mathematical definition.)
  """
  with tf.name_scope(
      name, 'compute_log_acceptance_correction',
      [independent_chain_ndims, current_momentums, proposed_momentums]):
    log_current_kinetic, log_proposed_kinetic = [], []
    for current_momentum, proposed_momentum in zip(
        current_momentums, proposed_momentums):
      axis = tf.range(independent_chain_ndims, tf.rank(current_momentum))
      log_current_kinetic.append(_log_sum_sq(current_momentum, axis))
      log_proposed_kinetic.append(_log_sum_sq(proposed_momentum, axis))
    current_kinetic = 0.5 * tf.exp(
        tf.reduce_logsumexp(tf.stack(log_current_kinetic, axis=-1), axis=-1))
    proposed_kinetic = 0.5 * tf.exp(
        tf.reduce_logsumexp(tf.stack(log_proposed_kinetic, axis=-1), axis=-1))
    return mcmc_util.safe_sum([current_kinetic, -proposed_kinetic])


def _prepare_args(target_log_prob_fn,
                  state,
                  step_size,
                  target_log_prob=None,
                  grads_target_log_prob=None,
                  maybe_expand=False,
                  state_gradients_are_stopped=False):
  """Helper which processes input args to meet list-like assumptions."""
  state_parts = list(state) if mcmc_util.is_list_like(state) else [state]
  state_parts = [tf.convert_to_tensor(s, name='current_state')
                 for s in state_parts]
  if state_gradients_are_stopped:
    state_parts = [tf.stop_gradient(x) for x in state_parts]
  target_log_prob, grads_target_log_prob = mcmc_util.maybe_call_fn_and_grads(
      target_log_prob_fn,
      state_parts,
      target_log_prob,
      grads_target_log_prob)
  step_sizes = (list(step_size) if mcmc_util.is_list_like(step_size)
                else [step_size])
  step_sizes = [
      tf.convert_to_tensor(
          s, name='step_size', dtype=target_log_prob.dtype)
      for s in step_sizes]
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


def _log_sum_sq(x, axis=None):
  """Computes log(sum(x**2))."""
  return tf.reduce_logsumexp(2. * tf.log(tf.abs(x)), axis)
