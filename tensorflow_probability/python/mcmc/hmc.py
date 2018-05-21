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

import tensorflow as tf

from tensorflow_probability.python.mcmc import kernel as kernel_base
from tensorflow_probability.python.mcmc import metropolis_hastings
from tensorflow_probability.python.mcmc import util as mcmc_util
from tensorflow.contrib import eager as tfe
from tensorflow.python.ops.distributions import util as distributions_util


__all__ = [
    'HamiltonianMonteCarlo',
    'UncalibratedHamiltonianMonteCarlo',
]


UncalibratedHamiltonianMonteCarloKernelResults = collections.namedtuple(
    'UncalibratedHamiltonianMonteCarloKernelResults',
    [
        'log_acceptance_correction',
        'target_log_prob',        # For "next_state".
        'grads_target_log_prob',  # For "next_state".
    ])


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
  import numpy as np

  tfd = tf.contrib.distributions

  # Tuning acceptance rates:
  dtype = np.float32
  num_warmup_iter = 500
  num_chain_iter = 500
  # Set the target average acceptance ratio for the HMC as suggested by
  # Beskos et al. (2013):
  # https://projecteuclid.org/download/pdfview_1/euclid.bj/1383661192

  target_accept_rate = 0.651

  x = tf.get_variable(name='x', initializer=dtype(1))
  step_size = tf.get_variable(name='step_size', initializer=dtype(1))

  # Target distribution is standard univariate Normal.
  target = tfd.Normal(loc=dtype(0), scale=dtype(1))

  # Initialize the HMC sampler.
  hmc = tfp.mcmc.HamiltonianMonteCarlo(
      target_log_prob_fn=target.log_prob,
      step_size=step_size,
      num_leapfrog_steps=3)

  # One iteration of the HMC
  next_x, other_results = hmc.one_step(
      current_state=x,
      previous_kernel_results=hmc.bootstrap_results(x))

  x_update = x.assign(next_x)

  # Adapt the step size using standard adaptive MCMC procedure. See Section 4.2
  # of Andrieu and Thoms (2008):
  # http://www4.ncsu.edu/~rsmith/MA797V_S12/Andrieu08_AdaptiveMCMC_Tutorial.pdf

  step_size_update = step_size.assign_add(
      step_size * tf.where(
          tf.exp(tf.minimum(other_results.log_accept_ratio, 0.)) >
              target_accept_rate,
          0.01, -0.01))

  # Note, the adaptations are performed during warmup only.
  warmup = tf.group([x_update, step_size_update])

  init = tf.global_variables_initializer()

  with tf.Session() as sess:
    sess.run(init)
    # Warm up the sampler and adapt the step size
    for _ in xrange(num_warmup_iter):
      sess.run(warmup)
    # Collect samples without adapting step size
    samples = np.zeros([num_chain_iter])
    for i in xrange(num_chain_iter):
      _, x_,= sess.run([x_update, x])
      samples[i] = x_

  print(samples.mean(), samples.std())
  ```

  ##### Estimate parameters of a more complicated posterior.

  In this example, we'll use Monte-Carlo EM to find best-fit parameters. See
  ["Implementations of the Monte Carlo EM algorithm " by Levine and Casella](
  https://ecommons.cornell.edu/bitstream/handle/1813/32030/BU-1431-M.pdf?sequence=1).

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

  We now implement MCEM using `tensorflow_probability` intrinsics.

  ```python
  import tensorflow as tf
  import tensorflow_probability as tfp
  import numpy as np

  tfd = tf.contrib.distributions

  def make_training_data(num_samples, dims, sigma):
    dt = np.asarray(sigma).dtype
    zeros = tf.zeros(dims, dtype=dt)
    x = tfd.MultivariateNormalDiag(
        loc=zeros).sample(num_samples, seed=1)
    w = tfd.MultivariateNormalDiag(
        loc=zeros,
        scale_identity_multiplier=sigma).sample(seed=2)
    noise = tfd.Normal(
        loc=dt.type(0),
        scale=dt.type(1)).sample(num_samples, seed=3)
    y = tf.tensordot(x, w, axes=[[1], [0]]) + noise
    return y, x, w

  def make_prior(sigma, dims):
    # p(w | sigma)
    return tfd.MultivariateNormalDiag(
        loc=tf.zeros([dims], dtype=sigma.dtype),
        scale_identity_multiplier=sigma)

  def make_likelihood(x, w):
    # p(y | x, w)
    return tfd.MultivariateNormalDiag(
        loc=tf.tensordot(x, w, axes=[[1], [0]]))

  # Setup assumptions.
  dtype = np.float32
  num_samples = 150
  dims = 10
  num_iters = int(5e3)

  true_sigma = dtype(0.3)
  y, x, true_weights = make_training_data(num_samples, dims, true_sigma)

  # Estimate of `log(true_sigma)`.
  log_sigma = tf.get_variable(name='log_sigma', initializer=dtype(0))
  sigma = tf.exp(log_sigma)

  # State of the Markov chain.
  # We set `trainable=False` so it is unaffected by the M-step.
  weights = tf.get_variable(
      name='weights',
      initializer=np.random.randn(dims).astype(dtype),
      trainable=False)

  prior = make_prior(sigma, dims)

  def joint_log_prob(w):
    # f(w) = log p(w, y | x)
    return prior.log_prob(w) + make_likelihood(x, w).log_prob(y)

  # Initialize the HMC sampler.
  hmc = tfp.mcmc.HamiltonianMonteCarlo(
      target_log_prob_fn=joint_log_prob,
      step_size=0.1,
      num_leapfrog_steps=5)

  weights_update = weights.assign(
      hmc.one_step(weights, hmc.bootstrap_results(weights))[0])

  # We do an optimization step to propagate `log_sigma` after one HMC step to
  # propagate `weights`. The loss function for the optimization algorithm is
  # exactly the prior distribution since the likelihood does not depend on
  # `log_sigma`.
  with tf.control_dependencies([weights_update]):
    loss = -prior.log_prob(weights)

  optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
  log_sigma_update = optimizer.minimize(loss)

  init = tf.global_variables_initializer()

  sigma_history = np.zeros(num_iters, dtype)
  weights_history = np.zeros([num_iters, dims], dtype)

  with tf.Session() as sess:
    sess.run(init)
    for i in xrange(num_iters):
      _, sigma_, weights_ = sess.run([log_sigma_update, sigma, weights])
      weights_history[i, :] = weights_
      sigma_history[i] = sigma_
    true_weights_ = sess.run(true_weights)

  # Should oscillate around true_sigma.
  import matplotlib.pyplot as plt
  plt.plot(sigma_history)
  plt.ylabel('sigma')
  plt.xlabel('iteration')

  # Mean error should be close to zero
  print('mean error:', np.abs(np.mean(sigma_history) - true_sigma))
  ```

  #### References

  [1]: Radford Neal. MCMC Using Hamiltonian Dynamics. _Handbook of Markov Chain
       Monte Carlo_, 2011. https://arxiv.org/abs/1206.1901
  """

  def __init__(self,
               target_log_prob_fn,
               step_size,
               num_leapfrog_steps,
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
      seed: Python integer to seed the random number generator.
      name: Python `str` name prefixed to Ops created by this function.
        Default value: `None` (i.e., 'hmc_kernel').

    Returns:
      next_state: Tensor or Python list of `Tensor`s representing the state(s)
        of the Markov chain(s) at each result step. Has same shape as
        `current_state`.
      kernel_results: `collections.namedtuple` of internal calculations used to
        advance the chain.

    Raises:
      ValueError: if there isn't one `step_size` or a list with same length as
        `current_state`.
    """
    self._impl = metropolis_hastings.MetropolisHastings(
        inner_kernel=UncalibratedHamiltonianMonteCarlo(
            target_log_prob_fn=target_log_prob_fn,
            step_size=step_size,
            num_leapfrog_steps=num_leapfrog_steps,
            seed=seed,
            name='hmc_kernel' if name is None else name),
        seed=seed)

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
  def seed(self):
    return self._impl.inner_kernel.seed

  @property
  def name(self):
    return self._impl.inner_kernel.name

  @property
  def parameters(self):
    """Return `dict` of ``__init__`` arguments and their values."""
    return self._impl.inner_kernel.parameters

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
    return self._impl.one_step(current_state, previous_kernel_results)

  def bootstrap_results(self, init_state):
    """Creates initial `previous_kernel_results` using a supplied `state`."""
    return self._impl.bootstrap_results(init_state)


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
               seed=None,
               name=None):
    if seed is not None and tfe.executing_eagerly():
      # TODO(b/68017812): Re-enable once TFE supports `tf.random_shuffle` seed.
      raise NotImplementedError('Specifying a `seed` when running eagerly is '
                                'not currently supported. To run in Eager '
                                'mode with a seed, use `tf.set_random_seed`.')
    self._seed_stream = tf.contrib.distributions.SeedStream(
        seed, 'hmc_one_step')
    self._parameters = dict(
        target_log_prob_fn=target_log_prob_fn,
        step_size=step_size,
        num_leapfrog_steps=num_leapfrog_steps,
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
          maybe_expand=True)

      independent_chain_ndims = distributions_util.prefer_static_rank(
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
            current_target_log_prob_grad_parts=args[3])

      # Do leapfrog integration.
      [
          next_momentum_parts,
          next_state_parts,
          next_target_log_prob,
          next_target_log_prob_grad_parts,
      ] = tf.while_loop(
          cond=lambda i, *args: i < self.num_leapfrog_steps,
          body=lambda i, *args: [i + 1] + list(_leapfrog_one_step(*args)),
          loop_vars=[
              tf.zeros([], tf.int32, name='iter'),
              current_momentum_parts,
              current_state_parts,
              current_target_log_prob,
              current_target_log_prob_grad_parts,
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
  tfd = tf.contrib.distributions

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
        v + 0.5 * eps * g
        for v, eps, g
        in zip(current_momentum_parts,
               step_sizes,
               current_target_log_prob_grad_parts)]

    # Step 2: Update state.
    proposed_state_parts = [
        x + eps * v
        for x, eps, v
        in zip(current_state_parts,
               step_sizes,
               proposed_momentum_parts)]

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
        v + 0.5 * eps * g
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
                  maybe_expand=False):
  """Helper which processes input args to meet list-like assumptions."""
  state_parts = list(state) if mcmc_util.is_list_like(state) else [state]
  state_parts = [tf.convert_to_tensor(s, name='current_state')
                 for s in state_parts]
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
