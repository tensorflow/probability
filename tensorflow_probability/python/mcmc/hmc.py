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

from tensorflow_probability.python.mcmc import kernel as kernel_base
from tensorflow_probability.python.mcmc import util as mcmc_util
from tensorflow.python.ops.distributions import util as distributions_util


__all__ = [
    'HamiltonianMonteCarlo',
]


KernelResults = collections.namedtuple(
    'KernelResults',
    [
        'current_grads_target_log_prob',  # "Current result" means "accepted".
        'current_target_log_prob',  # "Current result" means "accepted".
        'is_accepted',
        'log_accept_ratio',
        'proposed_grads_target_log_prob',
        'proposed_state',
        'proposed_target_log_prob',
    ])


class HamiltonianMonteCarlo(kernel_base.TransitionKernel):
  """Runs one iteration of Hamiltonian Monte Carlo.

  Hamiltonian Monte Carlo (HMC) is a Markov chain Monte Carlo (MCMC)
  algorithm that takes a series of gradient-informed steps to produce
  a Metropolis proposal. This function takes one random HMC step from
  a given `current_state`.

  This function can update multiple chains in parallel. It assumes that all
  leftmost dimensions of `current_state` index independent chain states (and
  are therefore updated independently). The output of
  `target_log_prob_fn(*current_state)` should sum log-probabilities across all
  event dimensions. Slices along the rightmost dimensions may have different
  target distributions; for example, `current_state[0, :]` could have a
  different target distribution from `current_state[1, :]`. These semantics
  are governed by `target_log_prob_fn(*current_state)`. (The number of
  independent chains is `tf.size(target_log_prob_fn(*current_state))`.)

  #### Examples:

  ##### Simple chain with warm-up.

  ```python
  import tensorflow as tf
  import tensorflow_probability as tfp
  tfd = tf.contrib.distributions

  # Tuning acceptance rates:
  dtype = np.float32
  target_accept_rate = 0.631
  num_warmup_iter = 500
  num_chain_iter = 500

  x = tf.get_variable(name='x', initializer=dtype(1))
  step_size = tf.get_variable(name='step_size', initializer=dtype(1))

  target = tfd.Normal(loc=dtype(0), scale=dtype(1))

  hmc = tfp.mcmc.HamiltonianMonteCarlo(
      target_log_prob_fn=target.log_prob,
      step_size=step_size,
      num_leapfrog_steps=3)

  next_x, other_results = hmc(current_state=x,
                              previous_kernel_results=hmc.bootsrap_results(x))

  x_update = x.assign(next_x)

  step_size_update = step_size.assign_add(
      step_size * tf.where(
          tf.exp(tf.minimum(other_results.log_accept_ratio), 0.) >
              target_accept_rate,
          0.01, -0.01))

  warmup = tf.group([x_update, step_size_update])

  tf.global_variables_initializer().run()

  sess.graph.finalize()  # No more graph building.

  # Warm up the sampler and adapt the step size
  for _ in xrange(num_warmup_iter):
    sess.run(warmup)

  # Collect samples without adapting step size
  samples = np.zeros([num_chain_iter])
  for i in xrange(num_chain_iter):
    _, x_, target_log_prob_, grad_ = sess.run([
        x_update,
        x,
        other_results.target_log_prob,
        other_results.grads_target_log_prob])
    samples[i] = x_

  print(samples.mean(), samples.std())
  ```

  ##### Sample from more complicated posterior.

  I.e.,

  ```none
    W ~ MVN(loc=0, scale=sigma * eye(dims))
    for i=1...num_samples:
        X[i] ~ MVN(loc=0, scale=eye(dims))
      eps[i] ~ Normal(loc=0, scale=1)
        Y[i] = X[i].T * W + eps[i]
  ```

  ```python
  import tensorflow as tf
  import tensorflow_probability as tfp
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
        loc=dt(0),
        scale=dt(1)).sample(num_samples, seed=3)
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

  true_sigma = dtype(0.5)
  y, x, true_weights = make_training_data(num_samples, dims, true_sigma)

  # Estimate of `log(true_sigma)`.
  log_sigma = tf.get_variable(name='log_sigma', initializer=dtype(0))
  sigma = tf.exp(log_sigma)

  # State of the Markov chain.
  weights = tf.get_variable(
      name='weights',
      initializer=np.random.randn(dims).astype(dtype))

  prior = make_prior(sigma, dims)

  def joint_log_prob_fn(w):
    # f(w) = log p(w, y | x)
    return prior.log_prob(w) + make_likelihood(x, w).log_prob(y)

  hmc = tfp.mcmc.HamiltonianMonteCarlo(
      target_log_prob_fn=joint_log_prob,
      step_size=0.1,
      num_leapfrog_steps=5)

  weights_update = weights.assign(
      hmc(weights, hmc.bootstrap_results(weights))[0])

  with tf.control_dependencies([weights_update]):
    loss = -prior.log_prob(weights)

  optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
  log_sigma_update = optimizer.minimize(loss, var_list=[log_sigma])

  sess.graph.finalize()  # No more graph building.

  tf.global_variables_initializer().run()

  sigma_history = np.zeros(num_iters, dtype)
  weights_history = np.zeros([num_iters, dims], dtype)

  for i in xrange(num_iters):
    _, sigma_, weights_, _ = sess.run([log_sigma_update, sigma, weights])
    weights_history[i, :] = weights_
    sigma_history[i] = sigma_

  true_weights_ = sess.run(true_weights)

  # Should converge to something close to true_sigma.
  import matplotlib.plot as plt
  plt.plot(sigma_history);
  plt.ylabel('sigma');
  plt.xlabel('iteration');
  ```

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
        for. Total progress per HMC step is roughly proportional to `step_size *
        num_leapfrog_steps`.
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
    self._target_log_prob_fn = target_log_prob_fn
    self._step_size = step_size
    self._num_leapfrog_steps = num_leapfrog_steps
    self._seed = seed
    self._name = name

  @property
  def target_log_prob_fn(self):
    return self._target_log_prob_fn

  @property
  def step_size(self):
    return self._step_size

  @property
  def num_leapfrog_steps(self):
    return self._num_leapfrog_steps

  @property
  def seed(self):
    return self._seed

  @property
  def name(self):
    return self._name

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
        of the Markov chain(s) at each result step. Has same shape as
        `current_state`.
      kernel_results: `collections.namedtuple` of internal calculations used to
        advance the chain.

    Raises:
      ValueError: if there isn't one `step_size` or a list with same length as
        `current_state`.
    """
    with tf.name_scope(
        self.name, 'hmc_kernel',
        [self.step_size, self.num_leapfrog_steps, self.seed,
         current_state,
         previous_kernel_results.current_target_log_prob,
         previous_kernel_results.current_grads_target_log_prob]):
      with tf.name_scope('initialize'):
        [
            current_state_parts,
            step_sizes,
            current_target_log_prob,
            current_grads_target_log_prob,
        ] = _prepare_args(
            self.target_log_prob_fn,
            current_state,
            self.step_size,
            previous_kernel_results.current_target_log_prob,
            previous_kernel_results.current_grads_target_log_prob,
            maybe_expand=True)

        current_momentums = []
        for s in current_state_parts:
          self._seed = distributions_util.gen_new_seed(
              self.seed, salt='hmc_kernel_momentums')
          current_momentums.append(tf.random_normal(
              shape=tf.shape(s),
              dtype=s.dtype.base_dtype,
              seed=self.seed))

        num_leapfrog_steps = tf.convert_to_tensor(
            self.num_leapfrog_steps,
            dtype=tf.int32,
            name='num_leapfrog_steps')

        independent_chain_ndims = distributions_util.prefer_static_rank(
            current_target_log_prob)

      [
          proposed_momentums,
          proposed_state_parts,
          proposed_target_log_prob,
          proposed_grads_target_log_prob,
      ] = _leapfrog_integrator(current_momentums,
                               self.target_log_prob_fn,
                               current_state_parts,
                               step_sizes,
                               num_leapfrog_steps,
                               current_target_log_prob,
                               current_grads_target_log_prob)

      log_accept_ratio = -_compute_energy_change(current_target_log_prob,
                                                 current_momentums,
                                                 proposed_target_log_prob,
                                                 proposed_momentums,
                                                 independent_chain_ndims)

      # If proposed state reduces likelihood: randomly accept.
      # If proposed state increases likelihood: always accept.
      # I.e., u < min(1, accept_ratio),  where u ~ Uniform[0,1)
      #       ==> log(u) < log_accept_ratio
      self._seed = distributions_util.gen_new_seed(
          self.seed, salt='metropolis_hastings_one_step')
      log_uniform = tf.log(tf.random_uniform(
          shape=tf.shape(proposed_target_log_prob),
          dtype=proposed_target_log_prob.dtype,
          seed=self.seed))
      is_accepted = log_uniform < log_accept_ratio

      accepted_target_log_prob = tf.where(
          is_accepted,
          proposed_target_log_prob,
          current_target_log_prob)

      next_state_parts = mcmc_util.choose(
          is_accepted,
          proposed_state_parts,
          current_state_parts,
          independent_chain_ndims)

      accepted_grads_target_log_prob = mcmc_util.choose(
          is_accepted,
          proposed_grads_target_log_prob,
          current_grads_target_log_prob,
          independent_chain_ndims)

      def maybe_flatten(x):
        return x if mcmc_util.is_list_like(current_state) else x[0]

      return [
          maybe_flatten(next_state_parts),
          KernelResults(
              log_accept_ratio=log_accept_ratio,
              current_grads_target_log_prob=accepted_grads_target_log_prob,
              current_target_log_prob=accepted_target_log_prob,
              is_accepted=is_accepted,
              proposed_grads_target_log_prob=proposed_grads_target_log_prob,
              proposed_state=maybe_flatten(proposed_state_parts),
              proposed_target_log_prob=proposed_target_log_prob,
          ),
      ]

  def bootstrap_results(self, init_state):
    init_target_log_prob = self.target_log_prob_fn(*(
        init_state if mcmc_util.is_list_like(init_state) else [init_state]))
    init_grads_target_log_prob = tf.gradients(init_target_log_prob, init_state)
    return KernelResults(
        log_accept_ratio=init_target_log_prob,
        current_grads_target_log_prob=init_grads_target_log_prob,
        current_target_log_prob=init_target_log_prob,
        is_accepted=tf.ones_like(init_target_log_prob, tf.bool),
        proposed_grads_target_log_prob=init_grads_target_log_prob,
        proposed_state=init_state,
        proposed_target_log_prob=init_target_log_prob,
    )


def _leapfrog_integrator(current_momentums,
                         target_log_prob_fn,
                         current_state_parts,
                         step_sizes,
                         num_leapfrog_steps,
                         current_target_log_prob=None,
                         current_grads_target_log_prob=None,
                         name=None):
  """Applies `num_leapfrog_steps` of the leapfrog integrator.

  Assumes a simple quadratic kinetic energy function: `0.5 ||momentum||**2`.

  #### Examples:

  ##### Simple quadratic potential.

  ```python
  tfd = tf.contrib.distributions

  dims = 10
  num_iter = int(1e3)
  dtype = np.float32

  position = tf.placeholder(np.float32)
  momentum = tf.placeholder(np.float32)

  [
      next_momentums,
      next_positions,
  ] = hmc._leapfrog_integrator(
      current_momentums=[momentum],
      target_log_prob_fn=tfd.MultivariateNormalDiag(
          loc=tf.zeros(dims, dtype)).log_prob,
      current_state_parts=[position],
      step_sizes=0.1,
      num_leapfrog_steps=3)[:2]

  sess.graph.finalize()  # No more graph building.

  momentum_ = np.random.randn(dims).astype(dtype)
  position_ = np.random.randn(dims).astype(dtype)

  positions = np.zeros([num_iter, dims], dtype)
  for i in xrange(num_iter):
    position_, momentum_ = sess.run(
        [next_momentums[0], next_position[0]],
        feed_dict={position: position_, momentum: momentum_})
    positions[i] = position_

  plt.plot(positions[:, 0]);  # Sinusoidal.
  ```

  Args:
    current_momentums: Tensor containing the value(s) of the momentum
      variable(s) to update.
    target_log_prob_fn: Python callable which takes an argument like
      `*current_state_parts` and returns its (possibly unnormalized) log-density
      under the target distribution.
    current_state_parts: Python `list` of `Tensor`s representing the current
      state(s) of the Markov chain(s). The first `independent_chain_ndims` of
      the `Tensor`(s) index different chains.
    step_sizes: Python `list` of `Tensor`s representing the step size for the
      leapfrog integrator. Must broadcast with the shape of
      `current_state_parts`.  Larger step sizes lead to faster progress, but
      too-large step sizes make rejection exponentially more likely. When
      possible, it's often helpful to match per-variable step sizes to the
      standard deviations of the target distribution in each variable.
    num_leapfrog_steps: Integer number of steps to run the leapfrog integrator
      for. Total progress per HMC step is roughly proportional to `step_size *
      num_leapfrog_steps`.
    current_target_log_prob: (Optional) `Tensor` representing the value of
      `target_log_prob_fn(*current_state_parts)`. The only reason to specify
      this argument is to reduce TF graph size.
      Default value: `None` (i.e., compute as needed).
    current_grads_target_log_prob: (Optional) Python list of `Tensor`s
      representing gradient of `target_log_prob_fn(*current_state_parts`) wrt
      `current_state_parts`. Must have same shape as `current_state_parts`. The
      only reason to specify this argument is to reduce TF graph size.
      Default value: `None` (i.e., compute as needed).
    name: Python `str` name prefixed to Ops created by this function.
      Default value: `None` (i.e., 'hmc_leapfrog_integrator').

  Returns:
    proposed_momentums: Updated value of the momentum.
    proposed_state_parts: Tensor or Python list of `Tensor`s representing the
      state(s) of the Markov chain(s) at each result step. Has same shape as
      input `current_state_parts`.
    proposed_target_log_prob: `Tensor` representing the value of
      `target_log_prob_fn` at `next_state`.
    proposed_grads_target_log_prob: Gradient of `proposed_target_log_prob` wrt
      `next_state`.

  Raises:
    ValueError: if `len(momentums) != len(state_parts)`.
    ValueError: if `len(state_parts) != len(step_sizes)`.
    ValueError: if `len(state_parts) != len(grads_target_log_prob)`.
    TypeError: if `not target_log_prob.dtype.is_floating`.
  """
  def _loop_body(step,
                 current_momentums,
                 current_state_parts,
                 ignore_current_target_log_prob,  # pylint: disable=unused-argument
                 current_grads_target_log_prob):
    return [step + 1] + list(_leapfrog_step(current_momentums,
                                            target_log_prob_fn,
                                            current_state_parts,
                                            step_sizes,
                                            current_grads_target_log_prob))

  with tf.name_scope(
      name, 'hmc_leapfrog_integrator',
      [current_momentums, current_state_parts, step_sizes, num_leapfrog_steps,
       current_target_log_prob, current_grads_target_log_prob]):
    if len(current_momentums) != len(current_state_parts):
      raise ValueError('`momentums` must be in one-to-one correspondence '
                       'with `state_parts`')
    num_leapfrog_steps = tf.convert_to_tensor(num_leapfrog_steps,
                                              name='num_leapfrog_steps')
    [
        current_target_log_prob,
        current_grads_target_log_prob,
    ] = _maybe_call_fn_and_grads(
        target_log_prob_fn,
        current_state_parts,
        current_target_log_prob,
        current_grads_target_log_prob)
    return tf.while_loop(
        cond=lambda iter_, *args: iter_ < num_leapfrog_steps,
        body=_loop_body,
        loop_vars=[
            np.int32(0),  # iter_
            current_momentums,
            current_state_parts,
            current_target_log_prob,
            current_grads_target_log_prob,
        ],
        back_prop=False)[1:]  # Lop-off 'iter_'.


def _leapfrog_step(current_momentums,
                   target_log_prob_fn,
                   current_state_parts,
                   step_sizes,
                   current_grads_target_log_prob,
                   name=None):
  """Applies one step of the leapfrog integrator."""
  with tf.name_scope(
      name, '_leapfrog_step',
      [current_momentums, current_state_parts, step_sizes,
       current_grads_target_log_prob]):
    proposed_momentums = [m + 0.5 * ss * g for m, ss, g
                          in zip(current_momentums,
                                 step_sizes,
                                 current_grads_target_log_prob)]
    proposed_state_parts = [x + ss * m for x, ss, m
                            in zip(current_state_parts,
                                   step_sizes,
                                   proposed_momentums)]
    proposed_target_log_prob = target_log_prob_fn(*proposed_state_parts)
    if not proposed_target_log_prob.dtype.is_floating:
      raise TypeError('`target_log_prob_fn` must produce a `Tensor` '
                      'with `float` `dtype`.')
    proposed_grads_target_log_prob = tf.gradients(
        proposed_target_log_prob, proposed_state_parts)
    if any(g is None for g in proposed_grads_target_log_prob):
      raise ValueError(
          'Encountered `None` gradient. Does your target `target_log_prob_fn` '
          'access all `tf.Variable`s via `tf.get_variable`?\n'
          '  current_state_parts: {}\n'
          '  proposed_state_parts: {}\n'
          '  proposed_grads_target_log_prob: {}'.format(
              current_state_parts,
              proposed_state_parts,
              proposed_grads_target_log_prob))
    proposed_momentums = [m + 0.5 * ss * g for m, ss, g
                          in zip(proposed_momentums,
                                 step_sizes,
                                 proposed_grads_target_log_prob)]
    return [
        proposed_momentums,
        proposed_state_parts,
        proposed_target_log_prob,
        proposed_grads_target_log_prob,
    ]


def _compute_energy_change(current_target_log_prob,
                           current_momentums,
                           proposed_target_log_prob,
                           proposed_momentums,
                           independent_chain_ndims,
                           name=None):
  """Helper to `kernel` which computes the energy change."""
  with tf.name_scope(
      name, 'compute_energy_change',
      [current_target_log_prob, proposed_target_log_prob,
       independent_chain_ndims, current_momentums, proposed_momentums]):
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
    return mcmc_util.safe_sum([
        -proposed_target_log_prob, proposed_kinetic,
        current_target_log_prob, -current_kinetic,
    ], alt_value=np.inf)


def _maybe_call_fn_and_grads(fn,
                             fn_arg_list,
                             fn_result=None,
                             grads_fn_result=None,
                             description='target_log_prob'):
  """Helper which computes `fn_result` and `grads` if needed."""
  fn_arg_list = (list(fn_arg_list) if mcmc_util.is_list_like(fn_arg_list)
                 else [fn_arg_list])
  if fn_result is None:
    fn_result = fn(*fn_arg_list)
  if not fn_result.dtype.is_floating:
    raise TypeError('`{}` must be a `Tensor` with `float` `dtype`.'.format(
        description))
  if grads_fn_result is None:
    grads_fn_result = tf.gradients(fn_result, fn_arg_list)
  if len(fn_arg_list) != len(grads_fn_result):
    raise ValueError('`{}` must be in one-to-one correspondence with '
                     '`grads_{}`'.format(*[description]*2))
  if any(g is None for g in grads_fn_result):
    raise ValueError('Encountered `None` gradient.')
  return fn_result, grads_fn_result


def _prepare_args(target_log_prob_fn, state, step_size,
                  target_log_prob=None, grads_target_log_prob=None,
                  maybe_expand=False, description='target_log_prob'):
  """Helper which processes input args to meet list-like assumptions."""
  state_parts = list(state) if mcmc_util.is_list_like(state) else [state]
  state_parts = [tf.convert_to_tensor(s, name='current_state')
                 for s in state_parts]
  target_log_prob, grads_target_log_prob = _maybe_call_fn_and_grads(
      target_log_prob_fn,
      state_parts,
      target_log_prob,
      grads_target_log_prob,
      description)
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
