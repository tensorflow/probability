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
"""Replica Exchange Monte Carlo Transition Kernel."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import tensorflow as tf

from tensorflow_probability.python.mcmc import kernel as kernel_base
from tensorflow_probability.python.mcmc import util as mcmc_util
from tensorflow.python.ops.distributions import util as distributions_util


__all__ = [
    'ReplicaExchangeMC',
    'default_exchange_proposed_fn',
]


ReplicaExchangeMCKernelResults = collections.namedtuple(
    'ReplicaExchangeMCKernelResults',
    [
        'replica_results',
        'replica_states',
        'next_replica_idx',
        'exchange_proposed',
        'exchange_proposed_n',
    ])


def default_exchange_proposed_fn(freq):
  """Default function for `exchange_proposed_fn` of `kernel`.
  Depending on the probability of `freq`, decide whether to propose combinations
  of replica for exchange.
  When exchanging, create combinations of adjacent replicas from 0 or 1 index.
  """
  def default_exchange_proposed_fn_(n_replica, seed=None):
    seed = distributions_util.gen_new_seed(seed, "default_exchange_proposed_fn")
    u = tf.random_uniform([], seed=seed)
    exist = u > freq

    seed = distributions_util.gen_new_seed(seed, "default_exchange_proposed_fn")
    i = tf.random_uniform([], minval=0, maxval=2, dtype=tf.int32, seed=seed)
    a = tf.range(i, n_replica - 1, 2)
    exchange_proposed = tf.transpose(tf.concat(([a], [a + 1]), axis=0))
    exchange_proposed_n = tf.to_int32(tf.floor(
        (tf.to_int32(n_replica) - i) / 2))

    exchange_proposed = tf.cond(
        exist, lambda: tf.to_int32([]), lambda: exchange_proposed)
    exchange_proposed_n = tf.cond(exist, lambda: 0, lambda: exchange_proposed_n)
    return exchange_proposed, exchange_proposed_n
  return default_exchange_proposed_fn_


class ReplicaExchangeMC(kernel_base.TransitionKernel):
  """Runs one step of the Replica Exchange Monte Carlo.

  [Replica Exchange Monte Carlo](
  https://en.wikipedia.org/wiki/Parallel_tempering) is a Markov chain
  Monte Carlo (MCMC) algorithm that is also known as Parallel Tempering. This
  algorithm performs multiple sampling with different temperatures in parallel,
  and exchange those samplings according to the Metropolis-Hastings criterion.
  By using the sampling result of high temperature, sampling with less influence
  of the local solution becomes possible.

  #### Examples

  ```python
  import tensorflow as tf
  import tensorflow_probability as tfp
  import numpy as np
  import matplotlib.pyplot as plt

  tfd = tf.contrib.distributions

  # Tuning acceptance rates:
  dtype = np.float32
  num_warmup_iter = 1000
  num_chain_iter = 1000

  x = tf.get_variable(name='x', initializer=np.zeros(2, dtype=dtype))

  # Target distribution is Mixture Normal.
  target = tfd.MixtureSameFamily(
      mixture_distribution=tfd.Categorical(probs=[0.5, 0.5]),
      components_distribution=tfd.MultivariateNormalDiag(
          loc=[[-5., -5], [5., 5.]],
          scale_identity_multiplier=[1., 1.]))

  # Initialize the ReplicaExchangeMC sampler.
  remc = tfp.mcmc.ReplicaExchangeMC(
      target_log_prob_fn=target.log_prob,
      inverse_temperatures=tf.pow(10., tf.linspace(0., -2., 5)),
      replica_kernel_class=tfp.mcmc.HamiltonianMonteCarlo,
      step_size=0.5,
      num_leapfrog_steps=3)

  # One iteration of the ReplicaExchangeMC
  init_results = remc.bootstrap_results(x)
  next_x, other_results = remc.one_step(
      current_state=x,
      previous_kernel_results=init_results)

  x_update = x.assign(next_x)
  replica_update = [init_results.replica_states[i].assign(
      other_results.replica_states[i]) for i in range(remc.n_replica)]

  warmup = tf.group([x_update, replica_update])

  init = tf.global_variables_initializer()

  with tf.Session() as sess:
    sess.run(init)
    # Warm up the sampler
    for _ in range(num_warmup_iter):
      sess.run(warmup)
    # Collect samples
    samples = np.zeros([num_chain_iter, 2])
    replica_samples = np.zeros([num_chain_iter, 5, 2])
    for i in range(num_chain_iter):
      _, x_, replica_x_ = sess.run([x_update, x, replica_update])
      samples[i] = x_
      replica_samples[i] = replica_x_

  plt.figure(figsize=(8, 8))
  plt.xlim(-10, 10)
  plt.ylim(-10, 10)
  plt.plot(samples[:, 0], samples[:, 1], '.')
  plt.show()
  ```

  """
  def __init__(self, target_log_prob_fn, inverse_temperatures,
               replica_kernel_class,
               exchange_proposed_fn=default_exchange_proposed_fn(1.),
               seed=None, name=None, **kwargs):
    """Instantiates this object.

    Args:
      target_log_prob_fn: Python callable which takes an argument like
        `current_state` (or `*current_state` if it's a list) and returns its
        (possibly unnormalized) log-density under the target distribution.
      inverse_temperatures: sequence of inverse temperatures to perform
        samplings with each replica.
      replica_kernel_class: mcmc kernel class for multiple sampling with
        different temperatures in parallel
      exchange_proposed_fn: Python callable which take a number of replicas, and
        return combinations of replicas for exchange.
      seed: Python integer to seed the random number generator.
      name: Python `str` name prefixed to Ops created by this function.
        Default value: `None` (i.e., "remc_kernel").
      **kwargs: Arguments for `replica_kernel_class`.
    """
    self._seed_stream = seed  # This will be mutated with use.
    self._parameters = dict(target_log_prob_fn=target_log_prob_fn,
                            inverse_temperatures=inverse_temperatures,
                            n_replica=inverse_temperatures.shape[0],
                            exchange_proposed_fn=exchange_proposed_fn,
                            seed=seed, name=name)
    self.replica_kernels = []
    for i in range(self.n_replica):
      self._seed_stream = distributions_util.gen_new_seed(
          self._seed_stream, salt='replica_kernels')
      self.replica_kernels.append(replica_kernel_class(
          target_log_prob_fn=_replica_log_prob_fn(
              inverse_temperatures[i], target_log_prob_fn),
          seed=self._seed_stream, name=name, **kwargs))

  @property
  def target_log_prob_fn(self):
    return self._parameters['target_log_prob_fn']

  @property
  def inverse_temperatures(self):
    return self._parameters['inverse_temperatures']

  @property
  def n_replica(self):
    return self._parameters['n_replica']

  @property
  def exchange_proposed_fn(self):
    return self._parameters['exchange_proposed_fn']

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
    return True

  def one_step(self, current_state, previous_kernel_results):
    """Takes one step of the TransitionKernel.

    Args:
      current_state: `Tensor` or Python `list` of `Tensor`s representing the
        current state(s) of the Markov chain(s).
      previous_kernel_results: A (possibly nested) `tuple`, `namedtuple` or
        `list` of `Tensor`s representing internal calculations made within the
        previous call to this function (or as returned by `bootstrap_results`).

    Returns:
      next_state: `Tensor` or Python `list` of `Tensor`s representing the
        next state(s) of the Markov chain(s).
      kernel_results: A (possibly nested) `tuple`, `namedtuple` or `list` of
        `Tensor`s representing internal calculations made within this function.
        This inculdes replica states.
    """
    with tf.name_scope(
        name=mcmc_util.make_name(self.name, 'remc', 'one_step'),
        values=[current_state, previous_kernel_results]):
      sampled_replica_states = []
      sampled_replica_results = []
      sampled_replica_ratios = []
      for i in range(self.n_replica):
        [
            sampled_state,
            sampled_results,
        ] = self.replica_kernels[i].one_step(
            previous_kernel_results.replica_states[i],
            previous_kernel_results.replica_results[i])
        sampled_replica_states.append(sampled_state)
        sampled_replica_results.append(sampled_results)
        sampled_replica_ratios.append(self.target_log_prob_fn(sampled_state))
      sampled_replica_ratios = tf.stack(sampled_replica_ratios, axis=0)

      next_replica_idx = tf.range(self.n_replica)
      self._seed_stream = distributions_util.gen_new_seed(
          self._seed_stream, salt='replica_exchange_one_step')
      exchange_proposed, exchange_proposed_n = \
          self.exchange_proposed_fn(self.n_replica, seed=self._seed_stream)
      i = tf.constant(0)

      def cond(i, next_replica_idx):
        return tf.less(i, exchange_proposed_n)

      def body(i, next_replica_idx):
        ratio = \
            sampled_replica_ratios[next_replica_idx[exchange_proposed[i, 0]]] \
            - sampled_replica_ratios[next_replica_idx[exchange_proposed[i, 1]]]
        ratio *= self.inverse_temperatures[exchange_proposed[i, 1]] \
            - self.inverse_temperatures[exchange_proposed[i, 0]]
        self._seed_stream = distributions_util.gen_new_seed(
            self._seed_stream, salt='replica_exchange_one_step')
        log_uniform = tf.log(tf.random_uniform(
            shape=tf.shape(ratio),
            dtype=ratio.dtype.base_dtype,
            seed=self._seed_stream))
        exchange = log_uniform < ratio
        exchange_op = tf.sparse_to_dense(
            [exchange_proposed[i, 0], exchange_proposed[i, 1]],
            [self.n_replica],
            [next_replica_idx[exchange_proposed[i, 1]] -
             next_replica_idx[exchange_proposed[i, 0]],
             next_replica_idx[exchange_proposed[i, 0]] -
             next_replica_idx[exchange_proposed[i, 1]]])
        next_replica_idx = tf.cond(exchange,
                                   lambda: next_replica_idx + exchange_op,
                                   lambda: next_replica_idx)
        return [i + 1, next_replica_idx]

      next_replica_idx = tf.while_loop(
          cond, body, loop_vars=[i, next_replica_idx])[1]

      next_replica_states = []
      next_replica_results = []
      for i in range(self.n_replica):
        next_replica_states.append(
            tf.case({tf.equal(next_replica_idx[i], j):
                    _stateful_lambda(sampled_replica_states[j])
                    for j in range(self.n_replica)}, exclusive=True))
        next_replica_results.append(
            tf.case({tf.equal(next_replica_idx[i], j):
                     _stateful_lambda(sampled_replica_results[j])
                     for j in range(self.n_replica)}, exclusive=True))

      next_state = tf.identity(next_replica_states[0])
      kernel_results = ReplicaExchangeMCKernelResults(
          replica_results=next_replica_results,
          replica_states=next_replica_states,
          next_replica_idx=next_replica_idx,
          exchange_proposed=exchange_proposed,
          exchange_proposed_n=exchange_proposed_n,
      )

      return next_state, kernel_results

  def bootstrap_results(self, init_state):
    """Returns an object with the same type as returned by `one_step`.

    Args:
      init_state: `Tensor` or Python `list` of `Tensor`s representing the
        a state(s) of the Markov chain(s).

    Returns:
      kernel_results: A (possibly nested) `tuple`, `namedtuple` or `list` of
        `Tensor`s representing internal calculations made within this function.
        This inculdes replica states.
    """
    with tf.name_scope(
        name=mcmc_util.make_name(self.name, 'remc', 'bootstrap_results'),
        values=[init_state]):
      replica_results = [self.replica_kernels[i].bootstrap_results(init_state)
                         for i in range(self.n_replica)]
      replica_states = [tf.Variable(init_state) for i in range(self.n_replica)]
      next_replica_idx = tf.range(self.n_replica)
      exchange_proposed = tf.to_int32([])
      exchange_proposed_n = 0
      return ReplicaExchangeMCKernelResults(
          replica_results=replica_results,
          replica_states=replica_states,
          next_replica_idx=next_replica_idx,
          exchange_proposed=exchange_proposed,
          exchange_proposed_n=exchange_proposed_n,
      )


def has_target_log_prob(kernel_results):
  """Returns `True` if `target_log_prob` is a member of input."""
  return getattr(kernel_results, 'target_log_prob', None) is not None


def _stateful_lambda(x):
  """Function to use instead of lambda.
  `lambda` is affected by the change of `x`,
  so `_stateful_lambda(x)()`` output `x` at the time of definition.
  """
  def _stateful_lambda_():
    return x
  return _stateful_lambda_


def _replica_log_prob_fn(inverse_temperature, target_log_prob_fn):
  """Return a log probability function made considering temperature.
  """
  def _replica_log_prob_fn_(*x):
    return inverse_temperature * target_log_prob_fn(*x)
  return _replica_log_prob_fn_
