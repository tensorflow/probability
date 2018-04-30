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
        'replica_states',
        'replica_results',
        'next_replica_idx',
        'exchange_proposed',
        'exchange_proposed_n',
        'sampled_replica_states',
        'sampled_replica_results',
    ])


def default_exchange_proposed_fn(probs):
  """Default function for `exchange_proposed_fn` of `kernel`.

  Depending on the probability of `probs`, decide whether to propose
  combinations of replica for exchange.
  When exchanging, create combinations of adjacent replicas from 0 or 1 index.

  Args:
    probs: A float-like Tensor which represents the probability of proposing
      combinations of replicas for exchange.

  Returns:
    default_exchange_proposed_fn_: Python callable which take a number of
      replicas, and return combinations of replicas for exchange and a number of
      combinations.
  """

  def default_exchange_proposed_fn_(num_replica, seed=None):
    """Default function for `exchange_proposed_fn` of `kernel`."""
    num_replica = tf.to_int32(num_replica)

    seed = distributions_util.gen_new_seed(seed, 'default_exchange_proposed_fn')
    random_uniform = tf.random_uniform([], seed=seed)
    accept_proposed_exchange = random_uniform < probs

    seed = distributions_util.gen_new_seed(seed, 'default_exchange_proposed_fn')
    zero_start = tf.random_uniform([], seed=seed) > 0.5
    if num_replica % 2 == 0:
      exchange_proposed = tf.where(
          zero_start, tf.range(num_replica),
          tf.sparse_to_dense(tf.range(num_replica - 2), (num_replica,),
                             tf.range(1, num_replica - 1)))
      exchange_proposed_n = tf.where(zero_start, num_replica // 2,
                                     num_replica // 2 - 1)
    else:
      exchange_proposed = tf.where(
          zero_start, tf.range(num_replica - 1), tf.range(1, num_replica))
      exchange_proposed_n = num_replica // 2

    exchange_proposed = tf.reshape(exchange_proposed, (num_replica // 2, 2))
    exchange_proposed = tf.where(accept_proposed_exchange, exchange_proposed,
                                 tf.zeros_like(exchange_proposed))
    exchange_proposed_n = tf.where(accept_proposed_exchange,
                                   exchange_proposed_n,
                                   tf.zeros_like(exchange_proposed_n))
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

  ##### Sampling from the Standard Normal Distribution.

  ```python
  import numpy as np
  import tensorflow as tf
  import tensorflow_probability as tfp
  tfd = tfp.distributions

  dtype = np.float32

  target = tfd.Normal(loc=dtype(0), scale=dtype(1))

  def make_kernel_fn(target_log_prob_fn, seed):
    return tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=target_log_prob_fn,
        seed=seed, step_size=1.0, num_leapfrog_steps=3)

  remc = tfp.mcmc.ReplicaExchangeMC(
      target_log_prob_fn=target.log_prob,
      inverse_temperatures=10.**tf.linspace(0., -2., 5),
      make_kernel_fn=make_kernel_fn,
      seed=42)

  samples, _ = tfp.mcmc.sample_chain(
      num_results=1000,
      current_state=dtype(1),
      kernel=remc,
      num_burnin_steps=500,
      parallel_iterations=1)  # For determinism.

  sample_mean = tf.reduce_mean(samples, axis=0)
  sample_std = tf.sqrt(
      tf.reduce_mean(tf.squared_difference(samples, sample_mean),
                     axis=0))
  with tf.Session() as sess:
    [sample_mean_, sample_std_] = sess.run([sample_mean, sample_std])

  print('Estimated mean: {}'.format(sample_mean_))
  print('Estimated standard deviation: {}'.format(sample_std_))
  ```

  ##### Sampling from a 2-D Mixture Normal Distribution.

  ```python
  import numpy as np
  import tensorflow as tf
  import tensorflow_probability as tfp
  import matplotlib.pyplot as plt
  tfd = tfp.distributions

  dtype = np.float32

  target = tfd.MixtureSameFamily(
      mixture_distribution=tfd.Categorical(probs=[0.5, 0.5]),
      components_distribution=tfd.MultivariateNormalDiag(
          loc=[[-1., -1], [1., 1.]],
          scale_identity_multiplier=[0.1, 0.1]))

  def make_kernel_fn(target_log_prob_fn, seed):
    return tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=target_log_prob_fn,
        seed=seed, step_size=0.3, num_leapfrog_steps=3)

  remc = tfp.mcmc.ReplicaExchangeMC(
      target_log_prob_fn=target.log_prob,
      inverse_temperatures=10.**tf.linspace(0., -2., 5),
      make_kernel_fn=make_kernel_fn,
      seed=42)

  samples, _ = tfp.mcmc.sample_chain(
      num_results=1000,
      current_state=np.zeros(2, dtype=dtype),
      kernel=remc,
      num_burnin_steps=500,
      parallel_iterations=1)  # For determinism.

  with tf.Session() as sess:
    samples_ = sess.run(samples)

  plt.figure(figsize=(8, 8))
  plt.xlim(-2, 2)
  plt.ylim(-2, 2)
  plt.plot(samples_[:, 0], samples_[:, 1], '.')
  plt.show()
  ```

  """

  def __init__(self, target_log_prob_fn, inverse_temperatures,
               make_kernel_fn,
               exchange_proposed_fn=default_exchange_proposed_fn(1.),
               seed=None, name=None, **kwargs):
    """Instantiates this object.

    Args:
      target_log_prob_fn: Python callable which takes an argument like
        `current_state` (or `*current_state` if it's a list) and returns its
        (possibly unnormalized) log-density under the target distribution.
      inverse_temperatures: sequence of inverse temperatures to perform
        samplings with each replica. Must have statically known `rank` and
        statically known leading shape, i.e.,
        `inverse_temperatures.shape[0].value is not None`
      make_kernel_fn: Python callable which takes target_log_prob_fn and seed
        args and returns a TransitionKernel instance.
      exchange_proposed_fn: Python callable which take a number of replicas, and
        return combinations of replicas for exchange and a number of
        combinations.
      seed: Python integer to seed the random number generator.
        Default value: `None` (i.e., no seed).
      name: Python `str` name prefixed to Ops created by this function.
        Default value: `None` (i.e., "remc_kernel").
      **kwargs: Arguments for `make_kernel_fn`.

    Raises:
      ValueError: if `inverse_temperatures` doesn't have statically known rank
        and statically known leading shape
    """
    if inverse_temperatures.shape.ndims is None or \
       inverse_temperatures.shape[0].value is None:
      raise ValueError('"inverse_temperatures" must have statically known rank '
                       'and statically known leading shape')
    self._seed_stream = seed  # This will be mutated with use.
    self._parameters = dict(target_log_prob_fn=target_log_prob_fn,
                            inverse_temperatures=inverse_temperatures,
                            num_replica=inverse_temperatures.shape[0],
                            exchange_proposed_fn=exchange_proposed_fn,
                            seed=seed, name=name)
    self.replica_kernels = []
    for i in range(self.num_replica):
      self._seed_stream = distributions_util.gen_new_seed(
          self._seed_stream, salt='replica_kernels')
      self.replica_kernels.append(make_kernel_fn(
          target_log_prob_fn=_replica_log_prob_fn(
              inverse_temperatures[i], target_log_prob_fn),
          seed=self._seed_stream))

  @property
  def target_log_prob_fn(self):
    return self._parameters['target_log_prob_fn']

  @property
  def inverse_temperatures(self):
    return self._parameters['inverse_temperatures']

  @property
  def num_replica(self):
    return self._parameters['num_replica']

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
      sampled_replica_states, sampled_replica_results = zip(*[
          rk.one_step(previous_kernel_results.replica_states[i],
                      previous_kernel_results.replica_results[i])
          for i, rk in enumerate(self.replica_kernels)])
      sampled_replica_states = list(sampled_replica_states)
      sampled_replica_results = list(sampled_replica_results)

      sampled_replica_results_modified = [
          srr._replace(target_log_prob=srr.target_log_prob /
                       self.inverse_temperatures[i])
          if 'target_log_prob' in srr._fields
          else srr._replace(accepted_results=srr.accepted_results._replace(
              target_log_prob=srr.accepted_results.target_log_prob /
              self.inverse_temperatures[i]))
          for i, srr in enumerate(sampled_replica_results)
      ]

      sampled_replica_ratios = [
          srr.target_log_prob if 'target_log_prob' in srr._fields
          else srr.accepted_results.target_log_prob
          for i, srr in enumerate(sampled_replica_results_modified)]
      sampled_replica_ratios = tf.stack(sampled_replica_ratios, axis=-1)

      next_replica_idx = tf.range(self.num_replica)
      self._seed_stream = distributions_util.gen_new_seed(
          self._seed_stream, salt='replica_exchange_one_step')
      exchange_proposed, exchange_proposed_n = self.exchange_proposed_fn(
          self.num_replica, seed=self._seed_stream)
      i = tf.constant(0)

      def cond(i, next_replica_idx):  # pylint: disable=unused-argument
        return tf.less(i, exchange_proposed_n)

      def body(i, next_replica_idx):
        """`tf.while_loop` body."""
        ratio = (
            sampled_replica_ratios[next_replica_idx[exchange_proposed[i, 0]]]
            - sampled_replica_ratios[next_replica_idx[exchange_proposed[i, 1]]])
        ratio *= (
            self.inverse_temperatures[exchange_proposed[i, 1]]
            - self.inverse_temperatures[exchange_proposed[i, 0]])
        self._seed_stream = distributions_util.gen_new_seed(
            self._seed_stream, salt='replica_exchange_one_step')
        log_uniform = tf.log(tf.random_uniform(
            shape=tf.shape(ratio),
            dtype=ratio.dtype.base_dtype,
            seed=self._seed_stream))
        exchange = log_uniform < ratio
        exchange_op = tf.sparse_to_dense(
            [exchange_proposed[i, 0], exchange_proposed[i, 1]],
            [self.num_replica],
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

      def _prep(list_):
        return list(
            tf.case({tf.equal(next_replica_idx[i], j):
                     _stateful_lambda(list_[j])
                     for j in range(self.num_replica)}, exclusive=True)
            for i in range(self.num_replica))
      next_replica_states = _prep(sampled_replica_states)
      next_replica_results = _prep(sampled_replica_results_modified)

      next_replica_results = [
          nrr._replace(target_log_prob=nrr.target_log_prob *
                       self.inverse_temperatures[i])
          if 'target_log_prob' in nrr._fields
          else nrr._replace(accepted_results=nrr.accepted_results._replace(
              target_log_prob=nrr.accepted_results.target_log_prob *
              self.inverse_temperatures[i]))
          for i, nrr in enumerate(next_replica_results)
      ]

      next_state = tf.identity(next_replica_states[0])
      kernel_results = ReplicaExchangeMCKernelResults(
          replica_states=next_replica_states,
          replica_results=next_replica_results,
          next_replica_idx=next_replica_idx,
          exchange_proposed=exchange_proposed,
          exchange_proposed_n=exchange_proposed_n,
          sampled_replica_states=sampled_replica_states,
          sampled_replica_results=sampled_replica_results,
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
                         for i in range(self.num_replica)]

      init_state_parts = (list(init_state)
                          if mcmc_util.is_list_like(init_state)
                          else [init_state])
      replica_states = [[tf.identity(s) for s in init_state_parts]
                        for i in range(self.num_replica)]

      def maybe_flatten(x):
        return x if mcmc_util.is_list_like(init_state) else x[0]
      replica_states = [maybe_flatten(s) for s in replica_states]
      next_replica_idx = tf.range(self.num_replica)
      [
          exchange_proposed,
          exchange_proposed_n,
      ] = self.exchange_proposed_fn(self.num_replica, seed=self._seed_stream)
      exchange_proposed = tf.zeros_like(exchange_proposed)
      exchange_proposed_n = tf.zeros_like(exchange_proposed_n)
      return ReplicaExchangeMCKernelResults(
          replica_states=replica_states,
          replica_results=replica_results,
          next_replica_idx=next_replica_idx,
          exchange_proposed=exchange_proposed,
          exchange_proposed_n=exchange_proposed_n,
          sampled_replica_states=replica_states,
          sampled_replica_results=replica_results,
      )


def _stateful_lambda(x):
  """Function to use instead of lambda."""
  # `lambda` is affected by the change of `x`,
  # so `_stateful_lambda(x)()`` output `x` at the time of definition.
  def _stateful_lambda_():
    return x
  return _stateful_lambda_


def _replica_log_prob_fn(inverse_temperature, target_log_prob_fn):
  """Return a log probability function made considering temperature."""
  def _replica_log_prob_fn_(*x):
    return inverse_temperature * target_log_prob_fn(*x)
  return _replica_log_prob_fn_
