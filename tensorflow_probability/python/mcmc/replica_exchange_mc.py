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

from tensorflow_probability.python import distributions
from tensorflow_probability.python.mcmc import kernel as kernel_base
from tensorflow_probability.python.mcmc.internal import util as mcmc_util

__all__ = [
    'ReplicaExchangeMC',
    'default_exchange_proposed_fn',
]

ReplicaExchangeMCKernelResults = collections.namedtuple(
    'ReplicaExchangeMCKernelResults',
    [
        # List of states for each replica.  Each state may itself be a list of
        # state parts.
        'replica_states',
        # List of KernelResults for each replica, post exchange.
        'replica_results',
        # List of state/state-parts with pre-exchange samples from each replica.
        'sampled_replica_states',
        # List of kernel-results with pre-exchange samples from each replica.
        'sampled_replica_results',
    ])


def default_exchange_proposed_fn(prob_exchange):
  """Default exchange proposal function, for replica exchange MC.

  With probability `prob_exchange` propose combinations of replica for exchange.
  When exchanging, create combinations of adjacent replicas in
  [Replica Exchange Monte Carlo](
  https://en.wikipedia.org/wiki/Parallel_tempering)

  ```
  exchange_fn = default_exchange_proposed_fn(prob_exchange=0.5)
  exchange_proposed = exchange_fn(num_replica=3)

  exchange_proposed.eval()
  ==> [[0, 1]]  # 1 exchange, 0 <--> 1

  exchange_proposed.eval()
  ==> []  # 0 exchanges
  ```

  Args:
    prob_exchange: Scalar `Tensor` giving probability that any exchanges will
      be generated.

  Returns:
    default_exchange_proposed_fn_: Python callable which take a number of
      replicas (a Python integer), and return combinations of replicas for
      exchange as an [n, 2] integer `Tensor`, `0 <= n <= num_replica // 2`,
      with *unique* values in the set `{0, ..., num_replica}`.
  """

  def default_exchange_proposed_fn_(num_replica, seed=None):
    """Default function for `exchange_proposed_fn` of `kernel`."""
    seed_stream = distributions.SeedStream(seed, 'default_exchange_proposed_fn')

    zero_start = tf.random.uniform([], seed=seed_stream()) > 0.5
    if num_replica % 2 == 0:

      def _exchange():
        flat_exchange = tf.range(num_replica)
        if num_replica > 2:
          start = tf.cast(~zero_start, dtype=tf.int32)
          end = num_replica - start
          flat_exchange = flat_exchange[start:end]
        return tf.reshape(flat_exchange, [tf.size(input=flat_exchange) // 2, 2])
    else:

      def _exchange():
        start = tf.cast(zero_start, dtype=tf.int32)
        end = num_replica - tf.cast(~zero_start, dtype=tf.int32)
        flat_exchange = tf.range(num_replica)[start:end]
        return tf.reshape(flat_exchange, [tf.size(input=flat_exchange) // 2, 2])

    def _null_exchange():
      return tf.reshape(tf.cast([], dtype=tf.int32), shape=[0, 2])

    return tf.cond(
        pred=tf.random.uniform([], seed=seed_stream()) < prob_exchange,
        true_fn=_exchange,
        false_fn=_null_exchange)

  return default_exchange_proposed_fn_


class ReplicaExchangeMC(kernel_base.TransitionKernel):
  """Runs one step of the Replica Exchange Monte Carlo.

  [Replica Exchange Monte Carlo](
  https://en.wikipedia.org/wiki/Parallel_tempering) is a Markov chain
  Monte Carlo (MCMC) algorithm that is also known as Parallel Tempering. This
  algorithm performs multiple sampling with different temperatures in parallel,
  and exchanges those samplings according to the Metropolis-Hastings criterion.

  The `K` replicas are parameterized in terms of `inverse_temperature`'s,
  `(beta[0], beta[1], ..., beta[K-1])`.  If the target distribution has
  probability density `p(x)`, the `kth` replica has density `p(x)**beta_k`.

  Typically `beta[0] = 1.0`, and `1.0 > beta[1] > beta[2] > ... > 0.0`.

  * `beta[0] == 1` ==> First replicas samples from the target density, `p`.
  * `beta[k] < 1`, for `k = 1, ..., K-1` ==> Other replicas sample from
    "flattened" versions of `p` (peak is less high, valley less low).  These
    distributions are somewhat closer to a uniform on the support of `p`.

  Samples from adjacent replicas `i`, `i + 1` are used as proposals for each
  other in a Metropolis step.  This allows the lower `beta` samples, which
  explore less dense areas of `p`, to occasionally be used to help the
  `beta == 1` chain explore new regions of the support.

  Samples from replica 0 are returned, and the others are discarded.

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
      inverse_temperatures=[1., 0.3, 0.1, 0.03],
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
      inverse_temperatures=[1., 0.3, 0.1, 0.03, 0.01],
      make_kernel_fn=make_kernel_fn,
      seed=42)

  samples, _ = tfp.mcmc.sample_chain(
      num_results=1000,
      # Start near the [1, 1] mode.  Standard HMC would get stuck there.
      current_state=np.ones(2, dtype=dtype),
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

  def __init__(self,
               target_log_prob_fn,
               inverse_temperatures,
               make_kernel_fn,
               exchange_proposed_fn=default_exchange_proposed_fn(1.),
               seed=None,
               name=None):
    """Instantiates this object.

    Args:
      target_log_prob_fn: Python callable which takes an argument like
        `current_state` (or `*current_state` if it's a list) and returns its
        (possibly unnormalized) log-density under the target distribution.
      inverse_temperatures: `1D` `Tensor of inverse temperatures to perform
        samplings with each replica. Must have statically known `shape`.
        `inverse_temperatures[0]` produces the states returned by samplers,
        and is typically == 1.
      make_kernel_fn: Python callable which takes target_log_prob_fn and seed
        args and returns a TransitionKernel instance.
      exchange_proposed_fn: Python callable which take a number of replicas, and
        return combinations of replicas for exchange.
      seed: Python integer to seed the random number generator.
        Default value: `None` (i.e., no seed).
      name: Python `str` name prefixed to Ops created by this function.
        Default value: `None` (i.e., "remc_kernel").

    Raises:
      ValueError: `inverse_temperatures` doesn't have statically known 1D shape.
    """
    inverse_temperatures = tf.convert_to_tensor(
        value=inverse_temperatures, name='inverse_temperatures')

    # Note these are static checks, and don't need to be embedded in the graph.
    inverse_temperatures.shape.assert_is_fully_defined()
    inverse_temperatures.shape.assert_has_rank(1)

    self._seed_stream = distributions.SeedStream(seed, salt=name)
    self._seeded_mcmc = seed is not None
    self._parameters = dict(
        target_log_prob_fn=target_log_prob_fn,
        inverse_temperatures=inverse_temperatures,
        num_replica=tf.compat.dimension_value(inverse_temperatures.shape[0]),
        exchange_proposed_fn=exchange_proposed_fn,
        seed=seed,
        name=name)
    self.replica_kernels = []
    for i in range(self.num_replica):
      self.replica_kernels.append(
          make_kernel_fn(
              target_log_prob_fn=_replica_log_prob_fn(inverse_temperatures[i],
                                                      target_log_prob_fn),
              seed=self._seed_stream()))

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
    # Key difficulty:  The type of exchanges differs from one call to the
    # next...even the number of exchanges can differ.
    # As a result, exchanges must happen dynamically, in while loops.
    with tf.compat.v1.name_scope(
        name=mcmc_util.make_name(self.name, 'remc', 'one_step'),
        values=[current_state, previous_kernel_results]):

      # Each replica does `one_step` to get pre-exchange states/KernelResults.
      sampled_replica_states, sampled_replica_results = zip(*[
          rk.one_step(previous_kernel_results.replica_states[i],
                      previous_kernel_results.replica_results[i])
          for i, rk in enumerate(self.replica_kernels)
      ])
      sampled_replica_states = list(sampled_replica_states)
      sampled_replica_results = list(sampled_replica_results)

      states_are_lists = mcmc_util.is_list_like(sampled_replica_states[0])

      if not states_are_lists:
        sampled_replica_states = [[s] for s in sampled_replica_states]
      num_state_parts = len(sampled_replica_states[0])

      dtype = sampled_replica_states[0][0].dtype

      # Must put states into TensorArrays.  Why?  We will read/write states
      # dynamically with Tensor index `i`, and you cannot do this with lists.
      # old_states[k][i] is Tensor of (old) state part k, for replica i.
      # The `k` will be known statically, and `i` is a Tensor.
      old_states = [
          tf.TensorArray(
              dtype,
              size=self.num_replica,
              dynamic_size=False,
              clear_after_read=False,
              tensor_array_name='old_states',
              # State part k has same shape, regardless of replica.  So use 0.
              element_shape=sampled_replica_states[0][k].shape)
          for k in range(num_state_parts)
      ]
      for k in range(num_state_parts):
        for i in range(self.num_replica):
          old_states[k] = old_states[k].write(i, sampled_replica_states[i][k])

      exchange_proposed = self.exchange_proposed_fn(
          self.num_replica, seed=self._seed_stream())
      exchange_proposed_n = tf.shape(input=exchange_proposed)[0]

      exchanged_states = self._get_exchanged_states(
          old_states, exchange_proposed, exchange_proposed_n,
          sampled_replica_states, sampled_replica_results)

      no_exchange_proposed, _ = tf.compat.v1.setdiff1d(
          tf.range(self.num_replica), tf.reshape(exchange_proposed, [-1]))

      exchanged_states = self._insert_old_states_where_no_exchange_was_proposed(
          no_exchange_proposed, old_states, exchanged_states)

      next_replica_states = []
      for i in range(self.num_replica):
        next_replica_states_i = []
        for k in range(num_state_parts):
          next_replica_states_i.append(exchanged_states[k].read(i))
        next_replica_states.append(next_replica_states_i)

      if not states_are_lists:
        next_replica_states = [s[0] for s in next_replica_states]
        sampled_replica_states = [s[0] for s in sampled_replica_states]

      # Now that states are/aren't exchanged, bootstrap next kernel_results.
      # The viewpoint is that after each exchange, we are starting anew.
      next_replica_results = [
          rk.bootstrap_results(state)
          for rk, state in zip(self.replica_kernels, next_replica_states)
      ]

      next_state = next_replica_states[0]  # Replica 0 is the returned state(s).

      kernel_results = ReplicaExchangeMCKernelResults(
          replica_states=next_replica_states,
          replica_results=next_replica_results,
          sampled_replica_states=sampled_replica_states,
          sampled_replica_results=sampled_replica_results,
      )

      return next_state, kernel_results

  def _get_exchanged_states(self, old_states, exchange_proposed,
                            exchange_proposed_n, sampled_replica_states,
                            sampled_replica_results):
    """Get list of TensorArrays holding exchanged states, and zeros."""
    with tf.compat.v1.name_scope('get_exchanged_states'):

      target_log_probs = []
      for replica in range(self.num_replica):
        replica_log_prob = _get_field(sampled_replica_results[replica],
                                      'target_log_prob')
        inverse_temp = self.inverse_temperatures[replica]
        target_log_probs.append(replica_log_prob / inverse_temp)
      target_log_probs = tf.stack(target_log_probs, axis=0)

      dtype = target_log_probs.dtype
      num_state_parts = len(sampled_replica_states[0])
      # exchanged_states[k][i] is Tensor of (new) state part k, for replica i.
      # The `k` will be known statically, and `i` is a Tensor.
      # We will insert values into indices `i` for every replica with a proposed
      # exchange.
      exchanged_states = [
          tf.TensorArray(
              dtype,
              size=self.num_replica,
              dynamic_size=False,
              tensor_array_name='exchanged_states',
              # State part k has same shape, regardless of replica.  So use 0.
              element_shape=sampled_replica_states[0][k].shape)
          for k in range(num_state_parts)
      ]

      # Draw random variables here, to avoid sampling in the loop (and losing
      # reproducibility).  This may mean we sample too many, but we will always
      # have enough.
      sample_shape = tf.concat(
          ([self.num_replica // 2], tf.shape(input=target_log_probs)[1:]),
          axis=0)
      log_uniforms = tf.math.log(
          tf.random.uniform(
              shape=sample_shape, dtype=dtype, seed=self._seed_stream()))

      def _swap(is_exchange_accepted, x, y):
        """Swap batches of x, y where accepted."""
        with tf.compat.v1.name_scope('swap_where_exchange_accepted'):
          new_x = mcmc_util.choose(is_exchange_accepted, y, x)
          new_y = mcmc_util.choose(is_exchange_accepted, x, y)
        return new_x, new_y

      def cond(i, unused_exchanged_states):
        return i < exchange_proposed_n

      def body(i, exchanged_states):
        """Body of while loop for exchanging states."""
        # Propose exchange between replicas indexed by m and n.
        m, n = tf.unstack(exchange_proposed[i])

        # Construct log_accept_ratio:  -temp_diff * target_log_prob_diff.
        # Note target_log_prob_diff = -EnergyDiff (common definition is in terms
        # of energy).
        temp_diff = self.inverse_temperatures[m] - self.inverse_temperatures[n]
        # Difference of target log probs may be +- Inf or NaN.  We want the
        # product of this with the temperature difference to have "alt value" of
        # -Inf.
        log_accept_ratio = mcmc_util.safe_sum(
            [-temp_diff * target_log_probs[m], temp_diff * target_log_probs[n]])

        is_exchange_accepted = log_uniforms[i] < log_accept_ratio

        for k in range(num_state_parts):
          new_m, new_n = _swap(is_exchange_accepted, old_states[k].read(m),
                               old_states[k].read(n))
          exchanged_states[k] = exchanged_states[k].write(m, new_m)
          exchanged_states[k] = exchanged_states[k].write(n, new_n)

        return i + 1, exchanged_states

      # At this point, exchanged_states[k] is a length num_replicas TensorArray.
      return tf.while_loop(
          cond=cond, body=body, loop_vars=[tf.constant(0),
                                           exchanged_states])[1]  # Remove `i`

  def _insert_old_states_where_no_exchange_was_proposed(
      self, no_exchange_proposed, old_states, exchanged_states):
    with tf.compat.v1.name_scope(
        'insert_old_states_where_no_exchange_was_proposed'):

      def cond(j, unused_exchanged_states):
        return j < tf.size(input=no_exchange_proposed)

      def body(j, exchanged_states):
        replica = no_exchange_proposed[j]
        for k in range(len(old_states)):  # k indexes state part
          exchanged_states[k] = exchanged_states[k].write(
              replica, old_states[k].read(replica))
        return j + 1, exchanged_states

      return tf.while_loop(
          cond=cond, body=body, loop_vars=[tf.constant(0),
                                           exchanged_states])[1]  # Remove `j`

  def bootstrap_results(self, init_state):
    """Returns an object with the same type as returned by `one_step`.

    Args:
      init_state: `Tensor` or Python `list` of `Tensor`s representing the
        initial state(s) of the Markov chain(s).

    Returns:
      kernel_results: A (possibly nested) `tuple`, `namedtuple` or `list` of
        `Tensor`s representing internal calculations made within this function.
        This inculdes replica states.
    """
    with tf.compat.v1.name_scope(
        name=mcmc_util.make_name(self.name, 'remc', 'bootstrap_results'),
        values=[init_state]):
      replica_results = [
          self.replica_kernels[i].bootstrap_results(init_state)
          for i in range(self.num_replica)
      ]

      init_state_parts = (
          list(init_state)
          if mcmc_util.is_list_like(init_state) else [init_state])

      # Convert all states parts to tensor...
      replica_states = [[
          tf.convert_to_tensor(value=s) for s in init_state_parts
      ] for i in range(self.num_replica)]

      if not mcmc_util.is_list_like(init_state):
        replica_states = [s[0] for s in replica_states]

      return ReplicaExchangeMCKernelResults(
          replica_states=replica_states,
          replica_results=replica_results,
          sampled_replica_states=replica_states,
          sampled_replica_results=replica_results,
      )


def _replica_log_prob_fn(inverse_temperature, target_log_prob_fn):
  """Return a log probability function made considering temperature."""

  def _replica_log_prob_fn_(*x):
    return inverse_temperature * target_log_prob_fn(*x)

  return _replica_log_prob_fn_


# TODO(b/111801087) Use a more standardized API when available.
def _get_field(kernel_results, field_name):
  """field_name from kernel_results or kernel_results.accepted_results."""
  if hasattr(kernel_results, field_name):
    return getattr(kernel_results, field_name)
  if hasattr(kernel_results, 'accepted_results'):
    return getattr(kernel_results.accepted_results, field_name)
  raise TypeError('Cannot extract %s from %s' % (field_name, kernel_results))
