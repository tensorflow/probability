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
import warnings
import numpy as np

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.internal import unnest
from tensorflow_probability.python.mcmc import kernel as kernel_base
from tensorflow_probability.python.mcmc.internal import util as mcmc_util


__all__ = [
    'ReplicaExchangeMC',
    'default_swap_proposal_fn',
    'even_odd_swap_proposal_fn',
]


# Cause all warnings to always be triggered.
# Not having this means subsequent calls wont trigger the warning.
warnings.filterwarnings('always',
                        module='tensorflow_probability.*replica_exchange_mc',
                        append=True)  # Don't override user-set filters.


class ReplicaExchangeMCKernelResults(
    mcmc_util.PrettyNamedTupleMixin,
    collections.namedtuple(
        # All tensors `x` with shape [num_replica, ...] are "ordered", meaning
        # x[k,...] holds values for replica `k`.
        'ReplicaExchangeMCKernelResults',
        [
            # List-like of [num_replica] + batch_shape Tensor (or list thereof)
            # holding state parts for all replicas, after swaps.
            # This will be state parts, *even* if the chain is working with
            # states.
            'post_swap_replica_states',

            # Kernel results for replicas, before any swaps.
            'pre_swap_replica_results',

            # Kernel results for replicas, after swaps.
            # Some fields are updated, and some removed!
            # The theme is to update whatever is necessary to to obtain correct
            # state swaps, and remove fields that are ambiguous (e.g.
            # proposed_results inside Metropolis KR).
            'post_swap_replica_results',

            # Shape [num_replica, num_replica] + batch_shape boolean Tensor
            # where is_swap_proposed[i, j, ...] == True indicates a swap between
            # replicas `i` and `j` has been proposed.
            # Note is_swap_proposed[i, i, ...] == True indicates no move is
            # proposed for replica `i`.
            # TODO(b/144166689) Consider whether it may be better to make the
            # user compute this post-sampling rather than in kernel results.
            'is_swap_proposed',

            # Similar to is_swap_proposed.
            # is_swap_accepted[i, j, ...] indicates a swap between replicas
            # `i` and `j` was accepted.
            'is_swap_accepted',

            # Shape [num_replica - 1] + batch_shape boolean vectors equal to the
            # first lower sub-diagonal of is_swap_proposed.
            # is_swap_proposed_adjacent[k, ...] == True just when an swap is
            # proposed between replicas `k` and `k+1`.
            # This is sufficient to track swaps in the common (and default)
            # case where swaps are between adjacent replicas only.
            'is_swap_proposed_adjacent',
            'is_swap_accepted_adjacent',

            # The inverse_temperatures used to calculate these results. (Other
            # TransitionKernels which want to intercept the inverse_temperatures
            # should rewrite this field.)  Shape [num_replica].
            'inverse_temperatures',

            # Shape [num_replica] + batch_shape permutation used to propose
            # swaps.
            'swaps',

            # Random seed for this step.
            'seed',

            # Count of how many steps have been taken. May be used to determine
            # swaps.
            'step_count',
        ])):
  """Internal state and diagnostics for Replica Exchange MC."""
  __slots__ = ()


def default_swap_proposal_fn(prob_swap, name=None):
  """Make the default swap proposal func, with `P[swap]`, for replica swap MC.

  With probability `prob_swap`, propose combinations of replicas to swap
  When exchanging, create combinations of adjacent replicas in
  [Replica Exchange Monte Carlo](
  https://en.wikipedia.org/wiki/Parallel_tempering).  See also review paper [1].

  ```
  swap_fn = default_swap_proposal_fn(prob_swap=0.5)

  swap_fn(num_replica=3)
  ==> [1, 0, 2]  # 1 swap, 0 <--> 1

  swap_fn(num_replica=3)
  ==> [0, 1, 2]  # 0 swaps

  swap_fn(num_replica=3, batch_shape=[2])
  ==> [[0, 1],
       [2, 0],
       [1, 2]]
  ```

  Args:
    prob_swap: Scalar `Tensor` in `[0, 1]` giving probability that any swaps
      will be generated.
    name: Python `str` name given to ops created by this function.
      Default value: `'adjacent_swaps'`.

  Returns:
    default_swap_proposal_fn_: Python callable which take a number of
      replicas (a Python integer), and integer `Tensor` `batch_shape`, an
      unused `step_count`, a `seed`, and returns `swaps`, a shape
      `[num_replica] + batch_shape` `Tensor`, where axis 0 indexes
      "one-time swaps", i.e., such that (if `rank(swaps) == 1`,
      `range(num_replicas) == tf.gather(swaps, swaps)`.

  #### References

  [1]: David J. Earl, Michael W. Deem
       Parallel Tempering: Theory, Applications, and New Perspectives
       https://arxiv.org/abs/physics/0508111
  """
  def adjacent_swaps(num_replica, batch_shape=(), step_count=None, seed=None):
    """Make random shuffle using only one time swaps."""
    del step_count  # Unused for this function.
    with tf.name_scope(name or 'adjacent_swaps'):
      parity_seed, proposal_seed = samplers.split_seed(seed)
      # u selects parity.  E.g.,
      #  u==False ==> [1, 0, 3, 2, 4] even parity swaps
      #  u==True ==>  [0, 2, 1, 4, 3] odd parity swaps
      # If there are only 2 replicas, then the "True" swaps are null
      # swaps...which would contradict the user provided `prob_swap`.
      # So special case num_replica==2, forcing u==False in this case.
      u_shape = ps.concat((
          ps.ones(1, dtype=tf.int32), ps.cast(batch_shape, tf.int32)), axis=0)
      u = samplers.uniform(u_shape, seed=parity_seed) < 0.5
      u = tf.where(num_replica > 2, u, False)

      x = mcmc_util.left_justified_expand_dims_to(
          ps.range(num_replica, dtype=tf.int64),
          rank=ps.size(u_shape))
      y = tf.where(tf.equal(x % 2, tf.cast(u, dtype=tf.int64)), x + 1, x - 1)
      y = tf.clip_by_value(y, 0, num_replica - 1)
      # TODO(b/142689785): Consider using tf.cond and returning an empty list
      # then in REMC consider using a tf.cond for short-circuiting.
      return tf.where(
          samplers.uniform(batch_shape, seed=proposal_seed) < prob_swap, y, x)

  return adjacent_swaps


def even_odd_swap_proposal_fn(swap_frequency, name=None):
  """Make a deterministic swap proposal function, alternating even/odd swaps.

  This proposal function swaps deterministically `swap_frequency` fraction of
  the time, alternating even and odd parity.
  This was shown in [2] to mix better than random schemes.

  Contrast this with `default_swap_proposal_fn`, which swaps randomly with
  probability `prob_swap`.

  ```
  swap_fn = even_odd_swap_proposal_fn(swap_frequency=1)

  even_odd_swap_proposal_fn(num_replica=4, step_count=0)
  ==> [1, 0, 3, 2]  # Swap 0 <--> 1 and 2 <--> 3, even parity.

  even_odd_swap_proposal_fn(num_replica=4, step_count=1)
  ==> [0, 2, 1, 3]  # Swap 1 <--> 2, odd parity.
  ```

  Args:
    swap_frequency: Scalar `Tensor` in `[0, 1]` giving the frequency of swaps.
      Swaps will occur, with alternating parity, every `N` steps, where
      `N = 1 / swap_frequency`.
    name: Python `str` name given to ops created by this function.
      Default value: `'even_odd_swaps'`.

  Returns:
    default_swap_proposal_fn_: Python callable which take a number of
      replicas (a Python integer), and integer `Tensor` `batch_shape`, a
      `step_count`, a `seed`, and returns `swaps`, a shape
      `[num_replica] + batch_shape` `Tensor`, where axis 0 indexes
      "one-time swaps", i.e., such that (if `rank(swaps) == 1`,
      `range(num_replicas) == tf.gather(swaps, swaps)`.

  #### References

  [1]: S. Syed, A. Bouchard-Cote G. Deligiannidis, A. Doucet
       Non-Reversible Parallel Tempering: a Scalable Highly Parallel MCMC Scheme
       https://arxiv.org/abs/1905.02939
  """

  def even_odd_swaps(num_replica, batch_shape=(), step_count=None, seed=None):
    """Make deterministic even_odd one time swaps."""
    if step_count is None:
      raise ValueError('`step_count` must be supplied. Found `None`.')
    del seed  # Unused for this function.
    with tf.name_scope(name or 'even_odd_swaps'):
      # Period is 1 / frequency, and we want period = Inf if frequency = 0.
      # safe_swap_period is the correct swap period in case swap_frequency > 0.
      # If swap_frequency == 0, safe_swap_period is set to 1 (to avoid integer
      # div by zero below). We will hard-set this case to "null swap."
      swap_freq = tf.convert_to_tensor(swap_frequency, name='swap_frequency')
      safe_swap_period = tf.cast(
          tf.where(swap_freq > 0,
                   tf.math.ceil(tf.math.reciprocal_no_nan(swap_freq)), 1),
          # Although period = 1 / frequency may have roundoff error, and result
          # in a period different than what the user intended, the
          # user will end up with a single integer period, and thus well defined
          # deterministic swaps.
          tf.int32,
      )

      # u selects parity.  E.g.,
      #  u==False ==> [1, 0, 3, 2, 4] even parity swaps
      #  u==True ==>  [0, 2, 1, 4, 3] odd parity swaps
      # If there are 2 replicas, then the "True" swaps are null
      # swaps...which would contradict the user provided `swap_frequency`.
      # So special case num_replica==2, forcing u==False in this case.
      u_shape = ps.concat((
          ps.ones(1, dtype=tf.int32), ps.cast(batch_shape, tf.int32)), axis=0)
      u = tf.fill(u_shape, tf.cast((step_count // safe_swap_period) % 2,
                                   tf.bool))
      u = tf.where(num_replica > 2, u, False)

      x = mcmc_util.left_justified_expand_dims_to(
          tf.range(num_replica, dtype=tf.int64),
          rank=ps.size(u_shape))
      y = tf.where(tf.equal(x % 2, tf.cast(u, dtype=tf.int64)), x + 1, x - 1)
      y = tf.clip_by_value(y, 0, num_replica - 1)
      # TODO(b/142689785): Consider using tf.cond and returning an empty list
      # then in REMC consider using a tf.cond for short-circuiting.
      return tf.where(
          (tf.cast(step_count % safe_swap_period, tf.bool) |
           tf.math.equal(swap_freq, 0)),
          x,  # Don't swap
          y,  # Swap
      )

  return even_odd_swaps


class ReplicaExchangeMC(kernel_base.TransitionKernel):
  """Runs one step of the Replica Exchange Monte Carlo.

  [Replica Exchange Monte Carlo](
  https://en.wikipedia.org/wiki/Parallel_tempering) is a Markov chain
  Monte Carlo (MCMC) algorithm that is also known as Parallel Tempering. This
  algorithm takes multiple samples (from tempered distributions) in parallel,
  then swaps these samples according to the Metropolis-Hastings criterion.
  See also the review paper [1].

  The `K` replicas are parameterized in terms of `inverse_temperature`'s,
  `(beta[0], beta[1], ..., beta[K-1])`. If the user provides
  `target_log_prob_fn`, then the `kth` replica samples from density `p_k(x)`,
  with `log(p_k(x)) = beta_k * target_log_prob(x)`.
  In this case, geometrically decaying `beta` often works well.  That is, with
  `R < 1`, we recommend trying `beta[k] = R^k` so that
  `1.0 = beta[0] > beta[1] > ... > 0`.  See [2].

  The user can also provide two functions, `tempered_log_prob_fn` and
  `untempered_log_prob_fn`. In this case, the `kth` replica samples from
  density `p_k(x)` with
  `log(p_k(x)) = beta_k * tempered_log_prob_fn(x) + untempered_log_prob_fn(x)`.
  The this case, `beta` may be zero, and one often sets `beta[-1]` to zero.
  This means the last replica samples using `untempered_log_prob_fn`.
  In the Bayesian setup, `untempered_log_prob_fn` will often be the log prior,
  and `tempered_log_prob_fn` the likelihood.

  In all cases,

  * `beta[0] == 1` ==> First replica samples from the target density.
  * `beta[k] < 1`, for `k = 1, ..., K-1` ==> Other replicas sample from
    "tempered" versions of target (peak is less high, valley less low).  These
    distributions should allow easier exploration of separated modes.

  By default, samples from adjacent replicas `i`, `i + 1` are used as proposals
  for each other in a Metropolis step.  This allows the lower `beta` samples,
  which explore less dense areas of `p`, to eventually swap state with the
  `beta == 1` chain, allowing it to explore these new regions.

  Samples from replica 0 are returned, and the others are discarded, unless
  `state_includes_replicas`.

  #### Examples

  ##### Sampling from the Standard Normal Distribution.

  ```python
  import numpy as np
  import tensorflow as tf
  import tensorflow_probability as tfp
  tfd = tfp.distributions

  dtype = np.float32

  target = tfd.Normal(loc=dtype(0), scale=dtype(1))

  # Geometric decay is a good rule of thumb.
  inverse_temperatures = 0.5**tf.range(4, dtype=dtype)

  # If everything was Normal, step_size should be ~ sqrt(temperature).
  step_size = 1.5 / tf.sqrt(inverse_temperatures)

  def make_kernel_fn(target_log_prob_fn):
    return tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=target_log_prob_fn,
        step_size=step_size, num_leapfrog_steps=3)

  remc = tfp.mcmc.ReplicaExchangeMC(
      target_log_prob_fn=target.log_prob,
      inverse_temperatures=inverse_temperatures,
      make_kernel_fn=make_kernel_fn)

  def trace_swaps(unused_state, results):
    return (results.is_swap_proposed_adjacent,
            results.is_swap_accepted_adjacent)

  samples, (is_swap_proposed_adjacent, is_swap_accepted_adjacent) = (
      tfp.mcmc.sample_chain(
          num_results=1000,
          current_state=1.0,
          kernel=remc,
          num_burnin_steps=500,
          trace_fn=trace_swaps)
  )

  # conditional_swap_prob[k] = P[ExchangeAccepted | ExchangeProposed],
  # for the swap between replicas k and k+1.
  conditional_swap_prob = (
      tf.reduce_sum(tf.cast(is_swap_accepted_adjacent, tf.float32), axis=0)
      /
      tf.reduce_sum(tf.cast(is_swap_proposed_adjacent, tf.float32), axis=0))
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

  inverse_temperatures = 0.2**tf.range(4, dtype=dtype)

  # step_size must broadcast with all batch and event dimensions of target.
  # Here, this means it must broadcast with:
  #  [len(inverse_temperatures)] + target.event_shape
  step_size = 0.075 / tf.reshape(tf.sqrt(inverse_temperatures), shape=(4, 1))

  def make_kernel_fn(target_log_prob_fn):
    return tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=target_log_prob_fn,
        step_size=step_size, num_leapfrog_steps=3)

  remc = tfp.mcmc.ReplicaExchangeMC(
      target_log_prob_fn=target.log_prob,
      inverse_temperatures=inverse_temperatures,
      make_kernel_fn=make_kernel_fn)

  samples = tfp.mcmc.sample_chain(
      num_results=1000,
      # Start near the [1, 1] mode. Standard HMC would get stuck there.
      current_state=tf.ones(2, dtype=dtype),
      kernel=remc,
      trace_fn=None,
      num_burnin_steps=500)

  plt.figure(figsize=(8, 8))
  plt.xlim(-2, 2)
  plt.ylim(-2, 2)
  plt.plot(samples[:, 0], samples[:, 1], '.')
  plt.show()
  ```

  #### References

  [1]: David J. Earl, Michael W. Deem
       Parallel Tempering: Theory, Applications, and New Perspectives
       https://arxiv.org/abs/physics/0508111
  [2]: David A. Kofke
       On the acceptance probability of replica-exchange Monte Carlo trials.
       J. of Chem. Phys. Vol. 117 No. 5.
  """

  def __init__(self,
               target_log_prob_fn,
               inverse_temperatures,
               make_kernel_fn,
               swap_proposal_fn=default_swap_proposal_fn(1.),
               state_includes_replicas=False,
               untempered_log_prob_fn=None,
               tempered_log_prob_fn=None,
               validate_args=False,
               name=None):
    """Instantiates this object.

    Args:
      target_log_prob_fn: Python callable which takes an argument like
        `current_state` (or `*current_state` if it's a list) and returns its
        (possibly unnormalized) log-density under the target distribution.
        Must be `None` if the pair `tempered/untempered_log_prob_fn` is provided
      inverse_temperatures: `Tensor` of inverse temperatures to temper each
        replica. The leftmost dimension is the `num_replica` and the
        second dimension through the rightmost can provide different temperature
        to different batch members, doing a left-justified broadcast.
      make_kernel_fn: Python callable which takes a `target_log_prob_fn`
        arg and returns a `tfp.mcmc.TransitionKernel` instance.
      swap_proposal_fn: Python callable which take a number of replicas, and
        returns `swaps`, a shape `[num_replica] + batch_shape` `Tensor`, where
        axis 0 indexes a permutation of `{0,..., num_replica-1}`, designating
        replicas to swap.
      state_includes_replicas: Boolean indicating whether the leftmost dimension
        of each state sample should index replicas. If `True`, the leftmost
        dimension of the `current_state` kwarg to `tfp.mcmc.sample_chain` will
        be interpreted as indexing replicas.
      untempered_log_prob_fn: Python callable which takes an argument like
        `current_state` (or `*current_state` if it's a list) and returns its
        (possibly unnormalized) log-density under the target distribution.
        Must be `None` if `target_log_prob_fn` is provided.
      tempered_log_prob_fn: Optional Python callable with same signature as
        `untempered_log_prob_fn`. Provide this arg if and only if
        `untempered_log_prob_fn` is provided.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      name: Python `str` name prefixed to Ops created by this function.
        Default value: `None` (i.e., "remc_kernel").

    Raises:
      ValueError: `inverse_temperatures` doesn't have statically known 1D shape.
      ValueError: If wrong combination of log prob functions are provided.
    """
    self._parameters = {k: v for k, v in locals().items() if v is not self}
    self._state_includes_replicas = state_includes_replicas

    if (tempered_log_prob_fn is None) != (untempered_log_prob_fn is None):
      raise ValueError(
          'Must provide either neither or both of tempered/untempered log prob '
          'funs. Found `tempered_log_prob_fn is None = ({})`, '
          'and `untempered_log_prob_fn is None = ({})`'.format(
              tempered_log_prob_fn is None,
              untempered_log_prob_fn is None))

    if target_log_prob_fn is not None and tempered_log_prob_fn is not None:
      raise ValueError(
          'Exactly one of `target_log_prob_fn` and `tempered_log_prob_fn` '
          'should be provided. Instead, both were.')

    self._parameters['inverse_temperatures'] = (
        _maybe_embed_inverse_temperature_validation(
            self.inverse_temperatures,
            validate_args,
            using_untempered_log_prob=untempered_log_prob_fn is not None))

  @property
  def target_log_prob_fn(self):
    return self._parameters['target_log_prob_fn']

  @property
  def tempered_log_prob_fn(self):
    return self._parameters['tempered_log_prob_fn']

  @property
  def untempered_log_prob_fn(self):
    return self._parameters['untempered_log_prob_fn']

  @property
  def inverse_temperatures(self):
    return self._parameters['inverse_temperatures']

  def num_replica(self):
    """Integer (`Tensor`) number of replicas being tracked."""
    return tf.constant(ps.size0(self.inverse_temperatures))

  @property
  def make_kernel_fn(self):
    return self._parameters['make_kernel_fn']

  @property
  def swap_proposal_fn(self):
    return self._parameters['swap_proposal_fn']

  @property
  def validate_args(self):
    return self._parameters['validate_args']

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

  def one_step(self, current_state, previous_kernel_results, seed=None):
    """Takes one step of the TransitionKernel.

    Args:
      current_state: `Tensor` or Python `list` of `Tensor`s representing the
        current state(s) of the Markov chain(s).
      previous_kernel_results: A (possibly nested) `tuple`, `namedtuple` or
        `list` of `Tensor`s representing internal calculations made within the
        previous call to this function (or as returned by `bootstrap_results`).
      seed: Optional, a seed for reproducible sampling.

    Returns:
      next_state: `Tensor` or Python `list` of `Tensor`s representing the
        next state(s) of the Markov chain(s).
      kernel_results: A (possibly nested) `tuple`, `namedtuple` or `list` of
        `Tensor`s representing internal calculations made within this function.
        This inculdes replica states.
    """

    # The code below propagates one step states of shape
    #  [n_replica] + batch_shape + event_shape.
    #
    # The step is done in three parts:
    #  1) Call one_step to transition states via a tempered version of
    #     self.target_log_prob_fn (see _replica_target_log_prob).
    #  2) Permute values in states
    #  3) Update state-dependent values, such as log_probs.
    #
    # We chose to swap states, rather than temperatures, because...
    # (i)  If swapping temperatures, you *still* have to swap log_probs to
    #      determine acceptance, as well as states (for kernel results).
    #      So it's just as difficult to swap temperatures.
    # (ii) If swapping temperatures, you have to take care to swap any user-
    #      supplied temperature related things (like step size).
    #      A-priori, we don't know what else will need to be swapped!
    # (iii)In both cases, the kernel results need to be updated in a non-trivial
    #      manner....so we either special-case, or use bootstrap.

    with tf.name_scope(mcmc_util.make_name(self.name, 'remc', 'one_step')):
      # Force a read in case the `inverse_temperatures` is a `tf.Variable`.
      inverse_temperatures = tf.convert_to_tensor(
          previous_kernel_results.inverse_temperatures,
          name='inverse_temperatures')

      target_log_prob_for_inner_kernel = _make_replica_target_log_prob_fn(
          target_log_prob_fn=self.target_log_prob_fn,
          inverse_temperatures=inverse_temperatures,
          untempered_log_prob_fn=self.untempered_log_prob_fn,
          tempered_log_prob_fn=self.tempered_log_prob_fn,
      )
      # TODO(b/159636942): Clean up the helpful error msg after 2020-11-10.
      try:
        inner_kernel = self.make_kernel_fn(  # pylint: disable=not-callable
            target_log_prob_for_inner_kernel)
      except TypeError as e:
        if 'argument' not in str(e):
          raise
        raise TypeError(
            '`ReplicaExchangeMC`s `make_kernel_fn` no longer receives a `seed` '
            'argument. `TransitionKernel` instances now receive seeds via '
            '`one_step`.')

      seed = samplers.sanitize_seed(seed)  # Retain for diagnostics.
      inner_seed, swap_seed, logu_seed = samplers.split_seed(seed, n=3)
      # Step the inner TransitionKernel.
      [
          pre_swap_replica_states,
          pre_swap_replica_results,
      ] = inner_kernel.one_step(
          previous_kernel_results.post_swap_replica_states,
          previous_kernel_results.post_swap_replica_results,
          seed=inner_seed)

      pre_swap_replica_target_log_prob = _get_field(
          # These are tempered log probs (have been divided by temperature).
          pre_swap_replica_results, 'target_log_prob')

      dtype = pre_swap_replica_target_log_prob.dtype
      replica_and_batch_shape = ps.shape(
          pre_swap_replica_target_log_prob)
      batch_shape = replica_and_batch_shape[1:]
      replica_and_batch_rank = ps.rank(
          pre_swap_replica_target_log_prob)
      num_replica = ps.size0(inverse_temperatures)

      inverse_temperatures = mcmc_util.left_justified_broadcast_to(
          inverse_temperatures, replica_and_batch_shape)

      # Now that each replica has done one_step, it is time to consider swaps.

      # swap.shape = [n_replica], and is a "once only" permutation, meaning it
      # is achievable by a sequence of pairwise permutations, where each element
      # is moved at most once.
      # E.g. if swaps = [1, 0, 2], we will consider swapping temperatures 0 and
      # 1, keeping 2 fixed.  This exact same swap is considered for *every*
      # batch member.  Of course some batch members may accept and some reject.
      try:
        swaps = tf.cast(
            self.swap_proposal_fn(  # pylint: disable=not-callable
                num_replica,
                batch_shape=batch_shape,
                seed=swap_seed,
                step_count=previous_kernel_results.step_count),
            dtype=tf.int32)
      except TypeError as e:
        if 'step_count' not in str(e):
          raise
        warnings.warn(
            'The `swap_proposal_fn` given to ReplicaExchangeMC did not accept '
            'the `step_count` argument. Falling back to omitting the '
            'argument. This fallback will be removed after 24-Oct-2020.')
        swaps = tf.cast(
            self.swap_proposal_fn(  # pylint: disable=not-callable
                num_replica,
                batch_shape=batch_shape,
                seed=swap_seed),
            dtype=tf.int32)

      null_swaps = mcmc_util.left_justified_expand_dims_like(
          tf.range(num_replica, dtype=swaps.dtype), swaps)
      swaps = _maybe_embed_swaps_validation(swaps, null_swaps,
                                            self.validate_args)

      # Un-temper the log probs for use in the swap acceptance ratio.
      if self.tempered_log_prob_fn is None:
        # Efficient way of re-evaluating target_log_prob_fn on the
        # pre_swap_replica_states.
        untempered_energy_ignoring_ulp = (
            # Since untempered_log_prob_fn is None, we may assume
            # inverse_temperatures > 0 (else the target is improper).
            pre_swap_replica_target_log_prob / inverse_temperatures)
      else:
        # The untempered_log_prob_fn does not factor into the acceptance ratio.
        # Proof: Suppose the tempered target is
        #   p_k(x) = f(x)^{beta_k} g(x),
        # So f(x) is tempered, and g(x) is not.  Then, the acceptance ratio for
        # a 1 <--> 2 swap is...
        #   (p_1(x_2) p_2(x_1)) / (p_1(x_1) p_2(x_2))
        # which depends only on f(x), since terms involving g(x) cancel.
        untempered_energy_ignoring_ulp = self.tempered_log_prob_fn(
            *pre_swap_replica_states)

      # Since `swaps` is its own inverse permutation we automatically know the
      # swap counterpart: range(num_replica). We use this idea to compute the
      # acceptance in a vectorized manner at the cost of wasting roughly half
      # our computation. Although we could use `unique` to solve this problem,
      # we expect the cost of `unique` to be higher than the dozens of wasted
      # arithmetic calculations. Worse, it'd mean we need dynamic sized Tensors
      # (eg, using `tf.where(bool)`) and so we wouldn't be able to XLA compile.

      # Note: diffs would normally be "proposed - current" however energy is
      # flipped since `energy == -log_prob`.
      # Note: The untempered_log_prob_fn (if provided) is not included in
      # untempered_pre_swap_replica_target_log_prob, and hence does not factor
      # into energy_diff. Why? Because, it cancels out in the acceptance ratio.
      energy_diff = (
          untempered_energy_ignoring_ulp -
          mcmc_util.index_remapping_gather(
              untempered_energy_ignoring_ulp,
              swaps, name='gather_swap_tlp'))
      swapped_inverse_temperatures = mcmc_util.index_remapping_gather(
          inverse_temperatures, swaps, name='gather_swap_temps')
      inverse_temp_diff = swapped_inverse_temperatures - inverse_temperatures

      # If i and j are swapping, log_accept_ratio[] i and j are equal.
      log_accept_ratio = (
          energy_diff * mcmc_util.left_justified_expand_dims_to(
              inverse_temp_diff, replica_and_batch_rank))

      log_accept_ratio = tf.where(
          tf.math.is_finite(log_accept_ratio),
          log_accept_ratio, tf.constant(-np.inf, dtype=dtype))

      # Produce log[Uniform] draws that are identical at swapped indices.
      log_uniform = tf.math.log(
          samplers.uniform(shape=replica_and_batch_shape,
                           dtype=dtype,
                           seed=logu_seed))
      anchor_swaps = tf.minimum(swaps, null_swaps)
      log_uniform = mcmc_util.index_remapping_gather(log_uniform, anchor_swaps)

      is_swap_accepted_mask = tf.less(
          log_uniform,
          log_accept_ratio,
          name='is_swap_accepted_mask')

      def _swap_tensor(x):
        return mcmc_util.choose(
            is_swap_accepted_mask,
            mcmc_util.index_remapping_gather(x, swaps), x)

      post_swap_replica_states = [
          _swap_tensor(s) for s in pre_swap_replica_states]

      expanded_null_swaps = mcmc_util.left_justified_broadcast_to(
          null_swaps, replica_and_batch_shape)
      is_swap_proposed = _compute_swap_notmatrix(
          # Broadcast both so they have shape [num_replica] + batch_shape.
          # This (i) makes them have same shape as is_swap_accepted, and
          # (ii) keeps shape consistent if someday swaps has a batch shape.
          expanded_null_swaps,
          mcmc_util.left_justified_broadcast_to(swaps, replica_and_batch_shape))

      # To get is_swap_accepted in ordered position, we use
      # _compute_swap_notmatrix on current and next replica positions.
      post_swap_replica_position = _swap_tensor(expanded_null_swaps)

      is_swap_accepted = _compute_swap_notmatrix(
          post_swap_replica_position,
          expanded_null_swaps)

      if self._state_includes_replicas:
        post_swap_states = post_swap_replica_states
      else:
        post_swap_states = [s[0] for s in post_swap_replica_states]

      post_swap_replica_results = _set_swapped_fields_to_nan(
          _swap_log_prob_and_maybe_grads(
              pre_swap_replica_results,
              post_swap_replica_states,
              inner_kernel))

      if mcmc_util.is_list_like(current_state):
        # We *always* canonicalize the states in the kernel results.
        states = post_swap_states
      else:
        states = post_swap_states[0]

      post_swap_kernel_results = ReplicaExchangeMCKernelResults(
          post_swap_replica_states=post_swap_replica_states,
          pre_swap_replica_results=pre_swap_replica_results,
          post_swap_replica_results=post_swap_replica_results,
          is_swap_proposed=is_swap_proposed,
          is_swap_accepted=is_swap_accepted,
          is_swap_proposed_adjacent=_sub_diag(is_swap_proposed),
          is_swap_accepted_adjacent=_sub_diag(is_swap_accepted),
          # Store the original pkr.inverse_temperatures in case its a
          # `tf.Variable`.
          inverse_temperatures=previous_kernel_results.inverse_temperatures,
          swaps=swaps,
          step_count=previous_kernel_results.step_count + 1,
          seed=seed,
      )

      return states, post_swap_kernel_results

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
    with tf.name_scope(mcmc_util.make_name(
        self.name, 'remc', 'bootstrap_results')):
      init_state, unused_is_multipart_state = mcmc_util.prepare_state_parts(
          init_state)

      inverse_temperatures = tf.convert_to_tensor(
          self.inverse_temperatures,
          name='inverse_temperatures')

      if self._state_includes_replicas:
        it_n_replica = inverse_temperatures.shape[0]
        state_n_replica = init_state[0].shape[0]
        if ((it_n_replica is not None) and (state_n_replica is not None) and
            (it_n_replica != state_n_replica)):
          raise ValueError(
              'Number of replicas implied by initial state ({}) must equal '
              'number of replicas implied by inverse_temperatures ({}), but '
              'did not'.format(it_n_replica, state_n_replica))

      # We will now replicate each of a possible batch of initial stats, one for
      # each inverse_temperature. So if init_state=[x, y] of shapes [Sx, Sy]
      # then the new shape is [(T, Sx), (T, Sy)] where (a, b) means
      # concatenation and T=shape(inverse_temperature).
      num_replica = ps.size0(inverse_temperatures)
      replica_shape = ps.convert_to_shape_tensor([num_replica])

      if self._state_includes_replicas:
        replica_states = init_state
      else:
        replica_states = [
            tf.broadcast_to(  # pylint: disable=g-complex-comprehension
                x,
                ps.concat([replica_shape, ps.shape(x)], axis=0),
                name='replica_states')
            for x in init_state
        ]

      target_log_prob_for_inner_kernel = _make_replica_target_log_prob_fn(
          target_log_prob_fn=self.target_log_prob_fn,
          inverse_temperatures=inverse_temperatures,
          untempered_log_prob_fn=self.untempered_log_prob_fn,
          tempered_log_prob_fn=self.tempered_log_prob_fn,
      )
      # TODO(b/159636942): Clean up the helpful error msg after 2020-11-10.
      try:
        inner_kernel = self.make_kernel_fn(  # pylint: disable=not-callable
            target_log_prob_for_inner_kernel)
      except TypeError as e:
        if 'argument' not in str(e):
          raise
        raise TypeError(
            '`ReplicaExchangeMC`s `make_kernel_fn` no longer receives a second '
            '(`seed`) argument. `TransitionKernel` instances now receive seeds '
            'via `one_step`.')

      replica_results = inner_kernel.bootstrap_results(replica_states)

      pre_swap_replica_target_log_prob = _get_field(
          replica_results, 'target_log_prob')

      replica_and_batch_shape = ps.shape(
          pre_swap_replica_target_log_prob)
      batch_shape = replica_and_batch_shape[1:]

      inverse_temperatures = mcmc_util.left_justified_broadcast_to(
          inverse_temperatures, replica_and_batch_shape)

      # Pretend we did a "null swap", which will always be accepted.
      swaps = mcmc_util.left_justified_broadcast_to(
          tf.range(num_replica), replica_and_batch_shape)
      # is_swap_accepted.shape = [n_replica, n_replica] + batch_shape.
      is_swap_accepted = distribution_util.rotate_transpose(
          tf.eye(num_replica, batch_shape=batch_shape, dtype=tf.bool),
          shift=2)

      return ReplicaExchangeMCKernelResults(
          post_swap_replica_states=replica_states,
          pre_swap_replica_results=replica_results,
          post_swap_replica_results=_set_swapped_fields_to_nan(replica_results),
          is_swap_proposed=is_swap_accepted,
          is_swap_accepted=is_swap_accepted,
          is_swap_proposed_adjacent=_sub_diag(is_swap_accepted),
          is_swap_accepted_adjacent=_sub_diag(is_swap_accepted),
          inverse_temperatures=self.inverse_temperatures,
          swaps=swaps,
          step_count=tf.zeros(shape=(), dtype=tf.int32),
          seed=samplers.zeros_seed(),
      )


def _make_replica_target_log_prob_fn(
    target_log_prob_fn,
    inverse_temperatures,
    untempered_log_prob_fn=None,
    tempered_log_prob_fn=None,
):
  """Helper which creates inner kernel target_log_prob_fn."""
  def _replica_target_log_prob(*x):
    if tempered_log_prob_fn is not None:
      tlp = tempered_log_prob_fn(*x)
    else:
      tlp = target_log_prob_fn(*x)

    log_prob = tf.cast(mcmc_util.left_justified_expand_dims_like(
        inverse_temperatures, tlp), dtype=tlp.dtype) * tlp

    if untempered_log_prob_fn is not None:
      log_prob = log_prob + untempered_log_prob_fn(*x)

    return log_prob

  return _replica_target_log_prob


def _maybe_embed_swaps_validation(swaps, null_swaps, validate_args):
  """Return `swaps`, possibly with embedded "once only" assertion."""
  if not validate_args:
    return swaps

  assertions = [
      assert_util.assert_equal(
          null_swaps,
          mcmc_util.index_remapping_gather(swaps, swaps),
          message=(
              'Proposed replica swaps must be consist of "once only '
              'swaps," i.e., be a self-inverse permutation, '
              '`range(swaps.shape[0]) == gather(swaps, swaps).')),
  ]
  with tf.control_dependencies(assertions):
    return tf.identity(swaps)


def _maybe_embed_inverse_temperature_validation(
    inverse_temperatures,
    validate_args,
    using_untempered_log_prob,
):
  """Return `inverse_temperatures`, possibly with embedded asserts."""
  if not validate_args:
    return inverse_temperatures

  if using_untempered_log_prob:
    check = tf.debugging.assert_non_negative(
        inverse_temperatures,
        message=(
            '`inverse_temperatures` must be non-negative when using '
            'untempered_log_prob_fn.'
        ))
  else:
    check = tf.debugging.assert_positive(
        inverse_temperatures,
        message=(
            '`inverse_temperatures` must be positive when not using '
            'untempered_log_prob_fn.'
        ))

  with tf.control_dependencies([check]):
    return tf.identity(inverse_temperatures)


def _set_swapped_fields_to_nan(replica_results):
  """Get new replica results, with some fields set to NaN.

  It is unclear what values some fields should take after swapping. For these
  values, we set them to NaN.

  Args:
    replica_results: Replica results, before swapping.

  Returns:
    Copy of replica_results, with some fields set to NaN.
  """
  dtype = _get_field(replica_results, 'target_log_prob').dtype
  nan = tf.convert_to_tensor(np.nan, name='intentional_nan', dtype=dtype)

  for field in ['proposed_results', 'proposed_state']:
    replica_results = _update_field(replica_results, field, nan)

  return replica_results


def _swap_log_prob_and_maybe_grads(
    pre_swap_replica_results,
    post_swap_replica_states,
    inner_kernel,
):
  """Swap 'target_log_prob' and maybe 'grads_target_log_prob'.

  After swapping states, the log probs for replicas need to be evaluated at
  the new points, then tempered. The easiest way to do this is to call
  `inner_kernel.bootstrap_results`.

  Args:
    pre_swap_replica_results: Replica results, before swapping.
    post_swap_replica_states: The states, after swapping.
    inner_kernel: The tempered inner kernel used by ReplicaExchangeMC.

  Returns:
    Copy of pre_swap_replica_results, with some fields set to NaN.
  """
  # This call means we re-evaluate gradients. More efficient surgery could be
  # done, but this is much simpler.
  temp_kr = inner_kernel.bootstrap_results(post_swap_replica_states)

  post_swap_replica_results = pre_swap_replica_results

  # Fields we expect to find, else raise.
  for field in ['target_log_prob']:
    post_swap_replica_results = _update_field(
        post_swap_replica_results, field, _get_field(temp_kr, field))

  # Fields we don't always find.
  for field in ['grads_target_log_prob']:
    try:
      post_swap_replica_results = _update_field(
          post_swap_replica_results, field, _get_field(temp_kr, field))
    except REMCFieldNotFoundError:
      pass

  return post_swap_replica_results


def _compute_swap_notmatrix(before_positions, after_positions):
  """Determine which before_positions were equal to after_positions.

  Args:
    before_positions: Shape [n_replica,...] integer Tensor.  Replica positions
      before an swap.
    after_positions: Shape [n_replica,...] integer Tensor.  Replica positions
      after an swap.

  Returns:
    swaps: Shape [n_replica, n_replica, ...] boolean Tensor.
      swaps[i, j, ..] = True just when
      before_positions[i,...] == after_positions[j, ...].
  """
  with tf.name_scope('compute_swap_notmatrix'):
    # Which one is expanded doesn't matter for once-only swaps, since i --> j if
    # and only if j --> i.
    # Nonetheless, these are the correct expansions if general permutations are
    # allowed.
    return tf.equal(
        before_positions,
        after_positions[:, tf.newaxis],
    )


def _sub_diag(nonmatrix):
  """Get the first sub-diagonal of a shape [N, N, ...] 'non matrix'."""
  with tf.name_scope('sub_matrix'):
    # TODO(b/143702351) Once array_ops.matrix_diag_part_v3 is ready and exposed,
    # replace the call to matrix_diag_part_v2 below with tf.linalg.matrix_diag.
    # We can also stop special casing for matrix_dim < 2 at that point.
    # Until then, OpError raised for 1x1 matricies without static shape.
    # In fact, non-static shape breaks matrix_diag_part_v2, so we must raise
    # this message now.
    # See http://b/138403336 for the TF issue tracker.
    if not tensorshape_util.is_fully_defined(nonmatrix.shape[:2]):
      raise ValueError(
          '`inverse_temperatures did not have statically defined shape, '
          'which breaks tracking of is_swap_{proposed,accepted}.  '
          'Please provide an inverse_temperatures with statically known shape.')

    # The sub-matrix of a 1x1 matrix is not defined (throws exception), so in
    # this special case return an empty matrix.
    # TODO(b/143702351) Remove this special case handling once
    # matrix_diag_part_v3 is ready.
    matrix_dim = ps.size0(nonmatrix)
    if matrix_dim is not None and matrix_dim < 2:
      # Shape is [..., 0], so returned tensor is empty, thus contains no
      # values...and therefore the fact that we use 'ones' doesn't matter.
      shape = ps.pad(
          ps.shape(nonmatrix)[2:],
          paddings=[[0, 1]],
          constant_values=0)
      matrix_sub_diag = tf.cast(tf.ones(shape), nonmatrix.dtype)

    else:
      # Get first sub-diagonal.  `padding_value` is not used (since matrix is
      # square), but is required for the API since this is raw gen_array_ops.
      matrix_sub_diag = tf.raw_ops.MatrixDiagPartV2(
          input=distribution_util.rotate_transpose(nonmatrix, shift=-2),
          k=ps.convert_to_shape_tensor(-1, dtype=tf.int32),
          padding_value=tf.cast(0.0, dtype=nonmatrix.dtype))

    return distribution_util.rotate_transpose(matrix_sub_diag, shift=1)


_kernel_result_not_implemented_message_template = (
    '{} or a kernel '
    'result nested within it is currently not supported by the '
    'ReplicaExchangeMC kernel. Missing field {}. Please file an issue on '
    'the TensorFlow Probability GitHub page.')


class REMCFieldNotFoundError(AttributeError):
  pass


def _get_field(kernel_results, field_name):
  try:
    return unnest.get_innermost(kernel_results, field_name)
  except AttributeError:
    msg = _kernel_result_not_implemented_message_template.format(
        kernel_results, field_name)
    raise REMCFieldNotFoundError(msg)


def _update_field(kernel_results, field_name, value):
  try:
    return unnest.replace_innermost(kernel_results, **{field_name: value})
  except ValueError:
    msg = _kernel_result_not_implemented_message_template.format(
        kernel_results, field_name)
    raise REMCFieldNotFoundError(msg)
