# Copyright 2019 The TensorFlow Probability Authors.
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
"""DualAveragingStepSizeAdaptation TransitionKernel."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools

import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.math.generic import reduce_logmeanexp
from tensorflow_probability.python.mcmc import kernel as kernel_base
from tensorflow_probability.python.mcmc.internal import util as mcmc_util
from tensorflow_probability.python.mcmc.simple_step_size_adaptation import get_differing_dims
from tensorflow_probability.python.mcmc.simple_step_size_adaptation import hmc_like_log_accept_prob_getter_fn
from tensorflow_probability.python.mcmc.simple_step_size_adaptation import hmc_like_step_size_getter_fn
from tensorflow_probability.python.mcmc.simple_step_size_adaptation import hmc_like_step_size_setter_fn


class DualAveragingStepSizeAdaptationResults(
    mcmc_util.PrettyNamedTupleMixin,
    collections.namedtuple(
        'DualAveragingStepSizeAdaptationResults',
        'inner_results, target_accept_prob, log_shrinkage_target, '
        'exploration_shrinkage, step_count_smoothing, decay_rate, error_sum, '
        'log_averaging_step, step, num_adaptation_steps, new_step_size')):
  """Results of the DualAveragingStepSizeAdaptation TransitionKernel.

  Attributes:
    inner_results: Results of the inner kernel.
    target_accept_prob: Floating point scalar `Tensor`. Target accept
      probability.
    log_shrinkage_target: Floating point scalar `Tensor`. Logarithm of value the
      exploration step size is biased towards.
    exploration_shrinkage: Floating point scalar `Tensor`. How strongly the
      exploration rate is biased towards the shrinkage target.
    step_count_smoothing: Int32 scalar `Tensor`. Number of "pseudo-steps" added
      to the number of steps taken to prevents noisy exploration during the
      early samples.
    decay_rate: Floating point scalar `Tensor`. How much to favor recent
      iterations over earlier ones. A value of 1 gives equal weight to all
      history.
    error_sum: Floating point scalar `Tensor`. Aggregator for the sum of the
      acceptance probabilities minus target probability. Algorithm seeks to make
      this 0 in expectation.
    log_averaging_step: Floating point scalar `Tensor` or a list thereof (one
      for (each `state_part`). Smoothed version of the step size.
    step: Int32 scalar `Tensor`. The current step number as perceived by this
      kernel. Increases by 1 for every call to `one_step`.
    num_adaptation_steps: Scalar `int` `Tensor` number of initial steps to
      during which to adjust the step size.
    new_step_size:  Floating point scalar `Tensor` or a list thereof (one for
      each `state_part`). Step size that will be passed to the inner kernel
      during the next step.
  """
  __slots__ = ()


class DualAveragingStepSizeAdaptation(kernel_base.TransitionKernel):
  """Adapts the inner kernel's `step_size` based on `log_accept_prob`.

  The dual averaging policy uses a noisy step size for exploration, while
  averaging over tuning steps to provide a smoothed estimate of an optimal
  value. It is based on [section 3.2 of Hoffman and Gelman (2013)][1], which
  modifies the [stochastic convex optimization scheme of Nesterov (2009)][2].
  The modified algorithm applies extra weight to recent iterations while
  keeping the convergence guarantees of Robbins-Monro, and takes care not
  to make the step size too small too quickly when maintaining a constant
  trajectory length, to avoid expensive early iterations. A good target
  acceptance probability depends on the inner kernel. If this kernel is
  `HamiltonianMonteCarlo`, then 0.6-0.9 is a good range to aim for. For
  `RandomWalkMetropolis` this should be closer to 0.25. See the individual
  kernels' docstrings for guidance.

  In general, adaptation prevents the chain from reaching a stationary
  distribution, so obtaining consistent samples requires `num_adaptation_steps`
  be set to a value [somewhat smaller][3] than the number of burnin steps.
  However, it may sometimes be helpful to set `num_adaptation_steps` to a larger
  value during development in order to inspect the behavior of the chain during
  adaptation.

  The step size is assumed to broadcast with the chain state, potentially having
  leading dimensions corresponding to multiple chains. When there are fewer of
  those leading dimensions than there are chain dimensions, the corresponding
  dimensions in the `log_accept_prob` are averaged (in the direct space, rather
  than the log space) before being used to adjust the step size. This means that
  this kernel can do both cross-chain adaptation, or per-chain step size
  adaptation, depending on the shape of the step size.

  For example, if your problem has a state with shape `[S]`, your chain state
  has shape `[C0, C1, S]` (meaning that there are `C0 * C1` total chains) and
  `log_accept_prob` has shape `[C0, C1]` (one acceptance probability per chain),
  then depending on the shape of the step size, the following will happen:

  - Step size has shape [], [S] or [1], the `log_accept_prob` will be averaged
    across its `C0` and `C1` dimensions. This means that you will learn a shared
    step size based on the mean acceptance probability across all chains. This
    can be useful if you don't have a lot of steps to adapt and want to average
    away the noise.

  - Step size has shape [C1, 1] or [C1, S], the `log_accept_prob` will be
    averaged across its `C0` dimension. This means that you will learn a shared
    step size based on the mean acceptance probability across chains that share
    the coordinate across the `C1` dimension. This can be useful when the `C1`
    dimension indexes different distributions, while `C0` indexes replicas of a
    single distribution, all sampled in parallel.

  - Step size has shape [C0, C1, 1] or [C0, C1, S], then no averaging will
    happen. This means that each chain will learn its own step size. This can be
    useful when all chains are sampling from different distributions. Even when
    all chains are for the same distribution, this can help during the initial
    warmup period.

  - Step size has shape [C0, 1, 1] or [C0, 1, S], the `log_accept_prob` will be
    averaged across its `C1` dimension. This means that you will learn a shared
    step size based on the mean acceptance probability across chains that share
    the coordinate across the `C0` dimension. This can be useful when the `C0`
    dimension indexes different distributions, while `C1` indexes replicas of a
    single distribution, all sampled in parallel.

  By default, the averaging function used above is the arithmetic mean, which is
  not robust to stuck chains (e.g. average of one chain with `p_accept = 0` and
  three chains with `p_accept = 1` will result in an average `p_accept = 0.75`,
  which will cause this kernel keep the step size roughly the same rather than
  reducing it to unstick the stuck chain). A more robust choice would be to set
  `reduce_fn` argument to `tfp.math.reduce_log_harmonic_mean_exp` [4]. Note,
  however, that the harmonic mean of a set of numbers is usually smaller than
  the arithmetic mean, so its use will typically produce smaller than optimal
  step sizes even for well behaved target distributions.

  #### Examples

  ```python
  import tensorflow as tf
  import tensorflow_probability as tfp
  tfd = tfp.distributions

  target_log_prob_fn = tfd.Normal(loc=0., scale=1.).log_prob
  num_burnin_steps = 500
  num_results = 500
  num_chains = 64
  step_size = 0.1
  # Or, if you want per-chain step size:
  # step_size = tf.fill([num_chains], step_size)

  kernel = tfp.mcmc.HamiltonianMonteCarlo(
      target_log_prob_fn=target_log_prob_fn,
      num_leapfrog_steps=2,
      step_size=step_size)
  kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
      inner_kernel=kernel, num_adaptation_steps=int(num_burnin_steps * 0.8))

  # The chain will be stepped for num_results + num_burnin_steps, adapting for
  # the first num_adaptation_steps.
  samples, [step_size, log_accept_ratio] = tfp.mcmc.sample_chain(
      num_results=num_results,
      num_burnin_steps=num_burnin_steps,
      current_state=tf.zeros(num_chains),
      kernel=kernel,
      trace_fn=lambda _, pkr: [pkr.inner_results.accepted_results.step_size,
                               pkr.inner_results.log_accept_ratio])

  # ~0.75
  p_accept = tf.math.exp(tfp.math.reduce_logmeanexp(tf.minimum(
      log_accept_ratio, 0.)))
  ```
  #### References

  [1]: Matthew D. Hoffman, Andrew Gelman. The No-U-Turn Sampler: Adaptively
      Setting Path Lengths in Hamiltonian Monte Carlo.
      In _Journal of Machine Learning Research_, 15(1):1593-1623, 2014.
      http://jmlr.org/papers/volume15/hoffman14a/hoffman14a.pdf

  [2] Yurii Nesterov. Primal-dual subgradient methods for convex problems.
      Mathematical programming 120.1 (2009): 221-259
      https://link.springer.com/article/10.1007/s10107-007-0149-x

  [3]:
  http://andrewgelman.com/2017/12/15/burn-vs-warm-iterative-simulation-algorithms/#comment-627745

  [4]: Hoffman, M., Radul, A., & Sountsov, P. An Adaptive MCMC Scheme
       for Setting Trajectory Lengths in Hamiltonian Monte Carlo, 2020. In
       preparation.
  """

  def __init__(
      self,
      inner_kernel,
      num_adaptation_steps,
      target_accept_prob=0.75,
      exploration_shrinkage=0.05,
      shrinkage_target=None,
      step_count_smoothing=10,
      decay_rate=0.75,
      step_size_setter_fn=hmc_like_step_size_setter_fn,
      step_size_getter_fn=hmc_like_step_size_getter_fn,
      log_accept_prob_getter_fn=hmc_like_log_accept_prob_getter_fn,
      reduce_fn=reduce_logmeanexp,
      experimental_reduce_chain_axis_names=None,
      validate_args=False,
      name=None):
    """Initializes this transition kernel.

    Args:
      inner_kernel: `TransitionKernel`-like object.
      num_adaptation_steps: Scalar `int` `Tensor` number of initial steps to
        during which to adjust the step size. This may be greater, less than, or
        equal to the number of burnin steps.
      target_accept_prob: A floating point `Tensor` representing desired
        acceptance probability. Must be a positive number less than 1. This can
        either be a scalar, or have shape [num_chains]. Default value: `0.75`
          (the [center of asymptotically optimal rate for HMC][1]).
      exploration_shrinkage: Floating point scalar `Tensor`. How strongly the
        exploration rate is biased towards the shrinkage target.
      shrinkage_target: `Tensor` or list of tensors. Value the exploration
        step size(s) is/are biased towards.
        As `num_adaptation_steps --> infinity`, this bias goes to zero.
        Defaults to 10 times the initial step size.
      step_count_smoothing: Int32 scalar `Tensor`. Number of "pseudo-steps"
        added to the number of steps taken to prevents noisy exploration during
        the early samples.
      decay_rate: Floating point scalar `Tensor`. How much to favor recent
        iterations over earlier ones. A value of 1 gives equal weight to all
        history. A value of 0 gives weight only to the most recent iteration.
      step_size_setter_fn: A callable with the signature `(kernel_results,
        new_step_size) -> new_kernel_results` where `kernel_results` are the
        results of the `inner_kernel`, `new_step_size` is a `Tensor` or a nested
        collection of `Tensor`s with the same structure as returned by the
        `step_size_getter_fn`, and `new_kernel_results` are a copy of
        `kernel_results` with the step size(s) set.
      step_size_getter_fn: A callable with the signature `(kernel_results) ->
        step_size` where `kernel_results` are the results of the `inner_kernel`,
        and `step_size` is a floating point `Tensor` or a nested collection of
        such `Tensor`s.
      log_accept_prob_getter_fn: A callable with the signature `(kernel_results)
        -> log_accept_prob` where `kernel_results` are the results of the
        `inner_kernel`, and `log_accept_prob` is a floating point `Tensor`.
        `log_accept_prob` can either be a scalar, or have shape [num_chains]. If
        it's the latter, `step_size` should also have the same leading
        dimension.
      reduce_fn: A callable with signature `(input_tensor, axis, keepdims) ->
        tensor` that returns a log-reduction of `log_accept_prob`, typically
        some sort of mean. By default, this performs an arithmetic mean.
      experimental_reduce_chain_axis_names: A `str` or list of `str`s indicating
        the named axes that should additionally reduced during the log-reduction
        of `log_accept_prob`.
      validate_args: Python `bool`. When `True` kernel parameters are checked
        for validity. When `False` invalid inputs may silently render incorrect
        outputs.
      name: Python `str` name prefixed to Ops created by this function.
        Default value: `None` (i.e., 'dual_averaging_step_size_adaptation').
    """
    inner_kernel = mcmc_util.enable_store_parameters_in_results(inner_kernel)

    with tf.name_scope(
        mcmc_util.make_name(
            name, 'dual_averaging_step_size_adaptation', '__init__')) as name:
      dtype = dtype_util.common_dtype([
          target_accept_prob,
          exploration_shrinkage,
          shrinkage_target,
          decay_rate
      ], dtype_hint=tf.float32)
      target_accept_prob = tf.convert_to_tensor(
          target_accept_prob, dtype=dtype, name='target_accept_prob')
      exploration_shrinkage = tf.convert_to_tensor(
          exploration_shrinkage, dtype=dtype, name='exploration_shrinkage')
      step_count_smoothing = tf.cast(
          # Cast to dtype, since we asked the user to provide Int32, but we
          # want to convert to tensor here (and make it `dtype`).
          # I.e., convert_to_tensor will fail here if the user did what we
          # asked.
          step_count_smoothing, dtype=dtype, name='step_count_smoothing')
      decay_rate = tf.convert_to_tensor(
          decay_rate, dtype=dtype, name='decay_rate')
      num_adaptation_steps = tf.convert_to_tensor(
          num_adaptation_steps, dtype=tf.int32, name='num_adaptation_steps')
      target_accept_prob = _maybe_validate_target_accept_prob(
          target_accept_prob, validate_args)

      if shrinkage_target is not None:
        def _convert(x):
          return tf.convert_to_tensor(x, dtype=dtype, name='shrinkage_target')
        shrinkage_target = tf.nest.map_structure(_convert, shrinkage_target)

    self._parameters = dict(
        inner_kernel=inner_kernel,
        num_adaptation_steps=num_adaptation_steps,
        target_accept_prob=target_accept_prob,
        exploration_shrinkage=exploration_shrinkage,
        shrinkage_target=shrinkage_target,
        step_count_smoothing=step_count_smoothing,
        decay_rate=decay_rate,
        step_size_setter_fn=step_size_setter_fn,
        step_size_getter_fn=step_size_getter_fn,
        log_accept_prob_getter_fn=log_accept_prob_getter_fn,
        reduce_fn=reduce_fn,
        experimental_reduce_chain_axis_names=(
            experimental_reduce_chain_axis_names),
        name=name)

  @property
  def inner_kernel(self):
    return self._parameters['inner_kernel']

  @property
  def name(self):
    return self._parameters['name']

  @property
  def num_adaptation_steps(self):
    return self._parameters['num_adaptation_steps']

  def step_size_setter_fn(self, kernel_results, new_step_size):
    return self._parameters['step_size_setter_fn'](kernel_results,
                                                   new_step_size)

  def step_size_getter_fn(self, kernel_results):
    return self._parameters['step_size_getter_fn'](kernel_results)

  def log_accept_prob_getter_fn(self, kernel_results):
    return self._parameters['log_accept_prob_getter_fn'](kernel_results)

  def reduce_fn(self, input_tensor, axis, keepdims,
                experimental_named_axis=None):
    if experimental_named_axis is None:
      return self._parameters['reduce_fn'](
          input_tensor, axis=axis, keepdims=keepdims)
    return self._parameters['reduce_fn'](
        input_tensor, axis=axis, keepdims=keepdims,
        experimental_named_axis=experimental_named_axis)

  @property
  def parameters(self):
    """Return `dict` of `__init__` arguments and their values."""
    return self._parameters

  def _one_step_part(
      self,
      step_size,
      state,
      error_sum,
      log_averaging_step,
      log_shrinkage_target,
      log_accept_prob_rank=None,
      log_accept_prob=None,
      target_accept_prob=None,
      previous_kernel_results=None):
    """Compute new step sizes for each step size part.

    If step size part has smaller rank than the corresponding state part, then
    the difference is averaged away in the log accept prob.

    Example:

      state_part has shape      [2, 3, 4, 5]
      step_size_part has shape     [1, 4, 1]
      log_accept_prob has shape [2, 3, 4]

    Since step size has 1 rank fewer than the state, we reduce away the leading
    dimension of `log_accept_prob` to get a Tensor with shape [3, 4]. Next,
    since `log_accept_prob` must broadcast into step_size_part on the left, we
    reduce the dimensions where their shapes differ, to get a Tensor with shape
    [1, 4], which now is compatible with the leading dimensions of
    step_size_part.

    There is a subtlety here in that `step_size_parts` might be a length-1 list,
    which means that we'll be "structure-broadcasting" it for all the state
    parts (see logic in, e.g., hmc.py). In this case we must assume that that
    the lone step size provided broadcasts with the event dims of each state
    part. This means that either step size has no dimensions corresponding to
    chain dimensions, or all states are of the same shape. For the former, we
    want to reduce over all chain dimensions. For the later, we want to use
    the same logic as in the non-structure-broadcasted case.

    It turns out we can compute the reduction dimensions for both cases
    uniformly by taking the rank of any state part. This obviously works in
    the second case (where all state ranks are the same). In the first case,
    all state parts have the rank L + D_i + B, where L is the rank of
    log_accept_prob, D_i is the non-shared dimensions amongst all states, and
    B are the shared dimensions of all the states, which are equal to the step
    size. When we subtract B, we will always get a number >= L, which means
    we'll get the full reduction we want.

    Args:
      step_size: Previous step's step_size.
      state: Previous step's state value.
      error_sum: Previous step's error accumulator.
      log_averaging_step: Previous step's log_averaging_step.
      log_shrinkage_target: Floating point scalar `Tensor`. Logarithm of value
        the exploration step size is biased towards.
      log_accept_prob_rank: Rank of log_accept_prob.
      log_accept_prob: Floating point scalar `Tensor`. Target accept
        probability.
      target_accept_prob: A floating point `Tensor` representing desired
        acceptance probability. Must be a positive number less than 1.
      previous_kernel_results: Results struct from previous step.

    Returns:
      new_step_size: Updated `step_size`.
      new_log_averaging_step: Updated `log_averaging_step`.
      new_error_sum: Updated `error_sum`.
    """
    num_reduce_dims = ps.minimum(
        log_accept_prob_rank,
        (ps.rank(state) - ps.rank(step_size)))
    reduced_log_accept_prob = self.reduce_fn(
        log_accept_prob,
        axis=ps.range(num_reduce_dims),
        keepdims=False,
        experimental_named_axis=self.experimental_reduce_chain_axis_names)

    # reduced_log_accept_prob must broadcast into step_size on the
    # left, so we do an additional reduction over dimensions where their
    # shapes differ.
    reduce_indices = get_differing_dims(
        reduced_log_accept_prob, step_size)
    reduced_log_accept_prob = self.reduce_fn(
        reduced_log_accept_prob, axis=reduce_indices, keepdims=True,
        experimental_named_axis=self.experimental_reduce_chain_axis_names)
    new_error_sum = (error_sum +
                     target_accept_prob -
                     tf.math.exp(reduced_log_accept_prob))
    num_ones_to_pad = ps.maximum(
        ps.rank(log_shrinkage_target) -
        ps.rank(new_error_sum), 0)
    new_error_sum_extend = tf.reshape(
        new_error_sum,
        shape=ps.pad(
            ps.shape(new_error_sum),
            paddings=[[0, num_ones_to_pad]],
            constant_values=1))

    step_count_smoothing = previous_kernel_results.step_count_smoothing
    step = tf.cast(
        previous_kernel_results.step, step_count_smoothing.dtype) + 1.
    soft_t = step_count_smoothing + step

    new_log_step = (
        log_shrinkage_target -
        ((tf.cast(new_error_sum_extend, step.dtype) * tf.math.sqrt(step)) /
         (soft_t * previous_kernel_results.exploration_shrinkage)))

    eta = step**(-previous_kernel_results.decay_rate)
    new_log_averaging_step = (eta * new_log_step +
                              (1. - eta) * log_averaging_step)

    # - If still adapting, return an exploring step size,
    # - If just finished, return the averaging step size
    # - Otherwise, do not update
    num_adaptation_steps = previous_kernel_results.num_adaptation_steps
    step = previous_kernel_results.step + 1
    new_step_size = tf.where(
        step < num_adaptation_steps, tf.math.exp(new_log_step),
        tf.where(step > num_adaptation_steps, step_size,
                 tf.math.exp(new_log_averaging_step)))
    new_log_averaging_step = tf.where(step > num_adaptation_steps,
                                      log_averaging_step,
                                      new_log_averaging_step)
    return new_step_size, new_log_averaging_step, new_error_sum

  def one_step(self, current_state, previous_kernel_results, seed=None):
    with tf.name_scope(
        mcmc_util.make_name(self.name, 'dual_averaging_step_size_adaptation',
                            'one_step')):
      # Set the step_size.
      inner_results = self.step_size_setter_fn(
          previous_kernel_results.inner_results,
          previous_kernel_results.new_step_size,
      )

      # Step the inner kernel.
      inner_kwargs = {} if seed is None else dict(seed=seed)
      new_state, new_inner_results = self.inner_kernel.one_step(
          current_state, inner_results, **inner_kwargs)

      # Get the new step size.
      log_accept_prob = self.log_accept_prob_getter_fn(new_inner_results)
      target_accept_prob = previous_kernel_results.target_accept_prob

      step_size = self.step_size_getter_fn(new_inner_results)
      step_size_parts = tf.nest.flatten(step_size)
      log_accept_prob_rank = ps.rank(log_accept_prob)
      error_sum_parts = tf.nest.flatten(previous_kernel_results.error_sum)
      log_averaging_step_parts = tf.nest.flatten(
          previous_kernel_results.log_averaging_step)
      log_shrinkage_target_parts = tf.nest.flatten(
          previous_kernel_results.log_shrinkage_target)
      current_state = tf.nest.flatten(current_state)[:len(step_size_parts)]

      # Build partial function for step size
      step_func = functools.partial(
          self._one_step_part,
          log_accept_prob_rank=log_accept_prob_rank,
          log_accept_prob=log_accept_prob,
          target_accept_prob=target_accept_prob,
          previous_kernel_results=previous_kernel_results)
      # Apply adaptation to each part of the chains
      ret = tf.nest.map_structure(step_func,
                                  step_size_parts,
                                  current_state,
                                  error_sum_parts,
                                  log_averaging_step_parts,
                                  log_shrinkage_target_parts)

      new_step_size, new_log_averaging_step, new_error_sum = zip(*ret)
      new_step_size = tf.nest.pack_sequence_as(step_size, new_step_size)
      new_error_sum = tf.nest.pack_sequence_as(error_sum_parts, new_error_sum)
      new_log_averaging_step = tf.nest.pack_sequence_as(
          log_averaging_step_parts, new_log_averaging_step)

      return new_state, previous_kernel_results._replace(
          inner_results=new_inner_results,
          error_sum=new_error_sum,
          step=previous_kernel_results.step + 1,
          log_averaging_step=new_log_averaging_step,
          new_step_size=new_step_size)

  def bootstrap_results(self, init_state):
    with tf.name_scope(
        mcmc_util.make_name(self.name, 'dual_averaging_step_size_adaptation',
                            'bootstrap_results')):
      inner_results = self.inner_kernel.bootstrap_results(init_state)
      return self._bootstrap_from_inner_results(init_state, inner_results)

  # Pulled out to a separate method to enable more efficient windowed
  # sampling kernel.
  def _bootstrap_from_inner_results(self, init_state, inner_results):
    step_size = self.step_size_getter_fn(inner_results)

    log_accept_prob = self.log_accept_prob_getter_fn(inner_results)

    state_parts = tf.nest.flatten(init_state)
    step_size_parts = tf.nest.flatten(step_size)

    if self._parameters['shrinkage_target'] is None:
      shrinkage_target_parts = [None] * len(step_size_parts)
    else:
      shrinkage_target_parts = tf.nest.flatten(
          self._parameters['shrinkage_target'])
      if len(shrinkage_target_parts) not in [1, len(step_size_parts)]:
        raise ValueError(
            '`shrinkage_target` should be a Tensor or list of tensors of '
            'same length as `step_size`. Found len(`step_size`) = {} and '
            'len(shrinkage_target) = {}'.format(
                len(step_size_parts), len(shrinkage_target_parts)))
      if len(shrinkage_target_parts) < len(step_size_parts):
        shrinkage_target_parts *= len(step_size_parts)

    dtype = dtype_util.common_dtype(step_size_parts, tf.float32)
    error_sum, log_averaging_step, log_shrinkage_target = [], [], []
    for state_part, step_size_part, shrinkage_target_part in zip(
        state_parts, step_size_parts, shrinkage_target_parts):
      num_reduce_dims = ps.minimum(
          ps.rank(log_accept_prob),
          ps.rank(state_part) - ps.rank(step_size_part))
      reduced_log_accept_prob = reduce_logmeanexp(
          log_accept_prob,
          axis=ps.range(num_reduce_dims),
          experimental_named_axis=self.experimental_reduce_chain_axis_names)
      reduce_indices = get_differing_dims(reduced_log_accept_prob,
                                          step_size_part)
      reduced_log_accept_prob = reduce_logmeanexp(
          reduced_log_accept_prob, axis=reduce_indices, keepdims=True)
      error_sum.append(tf.zeros_like(reduced_log_accept_prob, dtype=dtype))
      log_averaging_step.append(tf.zeros_like(step_size_part, dtype=dtype))

      if shrinkage_target_part is None:
        log_shrinkage_target.append(
            float(np.log(10.)) + tf.math.log(step_size_part))
      else:
        log_shrinkage_target.append(
            tf.math.log(tf.cast(shrinkage_target_part, dtype)))

    return DualAveragingStepSizeAdaptationResults(
        inner_results=inner_results,
        step=tf.constant(0, dtype=tf.int32),
        target_accept_prob=tf.cast(self.parameters['target_accept_prob'],
                                   log_accept_prob.dtype),
        log_shrinkage_target=log_shrinkage_target,
        exploration_shrinkage=tf.cast(self.parameters['exploration_shrinkage'],
                                      dtype),
        step_count_smoothing=tf.cast(self.parameters['step_count_smoothing'],
                                     dtype),
        decay_rate=tf.cast(self.parameters['decay_rate'], dtype),
        error_sum=error_sum,
        log_averaging_step=log_averaging_step,
        new_step_size=step_size,
        num_adaptation_steps=tf.cast(self.num_adaptation_steps, dtype=tf.int32))

  @property
  def is_calibrated(self):
    return self.inner_kernel.is_calibrated

  @property
  def experimental_shard_axis_names(self):
    return self.inner_kernel.experimental_shard_axis_names

  def experimental_with_shard_axes(self, shard_axis_names):
    return self.copy(
        inner_kernel=self.inner_kernel.experimental_with_shard_axes(
            shard_axis_names))

  @property
  def experimental_reduce_chain_axis_names(self):
    return self._parameters['experimental_reduce_chain_axis_names']


def _maybe_validate_target_accept_prob(target_accept_prob, validate_args):
  """Validates that target_accept_prob is in (0, 1)."""
  if not validate_args:
    return target_accept_prob
  assertions = [
      assert_util.assert_greater(
          target_accept_prob,
          tf.zeros([], dtype=target_accept_prob.dtype),
          message='`target_accept_prob` must be > 0.'),
      assert_util.assert_less(
          target_accept_prob,
          tf.ones([], dtype=target_accept_prob.dtype),
          message='`target_accept_prob` must be < 1.')
  ]
  with tf.control_dependencies(assertions):
    return tf.identity(target_accept_prob)
