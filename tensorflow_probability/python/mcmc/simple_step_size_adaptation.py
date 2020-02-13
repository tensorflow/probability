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
"""SimpleStepSizeAdaptation TransitionKernel."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.math.generic import reduce_logmeanexp
from tensorflow_probability.python.mcmc import kernel as kernel_base
from tensorflow_probability.python.mcmc.internal import util as mcmc_util


@mcmc_util.make_innermost_setter
def hmc_like_step_size_setter_fn(kernel_results, new_step_size):
  """Setter for `step_size` so it can be adapted."""
  # For some `TransitionKernels` (e.g. `MetropolisHastings`), `step_size` is
  # nested inside another `kernel_results` member. Here, we handle the special
  # case in which that member is called `accepted_results`.
  # Note that the `make_innermost_setter` wrapper above handles the relatively
  # common case of that member being called `inner_results`.
  if hasattr(kernel_results, 'accepted_results'):
    return kernel_results._replace(
        accepted_results=kernel_results.accepted_results._replace(
            step_size=new_step_size))
  else:
    return kernel_results._replace(step_size=new_step_size)


@mcmc_util.make_innermost_getter
def hmc_like_step_size_getter_fn(kernel_results):
  """Getter for `step_size` so it can be inspected."""
  # For some `TransitionKernels` (e.g. `MetropolisHastings`), `step_size` is
  # nested inside another `kernel_results` member. Here, we handle the special
  # case in which that member is called `accepted_results`.
  # Note that the `make_innermost_getter` wrapper above handles the relatively
  # common case of that member being called `inner_results`.
  if hasattr(kernel_results, 'accepted_results'):
    return kernel_results.accepted_results.step_size
  else:
    return kernel_results.step_size


@mcmc_util.make_innermost_getter
def hmc_like_log_accept_prob_getter_fn(kernel_results):
  safe_accept_ratio = tf.where(
      tf.math.is_finite(kernel_results.log_accept_ratio),
      kernel_results.log_accept_ratio,
      tf.constant(-np.inf, dtype=kernel_results.log_accept_ratio.dtype))
  return tf.minimum(safe_accept_ratio, 0.)


def get_differing_dims(a, b):
  # Get the indices of dimensions where shapes of `a` and `b` differ.
  # `a` is allowed to have fewer dimensions than `b`.
  if (not tensorshape_util.is_fully_defined(a.shape) or
      not tensorshape_util.is_fully_defined(b.shape)):
    return tf.where(
        tf.not_equal(tf.shape(a), tf.shape(b)[:tf.rank(a)]))[:, 0]
  a_shape = np.int32(a.shape)
  b_shape = np.int32(b.shape)
  return np.where(a_shape != b_shape[:len(a_shape)])[0]


class SimpleStepSizeAdaptationResults(
    collections.namedtuple(
        'SimpleStepSizeAdaptationResults',
        [
            'inner_results',
            'target_accept_prob',
            'adaptation_rate',
            'step',
            'new_step_size',
        ])):
  """Results of the SimpleStepSizeAdaptation TransitionKernel.

  Attributes:
    inner_results: Results of the inner kernel.
    target_accept_prob: Floating point scalar `Tensor`. Target accept
      probability.
    adaptation_rate: Floating point scalar `Tensor`. Fraction by which to adjust
      the step size during each step.
    step: Int32 scalar `Tensor`. The current step number as perceived by this
      kernel. Increases by 1 for every call to `one_step`.
    new_step_size:  Floating point scalar `Tensor` or a list thereof (one for
      each `state_part`). Step size that will be passed to the inner kernel
      during the next step.
  """
  __slots__ = ()


class SimpleStepSizeAdaptation(kernel_base.TransitionKernel):
  """Adapts the inner kernel's `step_size` based on `log_accept_prob`.

  The simple policy multiplicatively increases or decreases the `step_size` of
  the inner kernel based on the value of `log_accept_prob`. It is based on
  [equation 19 of Andrieu and Thoms (2008)][1]. Given enough steps and small
  enough `adaptation_rate` the median of the distribution of the acceptance
  probability will converge to the `target_accept_prob`. A good target
  acceptance probability depends on the inner kernel. If this kernel is
  `HamiltonianMonteCarlo`, then 0.6-0.9 is a good range to aim for. For
  `RandomWalkMetropolis` this should be closer to 0.25. See the individual
  kernels' docstrings for guidance.

  In general, adaptation prevents the chain from reaching a stationary
  distribution, so obtaining consistent samples requires `num_adaptation_steps`
  be set to a value [somewhat smaller][2] than the number of burnin steps.
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
  has shape `[C0, C1, Y]` (meaning that there are `C0 * C1` total chains) and
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
  kernel = tfp.mcmc.SimpleStepSizeAdaptation(
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
  p_accept = tf.math.exp(tfp.math.reduce_logmeanexp(
      tf.minimum(log_accept_ratio, 0.)))
  ```

  #### References

  [1]: Andrieu, Christophe, Thoms, Johannes. A tutorial on adaptive MCMC.
       _Statistics and Computing_, 2008.
       https://people.eecs.berkeley.edu/~jordan/sail/readings/andrieu-thoms.pdf

  [2]:
  http://andrewgelman.com/2017/12/15/burn-vs-warm-iterative-simulation-algorithms/#comment-627745

  """

  def __init__(self,
               inner_kernel,
               num_adaptation_steps,
               target_accept_prob=0.75,
               adaptation_rate=0.01,
               step_size_setter_fn=hmc_like_step_size_setter_fn,
               step_size_getter_fn=hmc_like_step_size_getter_fn,
               log_accept_prob_getter_fn=hmc_like_log_accept_prob_getter_fn,
               validate_args=False,
               name=None):
    """Creates the step size adaptation kernel.

    The default setter_fn and the getter_fn callbacks assume that the inner
    kernel produces kernel results structurally the same as the
    `HamiltonianMonteCarlo` kernel.

    Args:
      inner_kernel: `TransitionKernel`-like object.
      num_adaptation_steps: Scalar `int` `Tensor` number of initial steps to
        during which to adjust the step size. This may be greater, less than, or
        equal to the number of burnin steps.
      target_accept_prob: A floating point `Tensor` representing desired
        acceptance probability. Must be a positive number less than 1. This can
        either be a scalar, or have shape [num_chains]. Default value: `0.75`
        (the [center of asymptotically optimal rate for HMC][1]).
      adaptation_rate: `Tensor` representing amount to scale the current
        `step_size`.
      step_size_setter_fn: A callable with the signature
        `(kernel_results, new_step_size) -> new_kernel_results` where
        `kernel_results` are the results of the `inner_kernel`, `new_step_size`
        is a `Tensor` or a nested collection of `Tensor`s with the same
        structure as returned by the `step_size_getter_fn`, and
        `new_kernel_results` are a copy of `kernel_results` with the step
        size(s) set.
      step_size_getter_fn: A callable with the signature
        `(kernel_results) -> step_size` where `kernel_results` are the results
        of the `inner_kernel`, and `step_size` is a floating point `Tensor` or a
        nested collection of such `Tensor`s.
      log_accept_prob_getter_fn: A callable with the signature
        `(kernel_results) -> log_accept_prob` where `kernel_results` are the
        results of the `inner_kernel`, and `log_accept_prob` is a floating point
        `Tensor`. `log_accept_prob` can either be a scalar, or have shape
        [num_chains]. If it's the latter, `step_size` should also have the same
        leading dimension.
      validate_args: Python `bool`. When `True` kernel parameters are checked
        for validity. When `False` invalid inputs may silently render incorrect
        outputs.
      name: Python `str` name prefixed to Ops created by this class. Default:
        'simple_step_size_adaptation'.

    #### References

    [1]: Betancourt, M. J., Byrne, S., & Girolami, M. (2014). _Optimizing The
         Integrator Step Size for Hamiltonian Monte Carlo_.
         http://arxiv.org/abs/1411.6669
    """

    inner_kernel = mcmc_util.enable_store_parameters_in_results(inner_kernel)

    with tf.name_scope(mcmc_util.make_name(
        name, 'simple_step_size_adaptation', '__init__')) as name:
      dtype = dtype_util.common_dtype([target_accept_prob, adaptation_rate],
                                      tf.float32)
      target_accept_prob = tf.convert_to_tensor(
          target_accept_prob, dtype=dtype, name='target_accept_prob')
      adaptation_rate = tf.convert_to_tensor(
          adaptation_rate, dtype=dtype, name='adaptation_rate')
      num_adaptation_steps = tf.convert_to_tensor(
          num_adaptation_steps,
          dtype=tf.int32,
          name='num_adaptation_steps')

      target_accept_prob = _maybe_validate_target_accept_prob(
          target_accept_prob, validate_args)

    self._parameters = dict(
        inner_kernel=inner_kernel,
        num_adaptation_steps=num_adaptation_steps,
        target_accept_prob=target_accept_prob,
        adaptation_rate=adaptation_rate,
        step_size_setter_fn=step_size_setter_fn,
        step_size_getter_fn=step_size_getter_fn,
        log_accept_prob_getter_fn=log_accept_prob_getter_fn,
        name=name,
    )

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

  @property
  def parameters(self):
    """Return `dict` of ``__init__`` arguments and their values."""
    return self._parameters

  def one_step(self, current_state, previous_kernel_results):
    with tf.name_scope(mcmc_util.make_name(
        self.name, 'simple_step_size_adaptation', 'one_step')):
      # Set the step_size.
      inner_results = self.step_size_setter_fn(
          previous_kernel_results.inner_results,
          previous_kernel_results.new_step_size)

      # Step the inner kernel.
      new_state, new_inner_results = self.inner_kernel.one_step(
          current_state, inner_results)

      # Get the new step size.
      log_accept_prob = self.log_accept_prob_getter_fn(new_inner_results)
      log_target_accept_prob = tf.math.log(
          tf.cast(previous_kernel_results.target_accept_prob,
                  dtype=log_accept_prob.dtype))

      state_parts = tf.nest.flatten(current_state)
      step_size = self.step_size_getter_fn(new_inner_results)
      step_size_parts = tf.nest.flatten(step_size)
      log_accept_prob_rank = prefer_static.rank(log_accept_prob)

      new_step_size_parts = []
      for step_size_part, state_part in zip(step_size_parts, state_parts):
        # Compute new step sizes for each step size part. If step size part has
        # smaller rank than the corresponding state part, then the difference is
        # averaged away in the log accept prob.
        #
        # Example:
        #
        # state_part has shape      [2, 3, 4, 5]
        # step_size_part has shape     [1, 4, 1]
        # log_accept_prob has shape [2, 3, 4]
        #
        # Since step size has 1 rank fewer than the state, we reduce away the
        # leading dimension of log_accept_prob to get a Tensor with shape [3,
        # 4]. Next, since log_accept_prob must broadcast into step_size_part on
        # the left, we reduce the dimensions where their shapes differ, to get a
        # Tensor with shape [1, 4], which now is compatible with the leading
        # dimensions of step_size_part.
        #
        # There is a subtlety here in that step_size_parts might be a length-1
        # list, which means that we'll be "structure-broadcasting" it for all
        # the state parts (see logic in, e.g., hmc.py). In this case we must
        # assume that that the lone step size provided broadcasts with the event
        # dims of each state part. This means that either step size has no
        # dimensions corresponding to chain dimensions, or all states are of the
        # same shape. For the former, we want to reduce over all chain
        # dimensions. For the later, we want to use the same logic as in the
        # non-structure-broadcasted case.
        #
        # It turns out we can compute the reduction dimensions for both cases
        # uniformly by taking the rank of any state part. This obviously works
        # in the second case (where all state ranks are the same). In the first
        # case, all state parts have the rank L + D_i + B, where L is the rank
        # of log_accept_prob, D_i is the non-shared dimensions amongst all
        # states, and B are the shared dimensions of all the states, which are
        # equal to the step size. When we subtract B, we will always get a
        # number >= L, which means we'll get the full reduction we want.
        num_reduce_dims = prefer_static.minimum(
            log_accept_prob_rank,
            prefer_static.rank(state_part) - prefer_static.rank(step_size_part))
        reduced_log_accept_prob = reduce_logmeanexp(
            log_accept_prob,
            axis=prefer_static.range(num_reduce_dims))
        # reduced_log_accept_prob must broadcast into step_size_part on the
        # left, so we do an additional reduction over dimensions where their
        # shapes differ.
        reduce_indices = get_differing_dims(reduced_log_accept_prob,
                                            step_size_part)
        reduced_log_accept_prob = reduce_logmeanexp(
            reduced_log_accept_prob, axis=reduce_indices, keepdims=True)

        one_plus_adaptation_rate = 1. + tf.cast(
            previous_kernel_results.adaptation_rate,
            dtype=step_size_part.dtype)
        new_step_size_part = mcmc_util.choose(
            reduced_log_accept_prob > log_target_accept_prob,
            step_size_part * one_plus_adaptation_rate,
            step_size_part / one_plus_adaptation_rate)

        new_step_size_parts.append(
            tf.where(previous_kernel_results.step < self.num_adaptation_steps,
                     new_step_size_part,
                     step_size_part))
      new_step_size = tf.nest.pack_sequence_as(step_size, new_step_size_parts)

      return new_state, previous_kernel_results._replace(
          inner_results=new_inner_results,
          step=1 + previous_kernel_results.step,
          new_step_size=new_step_size)

  def bootstrap_results(self, init_state):
    with tf.name_scope(mcmc_util.make_name(
        self.name, 'simple_step_size_adaptation', 'bootstrap_results')):
      inner_results = self.inner_kernel.bootstrap_results(init_state)
      step_size = self.step_size_getter_fn(inner_results)
      return SimpleStepSizeAdaptationResults(
          inner_results=inner_results,
          step=tf.constant(0, dtype=tf.int32),
          target_accept_prob=self.parameters['target_accept_prob'],
          adaptation_rate=self.parameters['adaptation_rate'],
          new_step_size=step_size)

  @property
  def is_calibrated(self):
    return self.inner_kernel.is_calibrated


def _maybe_validate_target_accept_prob(target_accept_prob, validate_args):
  """Validates that target_accept_prob is in (0, 1)."""
  if not validate_args:
    return target_accept_prob
  with tf.control_dependencies([
      assert_util.assert_positive(
          target_accept_prob, message='`target_accept_prob` must be > 0.'),
      assert_util.assert_less(
          target_accept_prob,
          tf.ones_like(target_accept_prob),
          message='`target_accept_prob` must be < 1.')
  ]):
    return tf.identity(target_accept_prob)
