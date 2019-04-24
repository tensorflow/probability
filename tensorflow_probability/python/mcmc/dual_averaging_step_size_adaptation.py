"""DualAveragingStepSizeAdaptation TransitionKernel."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.mcmc import kernel as kernel_base
from tensorflow_probability.python.mcmc.internal import util as mcmc_util


@mcmc_util.make_innermost_setter
def _hmc_like_step_size_setter_fn(kernel_results, new_step_size):
  return kernel_results._replace(
      accepted_results=kernel_results.accepted_results._replace(
          step_size=new_step_size))


@mcmc_util.make_innermost_getter
def _hmc_like_step_size_getter_fn(kernel_results):
  return kernel_results.accepted_results.step_size


@mcmc_util.make_innermost_getter
def _hmc_like_log_accept_prob_getter_fn(kernel_results):
  log_accept_ratio = kernel_results.log_accept_ratio
  return tf.minimum(tf.constant(0.0, log_accept_ratio.dtype), log_accept_ratio)


def _reduce_logmeanexp(value, dims, keepdims=False):
  # This is intentionally numerically imprecise for simplicity. For the purposes
  # of computing the mean acceptance probability this is more than sufficient.
  return tf.math.log(
      tf.reduce_mean(input_tensor=tf.exp(value), axis=dims, keepdims=keepdims)
  )


def _get_differing_dims(a, b):
  # Get the indices of dimensions where shapes of `a` and `b` differ.
  # `a` is allowed to have fewer dimensions than `b`.
  if a.shape.is_fully_defined() and b.shape.is_fully_defined():
    a_shape = np.array(a.shape.as_list())
    b_shape = np.array(b.shape.as_list())
    return np.where(a_shape != b_shape[: len(a_shape)])[0]
  else:
    return tf.where(
        tf.not_equal(tf.shape(input=a),
                     tf.shape(input=b)[: tf.rank(a)]))[:, 0]


class DualAveragingStepSizeAdaptationResults(
    collections.namedtuple(
        'DualAveragingStepSizeAdaptationResults',
        'inner_results, target_accept_prob, mu, gamma, t0, kappa, error_sum, '
        'log_averaging_step, step, new_step_size')):
  __slots__ = ()


class DualAveragingStepSizeAdaptation(kernel_base.TransitionKernel):
  def __init__(
      self,
      inner_kernel,
      num_adaptation_steps,
      target_accept_prob=0.75,
      gamma=0.05,
      t0=10.0,
      kappa=0.75,
      step_size_setter_fn=_hmc_like_step_size_setter_fn,
      step_size_getter_fn=_hmc_like_step_size_getter_fn,
      log_accept_prob_getter_fn=_hmc_like_log_accept_prob_getter_fn,
      validate_args=False,
      name=None,
  ):
    inner_kernel = mcmc_util.enable_store_parameters_in_results(inner_kernel)

    with tf.name_scope(
        mcmc_util.make_name(name, 'simple_step_size_adaptation', '__init__')
    ) as name:
      dtype = dtype_util.common_dtype([target_accept_prob, gamma, t0, kappa],
                                      tf.float32)
      target_accept_prob = tf.convert_to_tensor(
          value=target_accept_prob, dtype=dtype, name='target_accept_prob')
      gamma = tf.convert_to_tensor(value=gamma, dtype=dtype, name='gamma')
      t0 = tf.convert_to_tensor(value=t0, dtype=dtype, name='t0')
      kappa = tf.convert_to_tensor(value=kappa, dtype=dtype, name='kappa')
      num_adaptation_steps = tf.convert_to_tensor(
          value=num_adaptation_steps,
          dtype=tf.int32,
          name='num_adaptation_steps')

      target_accept_prob = _maybe_validate_target_accept_prob(
          target_accept_prob, validate_args)

    self._parameters = dict(
        inner_kernel=inner_kernel,
        num_adaptation_steps=num_adaptation_steps,
        target_accept_prob=target_accept_prob,
        gamma=gamma,
        t0=t0,
        kappa=kappa,
        step_size_setter_fn=step_size_setter_fn,
        step_size_getter_fn=step_size_getter_fn,
        log_accept_prob_getter_fn=log_accept_prob_getter_fn,
        name=name
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
    with tf.name_scope(
        mcmc_util.make_name(self.name, 'simple_step_size_adaptation',
                            'one_step')):
      # Set the step_size.
      inner_results = self.step_size_setter_fn(
          previous_kernel_results.inner_results,
          previous_kernel_results.new_step_size,
      )

      # Step the inner kernel.
      new_state, new_inner_results = self.inner_kernel.one_step(
          current_state, inner_results)

      # Get the new step size.
      log_accept_prob = self.log_accept_prob_getter_fn(new_inner_results)
      target_accept_prob = previous_kernel_results.target_accept_prob

      state_parts = tf.nest.flatten(current_state)
      step_size = self.step_size_getter_fn(new_inner_results)
      step_size_parts = tf.nest.flatten(step_size)
      log_accept_prob_rank = tf.rank(log_accept_prob)

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
        num_reduce_dims = tf.minimum(
            log_accept_prob_rank,
            tf.rank(state_part) - tf.rank(step_size_part))
        reduced_log_accept_prob = _reduce_logmeanexp(log_accept_prob,
                                                     tf.range(num_reduce_dims))
        # reduced_log_accept_prob must broadcast into step_size_part on the
        # left, so we do an additional reduction over dimensions where their
        # shapes differ.
        reduce_indices = _get_differing_dims(reduced_log_accept_prob,
                                             step_size_part)
        reduced_log_accept_prob = _reduce_logmeanexp(
            reduced_log_accept_prob, reduce_indices, keepdims=True)

        t0 = previous_kernel_results.t0
        t = t0 + tf.cast(previous_kernel_results.step, t0.dtype)
        new_error_sum = (previous_kernel_results.error_sum +
                         target_accept_prob -
                         tf.math.exp(reduced_log_accept_prob))
        log_step = (
            previous_kernel_results.mu -
            new_error_sum / (tf.math.sqrt(t) * previous_kernel_results.gamma))
        eta = tf.math.pow(t, -previous_kernel_results.kappa)
        new_log_averaging_step = (
            eta * log_step +
            (1 - eta) * previous_kernel_results.log_averaging_step)

        # - If still adapting, return an exploring step size,
        # - If just finished, return the averaging step size
        # - Otherwise, do not update
        new_step_size_parts.append(
            tf.where(
                previous_kernel_results.step < self.num_adaptation_steps,
                tf.math.exp(log_step),
                tf.where(
                    previous_kernel_results.step > self.num_adaptation_steps,
                    step_size_part,
                    tf.math.exp(new_log_averaging_step)
                )
            )
        )
      new_step_size = tf.nest.pack_sequence_as(step_size, new_step_size_parts)

      return new_state, previous_kernel_results._replace(
          inner_results=new_inner_results,
          error_sum=new_error_sum,
          step=previous_kernel_results.step + 1,
          log_averaging_step=new_log_averaging_step,
          new_step_size=new_step_size)

  def bootstrap_results(self, init_state):
    with tf.name_scope(
        mcmc_util.make_name(self.name, 'simple_step_size_adaptation',
                            'bootstrap_results')):
      inner_results = self.inner_kernel.bootstrap_results(init_state)
      step_size = self.step_size_getter_fn(inner_results)

      return DualAveragingStepSizeAdaptationResults(
          inner_results=inner_results,
          step=tf.constant(0, dtype=tf.int32),
          target_accept_prob=self.parameters['target_accept_prob'],
          mu=tf.math.log(10 * step_size),
          gamma=self.parameters['gamma'],
          t0=self.parameters['t0'],
          kappa=self.parameters['kappa'],
          error_sum=tf.constant(0., dtype=tf.float32),
          log_averaging_step=tf.constant(0., dtype=tf.float32),
          new_step_size=step_size,
      )

  def is_calibrated(self):
    return self.inner_kernel.is_calibrated()


def _maybe_validate_target_accept_prob(target_accept_prob, validate_args):
  """Validates that target_accept_prob is in (0, 1)."""
  if not validate_args:
    return target_accept_prob
  with tf.control_dependencies([
      tf.assert_positive(
          target_accept_prob, message='`target_accept_prob` must be > 0.'
      ),
      tf.assert_less(
          target_accept_prob,
          tf.ones_like(target_accept_prob),
          message='`target_accept_prob` must be < 1.'),
  ]):
    return tf.identity(target_accept_prob)

