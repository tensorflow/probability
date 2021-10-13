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
"""Convergence criterion based on decrease in a moving average of the loss."""

import collections

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.optimizer.convergence_criteria import convergence_criterion


LossNotDecreasingState = collections.namedtuple(
    'LossNotDecreasingState',
    ['previous_loss',
     'average_decrease_in_loss',
     'average_initial_decrease_in_loss'])


class LossNotDecreasing(convergence_criterion.ConvergenceCriterion):
  """Simple convergence criterion based on lack of decrease in loss values.

  This rule tracks an exponentially-weighted moving average of the decrease
  in loss values between successive steps, and stops when that average drops
  below a threshold.

  ```
  decrease_in_loss[t] = loss[t-1] - loss[t]
  average_decrease_in_loss[t] = (
    (window_size - 1) * average_decrease_in_loss[t - 1] +
     decrease_in_loss[t]) / window_size
  has_converged = (average_decrease_in_loss < threshold)
  ```

  The convergence threshold can be set directly as `atol`, or as a fraction
  of the average loss decrease across the first `window_size` steps
  of the optimization:
  `threshold = rtol * average_decrease_in_loss[window_size]`.
  If both `atol` and `rtol` are specified,
  the maximum of the two thresholds is used (equivalently, the optimization
  stops if either of the two conditions is met).

  The state propagated across training steps is
  `state[t] = LossNotDecreasingState(loss[t],
                                     average_decrease_in_loss[t],
                                     average_decrease_in_loss[window_size]).

  """

  def __init__(self,
               atol=None,
               rtol=None,
               window_size=10,
               min_num_steps=20,
               name=None):
    """Constructs a LossNotDecreasing convergence criterion.

    All numeric arguments may optionally be specified in batch
    (e.g., `atol=[0.3, 0.1]`), in which case the returned values of
    `has_converged` will have shape equal to the broadcast shape of the
    loss values and the convergence criterion arguments.

    Args:
      atol: float `Tensor` absolute tolerance. Convergence is assumed whenever
        (an exponentially-weighted moving average of) the decrease in loss
        values from one step to the next is less than `atol`. If both `atol`
        and `rtol` are specified, then convergence is assumed if *either* of the
        criteria is met.
      rtol: float `Tensor` relative tolerance. Convergence is assumed whenever
        (an exponentially-weighted moving average of) the decrease in loss
        values from one step to the next is less than
        `rtol * average_initial_decrease_in_loss`, where
        `average_initial_decrease_in_loss` is the
        exponentially-weighted moving average of the decrease in loss over the
        first `window_size` steps of the optimization. If both `atol`
        and `rtol` are specified, then convergence is assumed if *either* of
        the criteria is met.
      window_size: int `Tensor` effective window size for the moving average
        decrease in loss. The moving average is computed as
        `moving_average[t] = decrease_in_loss[t] + decay *
        (moving_average[t-1] - decrease_in_loss[t])` where
        `decay = 1. - 1. / window_size`.
        Default value: `10`.
      min_num_steps: int `Tensor` minimum number of steps before convergence.
        The criterion will not return `has_converged=True` until
        `step >= min_num_steps`. This should generally be a larger value than
        `window_size`.
        Default value: `20`.
      name: optional Python `str` name prefixed to ops created by this class.
    """

    if atol is None and rtol is None:
      raise ValueError('Must specify at least one of `atol` and `rtol`')

    self._atol = atol
    self._rtol = rtol
    self._window_size = window_size
    super(LossNotDecreasing, self).__init__(
        min_num_steps=min_num_steps, name=name or 'LossDecrease')

  @property
  def atol(self):
    return self._atol

  @property
  def dtype(self):
    return self._dtype

  @property
  def window_size(self):
    return self._window_size

  @property
  def rtol(self):
    return self._rtol

  def _bootstrap(self, loss, grads, parameters):
    del grads
    del parameters
    return LossNotDecreasingState(
        previous_loss=loss,
        average_decrease_in_loss=tf.zeros_like(loss),
        average_initial_decrease_in_loss=tf.zeros_like(loss))

  def _one_step(self, step, loss, grads, parameters, auxiliary_state):
    del grads
    del parameters

    atol = tf.convert_to_tensor(self.atol if self.atol is not None else 0.,
                                dtype=loss.dtype)
    rtol = tf.convert_to_tensor(self.rtol if self.rtol is not None else 0.,
                                dtype=loss.dtype)
    decay = 1. - 1. / tf.cast(self.window_size, loss.dtype)

    decrease_in_loss = auxiliary_state.previous_loss - loss
    average_decrease_in_loss = (
        decrease_in_loss +
        decay * (auxiliary_state.average_decrease_in_loss - decrease_in_loss))

    has_converged = average_decrease_in_loss < tf.maximum(
        atol, rtol * auxiliary_state.average_initial_decrease_in_loss)
    return has_converged, LossNotDecreasingState(
        previous_loss=loss,
        average_decrease_in_loss=average_decrease_in_loss,
        average_initial_decrease_in_loss=tf.where(
            (tf.cast(step, tf.convert_to_tensor(self.window_size).dtype)
             <= self.window_size),
            average_decrease_in_loss,
            auxiliary_state.average_initial_decrease_in_loss))
