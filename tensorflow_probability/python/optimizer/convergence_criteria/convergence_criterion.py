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
"""Base class for convergence criteria."""

import tensorflow.compat.v2 as tf


class ConvergenceCriterion(object):
  """Base class for stopping rules.

  A convergence criterion determines when an optimization has converged given
  its history of losses, gradients, and parameter values. Each criterion is
  responsible for propagating from step to step whatever state it needs to
  represent the relevant aspects of that history (for example, a moving average
  of previous loss values or gradients). In particular, subclasses must
  implement:

  - `_bootstrap(loss, grads, parameters)`: takes the
    initial loss, gradients, and values of parameters, and returns a (structure
    of) `Tensor`(s) representing the initial values of any auxiliary quantities
    tracked by the convergence criterion.
  - `_one_step(step, loss, grads, window_size, auxiliary_state)`: At
    integer `step >= 1`,
    takes the current loss, gradients, and values of parameters, along with
    any auxiliary state carried over from the previous step, and returns
    `(has_converged, updated_auxiliary_state)`, where `has_converged` is a
    boolean `Tensor`, and `updated_auxiliary_state` is a (structure of)
    Tensor(s) matching `auxiliary_state`, containing whatever information must
    be propagated to the next timestep.
  """

  def __init__(self, min_num_steps=None, name=None):
    """Constructs the `ConvergenceCriterion`.

    This is a private method for subclass use.

    Args:
      min_num_steps: optional int `Tensor` minimum number of steps before
        stopping. If set, subclass return values of `has_converged=True` will be
        ignored until `step >= min_num_steps`.
        Default value: `None`.
      name: optional Python `str` name prefixed to ops created by this class.
    """
    self._min_num_steps = tf.convert_to_tensor(min_num_steps, dtype=tf.int32)
    self._name = name

  @property
  def min_num_steps(self):
    return self._min_num_steps

  @property
  def name(self):
    return self._name

  def bootstrap(self, loss, grads, parameters):
    """Returns a structure of `Tensors` for the rule's state at step 0.

    The shape of the `Tensor`s specifying `loss`, `grads`, and `parameters` may
    optionally be prefixed by one or more batch dimension(s).

    Args:
      loss: float `Tensor` initial value of loss being optimized.
      grads: list of float `Tensor` gradients of `loss` wrt `parameters`.
      parameters: list of float `Tensor` initial values of parameters
        being optimized.
    Returns:
      initial_auxiliary_state: (Structure of) `Tensor`(s) representing the
        initial auxiliary state carried forward by this criterion.
    """
    with tf.name_scope(self.name):
      return self._bootstrap(
          loss=tf.convert_to_tensor(loss),
          grads=(tf.nest.map_structure(tf.convert_to_tensor, grads)
                 if grads is not None else grads),
          parameters=tf.nest.map_structure(tf.convert_to_tensor, parameters))

  def one_step(self, step, loss, grads, parameters, auxiliary_state):
    """Updates tracked quantities for a new step, and determines if converged.

    The shape of the `Tensor`s specifying `loss`, `grads`, and `parameters` may
    optionally be prefixed by one or more batch dimension(s). In this case,
    the returned value `has_converged` will have shape equal to the broadcast
    batch shape of whichever of those quantities is used by this convergence
    criterion, and the quantities defining the convergence criterion (
    `min_num_steps`, etc.).

    Args:
      step: integer `Tensor` index of the current step, where `step >= 1` (on
        step `0`, `initial_state` should be called instead).
      loss: float `Tensor`  value of loss at the current step.
      grads: list of float `Tensor` gradients of `loss` wrt `parameters`.
      parameters: list of float `Tensor` current values of parameters
        being optimized.
      auxiliary_state: the (structure of) `Tensor`(s) containing state carried
        forward from the previous step.
    Returns:
      has_converged: boolean `Tensor` indicating whether the optimization has
        converged.
      updated_auxiliary_state: (Structure of) `Tensor`(s) representing
        updated quantities tracked by the convergence criterion. This should
        match the structure of the value returned by `bootstrap`.
    """
    with tf.name_scope(self.name):
      has_converged, updated_auxiliary_state = self._one_step(
          step=tf.convert_to_tensor(step),
          loss=tf.convert_to_tensor(loss),
          grads=(tf.nest.map_structure(tf.convert_to_tensor, grads)
                 if grads is not None else grads),
          parameters=tf.nest.map_structure(tf.convert_to_tensor, parameters),
          auxiliary_state=auxiliary_state)
      if self.min_num_steps is not None:
        has_converged = has_converged & (step >= self.min_num_steps)
      return has_converged, updated_auxiliary_state

  def _bootstrap(self, loss, grads, parameters):
    raise NotImplementedError('`_bootstrap` not implemented.')

  def _one_step(self, step, loss, grads, parameters, auxiliary_state):
    raise NotImplementedError('`_one_step` not implemented.')
