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
"""Numpy backend for auto-batching VM.

It can be faster than TF for tiny examples and prototyping, and moderately
simpler due to immediate as opposed to deferred result computation.

All operations take and ignore name= arguments to allow for useful op names in
the TensorFlow backend.
"""

import collections

# Dependency imports
import numpy as np

from tensorflow_probability.python.experimental.auto_batching import instructions


__all__ = ['NumpyBackend']


class RegisterNumpyVariable(collections.namedtuple(
    'RegisterNumpyVariable', ['value'])):
  """A register-only variable.

  Efficiently stores and updates values whose lifetime does not cross function
  calls (and therefore does not require a stack).  This is different from
  `TemporaryVariable` because it supports crossing basic block boundaries.  A
  `RegisterNumpyVariable` therefore needs to store its content persistently
  across the `while_loop` in `execute`, and to handle divergence (and
  re-convergence) of logical threads.
  """

  def update(self, value, mask):
    new_value = np.where(mask, value, self.value)
    return type(self)(new_value)

  def push(self, mask):
    del mask
    return self

  def read(self):
    return self.value

  def pop(self, mask):
    del mask
    return self


# TODO(cl/196846925): Revisit decision to have this be a namedtuple as
# opposed to a class with (potentially private) members.
class Stack(collections.namedtuple('Stack', ['stack', 'stack_index'])):
  """Internal container for a batched stack.

  The implementation is a preallocated array and a (batched) stack
  pointer.

  The namedtuple structure exposes the full state of the stack, and is useful
  for testing, passing through flatten/unflatten operations, and general
  symmetry with the TensorFlow backend.
  """

  def pop(self, mask):
    """Pops each indicated batch member, returning a previous write.

    Args:
      mask: Boolean array of shape `[batch_size]`. The threads at `True`
        indices of `mask` will have their frame pointers regressed by 1.

    Returns:
      stack: Updated variable. Does not mutate `self`.
      read: The new top of the stack, after regressing the frame pointers
        indicated by `mask`.

    Raises:
      ValueError: On an attempt to pop the last value off a batch member.
    """
    new_stack_index = self.stack_index - mask
    if np.any(new_stack_index < 0):
      raise ValueError('Popping the last value off a stack.')
    batch_size = self.stack_index.shape[0]
    # stack:       [max_stack_depth * batch_size, ...]
    # stack_index:                   [batch_size]
    # returns:                       [batch_size, ...]
    indices = new_stack_index * batch_size + np.arange(batch_size)
    return (Stack(self.stack, new_stack_index),
            np.take(self.stack, indices, axis=0))

  def push(self, value, mask):
    """Writes new value to all threads, updates frame of those in `mask`.

    Args:
      value: Value to write into all threads top frame before updating `mask`
        frame pointers.
      mask: Boolean array of shape `[batch_size]`. The values at `True`
        indices of `mask` are updated; the others remain.

    Returns:
      stack: Updated stack. Does not mutate `self`.

    Raises:
      ValueError: If a push exceeds the maximum stack depth.
    """
    # stack:       [max_stack_depth * batch_size, ...]
    # stack_index:                   [batch_size]
    # value:                         [batch_size, ...]
    batch_size = self.stack_index.shape[0]
    max_stack_depth = self.stack.shape[0] // batch_size
    tiled_value = np.reshape(
        np.repeat(value[np.newaxis], max_stack_depth, axis=0),
        self.stack.shape)
    update_indices = self.stack_index * batch_size + np.arange(batch_size)
    new_stack = _where_leading_dims(
        np.any(
            np.equal(update_indices,
                     np.arange(self.stack.shape[0])[:, np.newaxis]),
            axis=1), tiled_value, self.stack)
    new_stack_index = self.stack_index + mask
    if np.any(new_stack_index >= max_stack_depth):
      raise ValueError('Maximum stack depth exceeded.')
    return Stack(new_stack, new_stack_index)


def _create_stack(max_stack_depth, value):
  """Creates a new Stack instance.

  Args:
    max_stack_depth: Python `int` indicating the depth of stack we
      should pre-allocate.
    value: An `ndarray`, the shape of a batch of values in a single frame.

  Returns:
    A new, initialized Stack object.
  """
  batch_size = value.shape[0]
  stack_index = np.zeros([batch_size], dtype=np.int64)
  stack = np.zeros(
      (max_stack_depth * batch_size,) + value.shape[1:], dtype=value.dtype)
  return Stack(stack, stack_index)


class FullNumpyVariable(
    collections.namedtuple('FullNumpyVariable', ['current', 'stack'])):
  """A variable backed by a batched numpy "stack" with a cache for the top.

  The purpose of the cache is to make reads from and writes to the top
  of the stack cheaper than they would be otherwise.

  The namedtuple structure exposes the full state of the variable, and is useful
  for testing, passing through flatten/unflatten operations, and general
  symmetry with the TensorFlow backend.
  """

  def read(self, name=None):
    """Returns the batch of top values.

    Args:
      name: Optional name for the op.

    Return:
      val: Read of the current variable value.
    """
    del name
    return self.current

  def update(self, value, mask, name=None):
    """Updates this variable at the indicated places.

    Args:
      value: Array of shape `[batch_size, e1, ..., eE]` of data to update with.
        Indices in the first dimension corresponding to `False`
        entries in `mask` are ignored.
      mask: Boolean array of shape `[batch_size]`. The values at `True`
        indices of `mask` are updated; the others remain.
      name: Optional name for the op.

    Returns:
      var: Updated variable. Does not mutate `self`.
    """
    del name
    new_val = _where_leading_dims(mask, value, self.current)
    return FullNumpyVariable(new_val, self.stack)

  def push(self, mask, name=None):
    """Pushes each indicated batch member, making room for a new write.

    The new top value is the same as the old top value (this is a
    "duplicating push").

    Args:
      mask: Boolean array of shape `[batch_size]`. The values at `True`
        indices of `mask` are updated; the others remain.
      name: Optional name for the op.

    Returns:
      var: Updated variable. Does not mutate `self`.

    Raises:
      ValueError: If a push exceeds the maximum stack depth.
    """
    del name
    return FullNumpyVariable(self.current, self.stack.push(self.current, mask))

  def pop(self, mask, name=None):
    """Pops each indicated batch member, restoring a previous write.

    Args:
      mask: Boolean array of shape `[batch_size]`. The values at `True`
        indices of `mask` are updated; the others remain.
      name: Optional name for the op.

    Returns:
      var: Updated variable. Does not mutate `self`.

    Raises:
      ValueError: On an attempt to pop the last value off a batch member.
    """
    del name
    new_stack, stack_val = self.stack.pop(mask)
    new_val = _where_leading_dims(mask, stack_val, self.current)
    return FullNumpyVariable(new_val, new_stack)


def _where_leading_dims(mask, val1, val2):
  """Same as `np.where`, but broadcasting to rightmost dimensions."""
  desired_shape = [len(np.array(mask))] + [1] * (len(np.array(val2).shape) - 1)
  return np.where(np.reshape(mask, desired_shape), val1, val2)


class NumpyBackend(object):
  """Implements the Numpy backend ops for a PC auto-batching VM."""

  @property
  def variable_class(self):
    return (instructions.NullVariable,
            instructions.TemporaryVariable,
            RegisterNumpyVariable,
            FullNumpyVariable)

  def type_of(self, t, dtype_hint=None):
    """Returns the `instructions.Type` of `t`.

    Args:
      t: `np.ndarray` or a Python constant.
      dtype_hint: dtype to prefer, if `t` is a constant.

    Returns:
      vm_type: `instructions.TensorType` describing `t`
    """
    t = np.array(t, dtype=dtype_hint)
    return instructions.TensorType(t.dtype, t.shape)

  def run_on_dummies(
      self, primitive_callable, input_types):
    """Runs the given `primitive_callable` with dummy input.

    This is useful for examining the outputs for the purpose of type inference.

    Args:
      primitive_callable: A python callable.
      input_types: `list` of `instructions.Type` type of each argument to the
        callable.  Note that the contained `TensorType` objects must match the
        dimensions with which the primitive is to be invoked at runtime, even
        though type inference conventionally does not store the batch dimension
        in the `TensorType`s.

    Returns:
      outputs: pattern of backend-specific objects whose types may be
        analyzed by the caller with `type_of`.
    """
    def at_tensor(vt):
      return np.zeros(vt.shape, dtype=vt.dtype)
    inputs = [
        instructions.pattern_map(
            at_tensor, type_.tensors, leaf_type=instructions.TensorType)
        for type_ in input_types]
    return primitive_callable(*inputs)

  def merge_dtypes(self, dt1, dt2):
    """Merges two dtypes, returning a compatible dtype.

    Args:
      dt1: A numpy dtype, or None.
      dt2: A numpy dtype, or None.

    Returns:
      dtype: The more precise numpy dtype (e.g. prefers int64 over int32).
    """
    return (np.zeros([], dtype=dt1) + np.zeros([], dtype=dt2)).dtype

  def merge_shapes(self, s1, s2):
    """Merges two shapes, returning a broadcasted shape.

    Args:
      s1: A `list` of Python `int` or None.
      s2: A `list` of Python `int` or None.

    Returns:
      shape: A `list` of Python `int` or None.

    Raises:
      ValueError: If `s1` and `s2` are not broadcast compatible.
    """
    return (np.zeros(s1) + np.zeros(s2)).shape

  def assert_matching_dtype(self, expected_dtype, val, message=''):
    """Asserts that the dtype of `val` matches `expected_dtype`.

    Args:
      expected_dtype: A numpy dtype
      val: An object convertible to `np.array`
      message: Optional diagnostic message.

    Raises:
      ValueError: If dtype does not match.
    """
    if np.array(val).dtype != expected_dtype:
      raise ValueError('Mismatched dtype: expected {} found {}. {}'.format(
          expected_dtype, val.dtype, message))

  def batch_size(self, val, name=None):
    """Returns the first (batch) dimension of `val`."""
    del name
    if np.array(val).ndim:
      return np.array(val).shape[0]
    else:
      return 1

  def static_value(self, t):
    """Gets the eager/immediate value of `t`."""
    return t

  def fill(self, value, size, dtype, shape, name=None):
    """Fill a fresh batched Tensor of the given shape and dtype with `value`.

    Args:
      value: Scalar to fill with.
      size: Scalar `int` `Tensor` specifying the number of VM threads.
      dtype: `tf.DType` of the zeros to be returned.
      shape: Rank 1 `int` `Tensor`, the per-thread value shape.
      name: Optional name for the op.

    Returns:
      result: `Tensor` of `dtype` `value`s with shape `[size, *shape]`
    """
    del name
    return np.full(shape=[size] + shape, fill_value=value, dtype=dtype)

  def create_variable(self, name, alloc, type_, max_stack_depth, batch_size):
    """Returns an intialized Variable.

    Args:
      name: Name for the variable.
      alloc: `VariableAllocation` for the variable.
      type_: `instructions.TensorType` describing the sub-batch shape and dtype
        of the variable being created.
      max_stack_depth: Python `int`, the maximum stack depth to enforce.
      batch_size: Python `int`, the number of parallel threads being executed.

    Returns:
      var: A new, initialized Variable object.
    """
    del name
    if alloc is instructions.VariableAllocation.NULL:
      return instructions.NullVariable()
    elif alloc is instructions.VariableAllocation.TEMPORARY:
      return instructions.TemporaryVariable.empty()
    else:
      dtype, event_shape = type_
      value = np.zeros([batch_size] + list(event_shape), dtype=dtype)
      if alloc is instructions.VariableAllocation.REGISTER:
        return RegisterNumpyVariable(value)
      else:
        return FullNumpyVariable(value, _create_stack(max_stack_depth, value))

  def full_mask(self, size, name=None):
    """Returns an all-True mask `np.ndarray` with shape `[size]`."""
    del name
    return np.ones(size, dtype=np.bool_)

  def broadcast_to_shape_of(self, val, target, name=None):
    """Broadcasts val to the shape of target.

    Args:
      val: Python or Numpy array to be broadcast. Must be `np.array` compatible
        and broadcast-compatible with `target`.
      target: Python or Numpy array whose shape we broadcast `val` to match.
      name: Optional name for the op.

    Returns:
      broadcast_val: A `np.ndarray` with shape matching `val + target`. Provided
        that `val`'s dimension sizes are all smaller or equal to `target`'s, the
        returned value will be the shape of `target`.
    """
    del name
    val = np.array(val)
    return val + np.zeros_like(target, dtype=val.dtype)

  def cond(self, pred, true_fn, false_fn, name=None):
    """Implements a conditional operation for the backend.

    Args:
      pred: A Python or Numpy `bool` scalar indicating the condition.
      true_fn: A callable accepting and returning nests of `np.ndarray`s
        with the same structure as `state`, to be executed when `pred` is True.
      false_fn: A callable accepting and returning nests of `np.ndarray`s with
        the same structure as `state`, to be executed when `pred` is False.
      name: Optional name for the op.

    Returns:
      state: Output state, matching nest structure of input argument `state`.
    """
    del name
    if pred:
      return true_fn()
    else:
      return false_fn()

  def prepare_for_cond(self, state):
    """Backend hook for preparing Tensors for `cond`.

    Does nothing in the numpy backend (needed by the TensorFlow backend).

    Args:
      state: A state to be prepared for use in conditionals.

    Returns:
      state: The prepared state.
    """
    return state

  def where(self, condition, x, y, name=None):
    """Implements a where selector for the Numpy backend.

    Extends `tf.where` to support broadcasting of `on_false`.

    Args:
      condition: A `bool` `np.ndarray`, either a vector having length
        `y.shape[0]` or matching the full shape of `y`.
      x: `np.ndarray` of values to take when `condition` is `True`.
      y: `np.ndarray` of values to take when `condition` is `False`. May
        be smaller than `x`, as long as it is broadcast-compatible.
      name: Optional name for the op.

    Returns:
      masked: A `np.ndarray` where indices corresponding to `True` values in
        `condition` come from the corresponding value in `x`, and others come
        from `y`.
    """
    del name
    return _where_leading_dims(condition, x, y)

  def reduce_min(self, t, name=None):
    """Implements reduce_min for Numpy backend."""
    del name
    return np.min(t)

  def while_loop(self, cond, body, loop_vars, name=None):
    """Implements while loops for Numpy backend."""
    del name
    while cond(*loop_vars):
      loop_vars = body(*loop_vars)
    return loop_vars

  def switch_case(self, branch_selector, branch_callables, name=None):
    """Implements a switch (branch_selector) { case ... } construct."""
    del name
    return branch_callables[int(branch_selector)]()

  def equal(self, t1, t2, name=None):
    """Implements equality comparison for Numpy backend."""
    del name
    return np.equal(t1, t2)

  def not_equal(self, t1, t2, name=None):
    """Implements inequality comparison for Numpy backend."""
    del name
    return np.not_equal(t1, t2)

  def any(self, t, name=None):
    del name
    return np.any(t)

  def wrap_straightline_callable(self, f):
    """Method exists solely to be stubbed, i.e. for defun or XLA compile."""
    return f
