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
"""TensorFlow (graph) backend for auto-batching VM.

Implements VM variable stack and registers backed by TF `Tensor`s.
"""

import collections
import contextlib

# Dependency imports
import six
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.experimental.auto_batching import instructions
from tensorflow_probability.python.experimental.auto_batching import xla
from tensorflow_probability.python.internal import dtype_util
from tensorflow.python.ops import control_flow_util  # pylint: disable=g-direct-tensorflow-import

__all__ = ['TensorFlowBackend']


@contextlib.contextmanager
def _control_flow_v2():
  enable_control_flow_v2_old = control_flow_util.ENABLE_CONTROL_FLOW_V2
  control_flow_util.ENABLE_CONTROL_FLOW_V2 = True
  try:
    yield
  finally:
    control_flow_util.ENABLE_CONTROL_FLOW_V2 = enable_control_flow_v2_old


def _generalized_where(mask, value, old_value):
  """Version of tf1.where that broadcasts `value` to `old_value`."""
  mask = tf.convert_to_tensor(value=mask, name='mask')
  mask.shape.assert_has_rank(1)
  value = tf.convert_to_tensor(value=value, name='value', dtype=old_value.dtype)
  if (not value.shape.is_fully_defined() or
      not old_value.shape.is_fully_defined() or
      value.shape != old_value.shape):
    # We force broadcast value w/ current, e.g. for program constants.
    if old_value.dtype == tf.bool:
      value |= tf.zeros_like(old_value)
    else:
      value += tf.zeros_like(old_value)
  new_value = tf1.where(mask, value, old_value, name='new_value')
  # TODO(b/78655271): Do we need 'new_val.set_shape(old_value.shape)'?
  return new_value


class RegisterTensorFlowVariable(collections.namedtuple(
    'RegisterTensorFlowVariable', ['value'])):
  """A register-only variable.

  Efficiently stores and updates values whose lifetime does not cross function
  calls (and therefore does not require a stack).  This is different from
  `TemporaryVariable` because it supports crossing basic block boundaries.  A
  `RegisterTensorFlowVariable` therefore needs to store its content persistently
  across the `while_loop` in `execute`, and to handle divergence (and
  re-convergence) of logical threads.
  """

  def update(self, value, mask):
    """Update with `value` at `mask`, propagate other positions."""
    if isinstance(self.value, tuple):
      # Support fast path for Eager mode initialization.  Initializing with a
      # well-formed value is only necessary in graph mode, where the value
      # Tensor needs to be part of while-carried state.  In Eager, however, it
      # does not, so the variable may just carry its type information as a
      # Python tuple.
      batch_size, dtype, event_shape = self.value
      value = tf.convert_to_tensor(value=value, dtype=dtype)
      new_value = tf.broadcast_to(value, shape=[batch_size] + list(event_shape))
    else:
      new_value = _generalized_where(mask, value, self.value)
    return type(self)(new_value)

  def push(self, mask):
    del mask
    return self

  def read(self):
    if isinstance(self.value, tuple):
      raise ValueError(
          'Accessing uninitialized variable {}'.format(self._name()))
    return self.value

  def pop(self, mask):
    del mask
    return self

  def ensure_initialized(self):
    if isinstance(self.value, tuple):
      return self.update(False, None)
    return self


class Stack(collections.namedtuple('Stack', ['stack', 'stack_index'])):
  """Immutable, internal container for a fixed size stack.

  The implementation is backed by a `Tensor` each for the stack and the
  (batched) stack pointer.

  As a namedtuple, it can be directly passed through TF's nest library for
  flattening and restructuring as an element passed to e.g. a TF while loop.
  """

  def _safety_checks(self):
    """Put in runtime asserts of stack bounds? Overridden by UnsafeStack."""
    return True

  def pop(self, mask, name=None):
    """Pops each indicated batch member, returns the new top of the stack.

    Does not mutate `self`.

    Args:
      mask: Boolean `Tensor` of shape `[batch_size]`. The stack frames at `True`
        indices of `mask` are regressed; the others are unchanged.
      name: Optional name for this op.

    Returns:
      new_stack: A new stack whose frames have been regressed where indicated
          by `mask`.
      read: The batch of values at the newly-current stack frame.
    """
    with tf.name_scope(name or 'Stack.pop'):
      mask = tf.convert_to_tensor(value=mask, name='mask')
      new_stack_index = self.stack_index - tf.cast(mask, self.stack_index.dtype)
      if self._safety_checks():
        with tf.control_dependencies(
            [tf1.assert_greater_equal(
                new_stack_index, tf.constant(0, new_stack_index.dtype))]):
          new_stack_index = tf.identity(new_stack_index)
      new_stack_index.set_shape(self.stack_index.shape)
      # self.stack:      [max_stack_depth * batch_size, ...]
      # self.stack_index:                  [batch_size]
      # returned:                          [batch_size, ...]
      batch_size = (
          tf.compat.dimension_value(self.stack_index.shape[0]) or
          tf.shape(input=self.stack_index, out_type=self.stack_index.dtype)[0])
      # Note that stack depth and batch are in a single dimension, stack major.
      gather_indices = (
          new_stack_index * batch_size + tf.range(
              batch_size, dtype=new_stack_index.dtype))
      read_value = tf.gather(self.stack, gather_indices)
      read_value.set_shape(
          self.stack_index.shape.concatenate(self.stack.shape[1:]))
      return type(self)(self.stack, new_stack_index), read_value

  def push(self, value, mask, name=None):
    """Pushes `value` onto the stack, advances frame of batch members in `mask`.

    In this impl, we update each thread's top-of-stack (regardless of `mask`) to
    the corresponding `value`, then advance the stack pointers of only those
    threads indicated by `mask`.

    Args:
      value: `Tensor` having the shape of a single batch of the variable.
      mask: Boolean `Tensor` of shape `[batch_size]`. Threads at `True` indices
          of `mask` have their stack frames advanced; the others remain.
      name: Optional name for this op.

    Returns:
      stack: Updated stack. Does not mutate `self`.
      asserted_value: A assertion-bound snapshot of the input `value`,
          assertions used to catch stack overflows.
    """
    with tf.name_scope(name or 'Stack.push'):
      value = tf.convert_to_tensor(value=value, name='value')
      mask = tf.convert_to_tensor(value=mask, name='mask')
      # self.stack:       [max_stack_depth * batch_size, ...]
      # self.stack_index:                   [batch_size]
      # value:                              [batch_size, ...]
      batch_size = (
          tf.compat.dimension_value(self.stack_index.shape[0]) or
          tf.shape(input=self.stack_index)[0])
      max_stack_depth = (tf.compat.dimension_value(self.stack.shape[0]) or
                         tf.shape(input=self.stack)[0]) // batch_size
      max_stack_depth_tensor = tf.convert_to_tensor(value=max_stack_depth)
      tiled_value = tf.tile(
          input=value[tf.newaxis, ...],
          multiples=tf.concat(
              [[max_stack_depth_tensor],
               tf.ones(tf.rank(value), dtype=max_stack_depth_tensor.dtype)],
              axis=0))
      update_stack_mask = tf.one_hot(
          self.stack_index,
          depth=max_stack_depth,
          axis=0,  # Stack depth x batch are both in outermost dim, stack major.
          on_value=True,
          off_value=False,
          dtype=tf.bool)
      new_stack = tf1.where(
          tf.reshape(update_stack_mask, [-1]),
          tf.reshape(tiled_value, tf.shape(input=self.stack)), self.stack)
      new_stack.set_shape(self.stack.shape)
      new_stack_index = self.stack_index + tf.cast(mask, self.stack_index.dtype)
      new_stack_index.set_shape(self.stack_index.shape)
      if self._safety_checks():
        with tf.control_dependencies(
            [tf1.assert_less(
                new_stack_index, tf.cast(
                    max_stack_depth_tensor, new_stack_index.dtype))]):
          value = tf.identity(value)
          new_stack_index = tf.identity(new_stack_index)
      return type(self)(new_stack, new_stack_index), value


class UnsafeStack(Stack):
  """Stack with runtime assertions disabled."""

  def _safety_checks(self):
    return False


def _create_stack(max_stack_depth, value, safety_checks=True, name=None):
  """Creates a new Stack instance.

  Args:
    max_stack_depth: A scalar int `Tensor` indicating the depth of stack we
        should pre-allocate.
    value: A batched `Tensor` giving the shape of a batch of values in a
        single stack frame.
    safety_checks: Python `bool` indicating whether we must use runtime
      assertions to detect stack overflow/underflow.
    name: Optional name for this op.

  Returns:
    stack: An initialized Stack object.
  """
  with tf.name_scope(name or 'Stack.initialize'):
    value = tf.convert_to_tensor(value=value, name='value')
    batch_size = _get_leftmost_dim_size(value)
    # Home the stack index in the same memory space as the value.  The
    # convention on GPU is that int32 are in host memory and int64 are in device
    # memory.
    stack_index_dtype = tf.int64 if value.dtype != tf.int32 else tf.int32
    stack_index = tf.zeros(
        [batch_size], dtype=stack_index_dtype, name='stack_index')
    stack = tf.zeros(
        shape=tf.concat([[max_stack_depth * batch_size],
                         tf.shape(input=value)[1:]],
                        axis=0),
        dtype=value.dtype,
        name='stack')
    stack_class = Stack if safety_checks else UnsafeStack
    return stack_class(stack, stack_index)


class FullTensorFlowVariable(
    collections.namedtuple('FullTensorFlowVariable', ['current', 'stack'])):
  """An immutable register + stack backed by batched TF `Tensor`s.

  All state-changing methods return new Variable instances.

  The register is used to make reads from and writes to the top of the stack
  cheaper than they would be otherwise, i.e. save slice updates.

  As a namedtuple, the variable can be passed through the TF nest library as
  part of the structure handed to/returned from the body of a while_loop, or
  even a Session.run call.
  """

  def _name(self):
    """The variable's name. Overridden by `NamedVariable` in create_variable."""
    return 'Variable'

  def read(self, name=None):
    """Returns the batch of top values."""
    with tf.name_scope(name or '{}.read'.format(self._name())):
      return tf.identity(self.current)

  def update(self, value, mask, name=None):
    """Updates the variable at the indicated places.

    Args:
      value: Array of shape `[batch_size, ...]` of data to update with.
        Indices in the first dimension corresponding to `False`
        entries in `mask` are ignored.
      mask: Boolean array of shape `[batch_size]`. The values at `True`
        indices of `mask` are updated; the others remain.
      name: Optional name for this op.

    Returns:
      var: Updated variable. Does not mutate `self`.
    """
    with tf.name_scope(name or '{}.update'.format(self._name())):
      new_value = _generalized_where(mask, value, self.current)
      return type(self)(new_value, self.stack)

  def push(self, mask, name=None):
    """Pushes each indicated batch member, making room for a new write.

    The new top value is the same as the old top value (this is a
    "duplicating push").

    Args:
      mask: Boolean array of shape `[batch_size]`. The values at `True`
        indices of `mask` are updated; the others remain.
      name: Optional name for this op.

    Returns:
      var: Updated variable. Does not mutate `self`.
    """
    with tf.name_scope(name or '{}.push'.format(self._name())):
      new_stack, asserted_value = self.stack.push(self.current, mask)
      return type(self)(asserted_value, new_stack)

  def pop(self, mask, name=None):
    """Pops each indicated batch member, restoring a previous write.

    Args:
      mask: Boolean `Tensor` of shape `[batch_size]`. The values at `True`
        indices of `mask` are updated; the others are unchanged.
      name: Optional name for this op.

    Returns:
      var: Updated variable. Does not mutate `self`.
    """
    with tf.name_scope(name or '{}.pop'.format(self._name())):
      mask = tf.convert_to_tensor(value=mask, name='mask')
      new_stack, stack_value = self.stack.pop(mask)
      new_value = tf1.where(
          mask, stack_value, self.current, name='new_value')
      return type(self)(new_value, new_stack)


class TensorFlowBackend(object):
  """Implements the TF backend ops for a PC auto-batching VM."""

  def __init__(self,
               safety_checks=True,
               while_parallel_iterations=10,
               while_maximum_iterations=None,
               basic_block_xla_device=None):
    """Construct a new backend instance.

    Args:
      safety_checks: Python `bool` indicating whether we should use runtime
        assertions to detect stack overflow/underflow.
      while_parallel_iterations: Python `int`, the argument to pass along to
        `tf.while_loop(..., parallel_iterations=while_parallel_iterations)`
      while_maximum_iterations: Python `int` or None, the argument to pass along
        to `tf.while_loop(..., maximum_iterations=while_maximum_iterations)`
      basic_block_xla_device: Python `str` indicating the device to which basic
        blocks should be targeted (i.e. 'CPU:0' or 'GPU:0'); if not None.
    """
    self._safety_checks = safety_checks
    self._while_parallel_iterations = while_parallel_iterations
    self._while_maximum_iterations = while_maximum_iterations
    self._basic_block_xla_device = basic_block_xla_device

  @property
  def variable_class(self):
    return (instructions.NullVariable,
            instructions.TemporaryVariable,
            RegisterTensorFlowVariable,
            FullTensorFlowVariable)

  def type_of(self, t, dtype_hint=None):
    """Returns the `instructions.Type` of `t`.

    Args:
      t: `tf.Tensor` or a Python or numpy constant.
      dtype_hint: dtype to prefer, if `t` is a constant.

    Returns:
      vm_type: `instructions.TensorType` describing `t`.
    """
    if tf.executing_eagerly():
      new_t = tf.convert_to_tensor(value=t, dtype=dtype_hint)
    else:
      with tf.Graph().as_default():  # Use a scratch graph.
        new_t = tf.convert_to_tensor(value=t, dtype=dtype_hint)
    dtype = new_t.dtype.base_dtype.as_numpy_dtype
    shape = None if new_t.shape.ndims is None else tuple(new_t.shape.as_list())
    return instructions.TensorType(dtype, shape)

  def run_on_dummies(self, primitive_callable, input_types):
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
    with tf.name_scope('VM.run_on_dummies'):
      # We cannot use a temporary graph in eager mode because user code may
      # close over eager tensors, causing `RuntimeError: Attempting to capture
      # an EagerTensor without building a function.`
      # In graph mode, capturing user Tensors has also been a problem, because
      # TF doesn't like the inputs of an op being in different graphs.
      # Status quo is unfortunate because it involves running the computation
      # in the primop to determine its shape behavior, instead of just invoking
      # shape inference.
      # There may be a solution involving FuncGraph; see b/118896442.
      def mk_placeholder(vt):
        return tf.ones(vt.shape, dtype=vt.dtype)
      phs = [
          instructions.pattern_map(
              mk_placeholder, vtype.tensors, leaf_type=instructions.TensorType)
          for vtype in input_types]
      return primitive_callable(*phs)

  def merge_dtypes(self, dt1, dt2):
    """Merges two dtypes, returning a compatible dtype.

    In practice, TF implementation asserts that the two dtypes are identical.

    Args:
      dt1: A numpy dtype, or None.
      dt2: A numpy dtype, or None.

    Returns:
      dtype: The common numpy dtype.

    Raises:
      ValueError: If dt1 and dt2 are not equal and both are non-`None`.
    """
    if None in (dt1, dt2):
      return dt1 or dt2
    if tf.as_dtype(dt1) == tf.as_dtype(dt2):
      return dtype_util.as_numpy_dtype(tf.as_dtype(dt1))
    raise ValueError('Mismatched dtypes {} vs {}'.format(dt1, dt2))

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
    new_shp = tf.broadcast_static_shape(
        tf.TensorShape(s1), tf.TensorShape(s2))
    return None if new_shp.ndims is None else tuple(new_shp.as_list())

  def assert_matching_dtype(self, expected_dtype, value, message=''):
    """Asserts that the dtype of `value` matches `expected_dtype`.

    Args:
      expected_dtype: A numpy dtype
      value: `Tensor` or convertible.
      message: Optional diagnostic message.

    Raises:
      ValueError: If dtype does not match.
    """
    with tf.name_scope('VM.assert_matching_dtype'):
      value = tf.convert_to_tensor(
          value=value, name='value', dtype=expected_dtype)
      if value.dtype.base_dtype.as_numpy_dtype != expected_dtype:
        raise ValueError('Mismatched dtype: expected {} found {}. {}'.format(
            expected_dtype, value.dtype.base_dtype.as_numpy_dtype, message))

  def batch_size(self, value, name=None):
    """Returns the first (batch) dimension of `value`."""
    with tf.name_scope(name or 'VM.batch_size'):
      value = tf.convert_to_tensor(value=value, name='value')
      return _get_leftmost_dim_size(value)

  def static_value(self, t):
    """Gets the eager/immediate value of `t`, or `None` if `t` is a Tensor."""
    if tf.executing_eagerly():
      return t.numpy()
    return None

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
    with tf.name_scope(name or 'VM.fill'):
      size = tf.convert_to_tensor(value=size, name='size')
      shape = tf.convert_to_tensor(value=shape, name='shape', dtype=size.dtype)
      return tf.fill(tf.concat([[size], shape], axis=0),
                     value=tf.cast(value, dtype=dtype))

  def create_variable(self, name, alloc, type_, max_stack_depth, batch_size):
    """Returns an intialized Variable.

    Args:
      name: Name for the variable.
      alloc: `VariableAllocation` for the variable.
      type_: `instructions.TensorType` describing the sub-batch shape and dtype
        of the variable being created.
      max_stack_depth: Scalar `int` `Tensor`, the maximum stack depth allocated.
      batch_size: Scalar `int` `Tensor`, the number of parallel threads being
        executed.

    Returns:
      var: A new, initialized Variable object.
    """
    if alloc is instructions.VariableAllocation.NULL:
      return instructions.NullVariable()
    elif alloc is instructions.VariableAllocation.TEMPORARY:
      return instructions.TemporaryVariable.empty()
    else:
      name = 'Variable' if name is None else 'VM.var_{}'.format(name)
      dtype, event_shape = type_

      with tf.name_scope('{}.initialize'.format(name)):
        if (alloc is instructions.VariableAllocation.REGISTER and
            tf.executing_eagerly()):
          # Don't need to construct the empty value in Eager mode, because there
          # is no tf.while_loop whose loop-carried state it would need to be.
          # This is a substantial optimization for stackless mode, because that
          # initializes variables on every function call, rather than just once.
          value = (batch_size, dtype, event_shape)
        else:
          value = self.fill(0, batch_size, dtype, event_shape)

        if alloc is instructions.VariableAllocation.REGISTER:
          klass = RegisterTensorFlowVariable
          extra = []
        else:
          klass = FullTensorFlowVariable
          extra = [_create_stack(max_stack_depth, value, self._safety_checks)]

        class NamedVariable(klass):
          """Captures `name` to yield improved downstream TF op names."""

          def _name(self):
            return name

        return NamedVariable(value, *extra)

  def full_mask(self, size, name=None):
    """Returns an all-True mask `Tensor` with shape `[size]`."""
    with tf.name_scope(name or 'VM.full_mask'):
      size = tf.convert_to_tensor(value=size, name='size')
      return tf.ones(size, dtype=tf.bool)

  def broadcast_to_shape_of(self, val, target, name=None):
    """Broadcasts val to the shape of target.

    Attempts to match the dtype of `broadcast_val` to the dtype of `target`, if
    `val` is not a `Tensor` and `target` has a dtype.

    Args:
      val: The value to be broadcast. Must be broadcast-compatible with
        `target`.
      target: `Tensor` whose shape we will broadcast `val` to match.
      name: Optional name for the op.

    Returns:
      broadcast_val: A `Tensor` with shape matching `val + target`. Provided
        that `val`'s dimension sizes are all smaller or equal to `target`'s, the
        returned value will be the shape of `target`.
    """
    # TODO(b/78594182): This is a compatibility shim, required because
    # `tf1.where` does not support broadcasting of its value operands.
    with tf.name_scope(name or 'VM.broadcast_to_shape_of'):
      dtype = getattr(target, 'dtype', getattr(val, 'dtype', None))
      target = tf.convert_to_tensor(value=target, name='target', dtype=dtype)
      val = tf.convert_to_tensor(value=val, name='val', dtype=target.dtype)
      if val.dtype == tf.bool:
        return val | tf.zeros_like(target, dtype=val.dtype)
      return val + tf.zeros_like(target, dtype=val.dtype)

  def cond(self, pred, true_fn, false_fn, name=None):
    """Implements a conditional operation for the backend.

    Args:
      pred: A boolean scalar `Tensor` indicating the condition.
      true_fn: A callable accepting and returning nests of `Tensor`s having
        the same structure as `state`, to be executed when `pred` is True.
      false_fn: A callable accepting and returning nests of `Tensor`s having
        the same structure as `state`, to be executed when `pred` is False.
      name: Optional name for the op.

    Returns:
      state: Output state, matching nest structure of input argument `state`.
    """
    with tf.name_scope(name or 'VM.cond'):
      with _control_flow_v2():
        return tf.cond(pred=pred, true_fn=true_fn, false_fn=false_fn)

  def prepare_for_cond(self, state):
    """Backend hook for preparing Tensors for `cond`.

    The TensorFlow backend uses this hook to apply `tf.convert_to_tensor` before
    entering the cond tree generated by `virtual_machine._staged_apply`.  One
    could do this inside `cond`, but when this API element was defined there
    seemed to be a performance reason (for Eager mode) to do it once per cond
    tree rather than once per cond.

    Args:
      state: A state to be prepared for use in conditionals.

    Returns:
      state: The prepared state.
    """
    if tf.executing_eagerly():
      # Eager doesn't need to pre-wrap the cond-carried state at all.  Also, in
      # Eager, lazy initialization for register variables means that the state
      # may not always be correct to convert to a Tensor.
      return state
    with tf.name_scope('VM.prepare_for_cond'):
      state_flat = [tf.convert_to_tensor(value=x)
                    for x in tf.nest.flatten(state)]
      return tf.nest.pack_sequence_as(state, state_flat)

  def where(self, condition, x, y, name=None):
    """Implements a where selector for the TF backend.

    Attempts to match the dtypes of the value operands, if they are not yet both
    `Tensor`s.

    Args:
      condition: A boolean `Tensor`, either a vector having length
        `(x + y).shape[0]` or matching the full shape of `x + y`.
      x: `Tensor` of values to take when `condition` is `True`. Shape must match
        that of `y`.
      y: `Tensor` of values to take when `condition` is `False`. Shape must
        match that of `x`.
      name: Optional name for the op.

    Returns:
      masked: A broadcast-shaped `Tensor` where elements corresponding to `True`
        values of `condition` come from `x`, and others come from `y`.
    """
    with tf.name_scope(name or 'VM.where'):
      condition = tf.convert_to_tensor(value=condition, name='condition')
      dtype = getattr(x, 'dtype', getattr(y, 'dtype', None))
      x = tf.convert_to_tensor(value=x, name='x', dtype=dtype)
      y = tf.convert_to_tensor(value=y, name='y', dtype=x.dtype)
      return tf1.where(condition, x, y)

  def reduce_min(self, t, name=None):
    """Implements reduce_min for TF backend."""
    with tf.name_scope('VM.reduce_min'):
      return tf.reduce_min(input_tensor=t, name=name)

  def while_loop(self, cond, body, loop_vars, name=None):
    """Implements while loops for TF backend."""
    with tf.name_scope('VM.while_loop'):
      if tf.executing_eagerly():
        # The reg. variable optimization (see create_variable) may change loop
        # structure across iterations, which now triggers an exception for eager
        # tf.while_loop.
        while cond(*loop_vars):
          loop_vars = body(*loop_vars)
        return loop_vars
      with _control_flow_v2():
        return tf.while_loop(
            cond=cond,
            body=body,
            loop_vars=loop_vars,
            back_prop=False,
            name=name,
            parallel_iterations=self._while_parallel_iterations,
            maximum_iterations=self._while_maximum_iterations)

  def switch_case(self, branch_selector, branch_callables, name=None):
    """Implements a switch (branch_selector) { case ... } construct."""
    with tf.name_scope('VM.switch_case'):
      with _control_flow_v2():
        return tf.switch_case(branch_selector, branch_callables, name=name)

  def equal(self, t1, t2, name=None):
    """Implements equality comparison for TF backend."""
    with tf.name_scope('VM.equal'):
      return tf.equal(t1, t2, name=name)

  def not_equal(self, t1, t2, name=None):
    """Implements inequality comparison for TF backend."""
    with tf.name_scope('VM.not_equal'):
      return tf.not_equal(t1, t2, name=name)

  def any(self, t, name=None):
    with tf.name_scope(name or 'VM.any'):
      return tf.reduce_any(input_tensor=t)

  def wrap_straightline_callable(self, f):
    """Method exists solely to be stubbed, i.e. for defun + XLA compile."""
    if self._basic_block_xla_device is None:
      return f

    @tf.function
    def _f(*args):
      with tf.device(self._basic_block_xla_device):
        return xla.compile_nested_output(
            f, tf.xla.experimental.compile)(*args)

    def _ensure_regvars_initialized(t):
      if isinstance(t, RegisterTensorFlowVariable):
        return t.ensure_initialized()
      return t

    def _init_f(env_dict, *args):
      """A RegisterTensorFlowVariable-initializing wrapper of `_f`."""
      # We ensure RegisterTensorFlowVariable instances have a Tensor value when
      # using XLA and/or defun. Otherwise, we will trigger cache misses on the
      # tfe.defun or get issues around "Cannot convert object of type [dtype] to
      # a Tensor" (XLA). This corresponds with the optimization in
      # `create_variable` conditioned on Eager & VariableAllocation.REGISTER.
      env_dict = dict({k: instructions.pattern_map(
          _ensure_regvars_initialized, v, leaf_type=RegisterTensorFlowVariable)
                       for k, v in six.iteritems(env_dict)})
      return _f(env_dict, *args)

    return _init_f


def _get_leftmost_dim_size(x, name=None):
  """Returns the size of the left most dimension, statically if possible."""
  with tf.name_scope(name or 'get_leftmost_dim_size'):
    x = tf.convert_to_tensor(value=x, name='x')
    if x.shape.ndims is None:
      # If tf.shape(x) is scalar, the [:1] will produce the empty list, whose
      # reduce_prod is 1 as desired.  Otherwise, the [:1] will select the first
      # dimension, and reduce_prod will not alter it.
      return tf.reduce_prod(input_tensor=tf.shape(input=x)[:1])
    if x.shape.ndims == 0:
      return 1
    leftmost = tf.compat.dimension_value(x.shape[0])
    return leftmost if leftmost is not None else tf.shape(input=x)[0]
