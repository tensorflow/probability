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
"""Test utility for implementations of auto-batch VM variables."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging


class ModelVariable(object):
  """A model of the expected behavior of a variable.

  The purpose of this model is to
  - Define, as simply as possible, the API and expected semantics of
    the Variable data structure.
  - Serve as a reference implementation for model-based testing
    of more sophisticated Variable data structures.

  Conceptually, a Variable is a batch of stacks, which admits masked
  reading and writing to the top element, and masked pushing and
  popping to save and restore previous writes. We model push as
  inserting an undefined value as the new top. This admits
  implementations that optimize push-pop sequences that do not have an
  interposing assignment.

  This model accepts and enforces a maximum stack depth, because I
  expect (some?) implementations that pre-allocate fixed-size arrays
  to hold their stacks.

  This model also chooses to require non-empty stacks, because that's
  convenient for the first implementation under test. It would have
  been equally valid to permit empty stacks, as long as they are not
  read from or written to before the next push.
  """

  def __init__(self, max_stack_depth, stacks):
    self._max_stack_depth = max_stack_depth
    self._stacks = stacks

  def read(self):
    """Returns the top value of every stack.

    A `None` value at an index means that result is not defined. This
    can happen when a `push` is followed by a `read`, because the
    pushes are not specified to be duplicating.

    Returns:
      results: A list of length `batch_size`. Each element is either
        `None`, indicating "undefined", or an array of shape `[...]`.
    """
    for s in self._stacks:
      assert s, "Internal invariant violation: empty stack detected."
    return [s[-1] for s in self._stacks]

  def update(self, value, mask):
    """Updates this variable at the indicated places.

    Args:
      value: Array of shape `[batch_size, ...]` of data to update with.
        Indices in the first dimension corresponding to `False`
        entries in `mask` are ignored.
      mask: Boolean array of shape `[batch_size]`. The values at `True`
        indices of `mask` are updated; the others remain.

    Returns:
      var: Updated variable. `self` may be mutated.

    Raises:
      ValueError: internal invariant violated
    """
    for s in self._stacks:
      if not s:
        raise ValueError("Internal invariant violation: empty stack detected.")
    for should_modify, val, stack in zip(mask, value, self._stacks):
      if should_modify:
        stack[-1] = val
    return self

  def push(self, mask):
    """Pushes each indicated stack, making room for a new write.

    The new top value is not defined (i.e., this need not be a
    "duplicating push").

    Args:
      mask: Boolean array of shape `[batch_size]`. The values at `True`
        indices of `mask` are updated; the others remain.

    Returns:
      var: Updated variable. `self` may be mutated.

    Raises:
      ValueError: If a push exceeds the maximum stack depth.
    """
    for should_modify, stack in zip(mask, self._stacks):
      if should_modify and len(stack) >= self._max_stack_depth:
        raise ValueError("Maximum stack depth exceeded.")
    for should_modify, stack in zip(mask, self._stacks):
      if should_modify:
        stack.append(None)
    return self

  def pop(self, mask):
    """Pops each indicated stack, restoring a previous write.

    Args:
      mask: Boolean array of shape `[batch_size]`. The values at `True`
        indices of `mask` are updated; the others remain.

    Returns:
      var: Updated variable. `self` may be mutated.

    Raises:
      ValueError: On an attempt to pop the last value off a stack.
    """
    for should_modify, stack in zip(mask, self._stacks):
      if should_modify and len(stack) <= 1:
        raise ValueError("Popping the last value off a stack.")
    for should_modify, stack in zip(mask, self._stacks):
      if should_modify:
        stack.pop()
    return self


def create_model_variable(max_stack_depth, value):
  """Returns a new ModelVariable.

  Args:
    max_stack_depth: int, Maximum stack depth to enforce.
    value: Initial value. The value is expected to already be
      batched; thus defines the batch size, dtype, and data shape.

  Returns:
    var: The new variable.

  Raises:
    ValueError: If `max_stack_depth` is less than 1, as that leaves
      no room for ensuring the stacks are non-empty.
  """
  if max_stack_depth < 1:
    template = "Cannot make a non-empty stack with maximum depth {}"
    raise ValueError(template.format(max_stack_depth))
  stacks = [[v] for v in value]
  return ModelVariable(max_stack_depth, stacks)


class VariableTestCase(object):
  """Test case that backend implementations can extend."""

  def check_same_results(self,
                         init,
                         ops,
                         test_var_initter,
                         to_numpy_arrays=lambda x: x,
                         exception_types=ValueError):
    """Check that the `test_var_type` gives the same results as `ModelVariable`.

    This is a model-based property test: For any initialization and
    any sequence of variable operations (read, update, push, pop), the
    implementation under test should behave the same as the model.

    By "behave the same", we mean that:

    - Neither the model nor the testee throw exceptions other than
      `ValueError`

    - Any operation that causes the model to throw `ValueError` also
      causes the testee to throw `ValueError` (alt. one of the errors in
      `exception_types`), and the testee does not throw otherwise

    - The result of reading from the model and from the testee is the
      same after any (sub-)sequence of operations, except where the
      model indicates that the result of a read is (partially)
      undefined.

    Args:
      init: Tuple of max_stack_depth and batched initial value
      ops: Sequence of operations to test. Each op is a tuple of a string,
          one of 'read', 'update', 'push', or 'pop', followed by the relevant
          number of arguments.
      test_var_initter: Callable taking init args (max_stack_depth,
        initial_value) and returning the a Variable object to test.
      to_numpy_arrays: Optional mapper from read results (or tuples thereof) to
          numpy `array`.
      exception_types: Exception classes to catch.
    """
    model_var = create_model_variable(*init)
    test_var = test_var_initter(*init)
    logging.info("Init: %s", init)

    def check_read_agrees():
      """Assert that the model var and the var under test agree on read."""
      model_res = model_var.read()
      test_res = to_numpy_arrays(test_var.read())
      assert len(model_res) == len(test_res)
      logging.vlog(1, "Internal state: %s", to_numpy_arrays(test_var))
      for i, (m, t) in enumerate(zip(model_res, test_res)):
        if m is not None:
          self.assertAllEqual(m, t, msg="Batch member {} disagrees".format(i))

    check_read_agrees()
    for op in ops:
      type_ = op[0]
      args = op[1:]
      logging.info("Executing %s: %s", type_, args)
      if type_ == "read":
        check_read_agrees()
      elif type_ == "update":
        assert len(args) == 2
        model_var = model_var.update(*args)
        test_var = test_var.update(*args)
        check_read_agrees()
      elif type_ == "push":
        assert len(args) == 1
        try:
          model_var = model_var.push(*args)
        except ValueError:
          logging.info("push fails")
          with self.assertRaises(exception_types):
            to_numpy_arrays(test_var.push(*args).read())
        else:  # If the model pushed without raising
          test_var = test_var.push(*args)
        check_read_agrees()
      elif type_ == "pop":
        assert len(args) == 1
        try:
          model_var = model_var.pop(*args)
        except ValueError:
          logging.info("pop fails")
          with self.assertRaises(exception_types):
            to_numpy_arrays(test_var.pop(*args).read())
        else:  # If the model popped without raising
          test_var = test_var.pop(*args)
        check_read_agrees()
      else:
        assert False, "Invalid op type"
