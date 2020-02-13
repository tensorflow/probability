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
"""Tests of the FunctionCallOp lowering compilation pass."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

# Dependency imports

from absl.testing import parameterized

import numpy as np
import six
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.experimental.auto_batching import allocation_strategy
from tensorflow_probability.python.experimental.auto_batching import instructions
from tensorflow_probability.python.experimental.auto_batching import lowering
from tensorflow_probability.python.experimental.auto_batching import numpy_backend
from tensorflow_probability.python.experimental.auto_batching import test_programs
from tensorflow_probability.python.experimental.auto_batching import tf_backend
from tensorflow_probability.python.experimental.auto_batching import type_inference
from tensorflow_probability.python.experimental.auto_batching import virtual_machine as vm
from tensorflow_probability.python.internal import test_util

NP_BACKEND = numpy_backend.NumpyBackend()

TF_BACKEND = tf_backend.TensorFlowBackend()


def shape_sequence_program(shape_sequence):
  """Program that writes into `answer` zeros having a sequence of shapes.

  This enables us to test that the final inferred shape is the broadcast of all
  intermediate shapes.

  Args:
    shape_sequence: The sequence of intermediate shapes.

  Returns:
    program: `instructions.Program` which returns an arbitrary value.
  """
  block_ops = []
  def op(shape, ans):
    return np.zeros(shape, dtype=np.array(ans).dtype),
  for shape in shape_sequence:
    # We use a partial instead of a lambda in order to capture a copy of shape.
    block_ops.append(
        instructions.prim_op(['ans'], ['ans'], functools.partial(op, shape)))
  shape_seq_block = instructions.Block(block_ops, instructions.halt_op())
  shape_seq_vars = {
      'ans': instructions.Type(None),
      instructions.pc_var: instructions.single_type(np.int64, ()),
  }
  return instructions.Program(
      instructions.ControlFlowGraph([shape_seq_block]), [],
      shape_seq_vars, ['ans'], ['ans'])


def _execute(prog, inputs, stack_depth, backend):
  return vm.execute(
      prog, [inputs], max_stack_depth=stack_depth, backend=backend)


@test_util.test_all_tf_execution_regimes
class TypeInferenceTest(test_util.TestCase):

  def assertSameTypes(self, expected_prog, typed, check_dtypes=True):
    for v, type_ in six.iteritems(typed.var_defs):
      for expected_type, got_type in instructions.pattern_zip(
          expected_prog.var_defs[v].tensors, type_.tensors,
          leaf_type=instructions.TensorType):
        if check_dtypes:
          self.assertEqual(expected_type.dtype, got_type.dtype)
        self.assertEqual(expected_type.shape, got_type.shape)

  @parameterized.parameters(np.int32, np.int64, np.float32, np.float64)
  def testShapeSequenceInferenceNumpy(self, dtype):
    shape_seq = [(1, 1, 3, 1), (1, 1, 1, 2), (1, 1, 3, 2), (1, 5, 1, 1)]
    init_val = np.zeros([1], dtype=dtype)
    prog = shape_sequence_program(shape_seq)
    typed = type_inference.infer_types(prog, [init_val], NP_BACKEND)
    self.assertEqual({instructions.pc_var, 'ans'},
                     set(typed.var_defs.keys()))
    self.assertEqual(dtype, typed.var_defs['ans'].tensors.dtype)
    # Note: the shapes used by the primop include the batch dimension, but the
    # returned type does not.
    self.assertEqual((5, 3, 2), typed.var_defs['ans'].tensors.shape)

  def testProductTypeInferenceNumpy(self):
    inputs = np.array([4, 5], dtype=np.int64)
    outputs = np.array(([6, 7], [7, 8]), dtype=np.int64)
    prog = test_programs.synthetic_pattern_variable_program(include_types=False)
    typed = type_inference.infer_types(prog, [inputs], NP_BACKEND)
    expected_prog = test_programs.synthetic_pattern_variable_program()
    self.assertSameTypes(expected_prog, typed)
    alloc = allocation_strategy.optimize(typed)
    lowered = lowering.lower_function_calls(alloc)
    self.assertAllEqual(
        outputs, _execute(lowered, inputs, 15, NP_BACKEND))

  def testProductTypeInferenceTF(self):
    inputs = np.array([4, 5], dtype=np.int64)
    outputs = np.array(([6, 7], [7, 8]), dtype=np.int64)
    inputs_t = tf.constant(inputs, dtype=np.int64)
    prog = test_programs.synthetic_pattern_variable_program(include_types=False)
    typed = type_inference.infer_types(prog, [inputs_t], TF_BACKEND)
    expected_prog = test_programs.synthetic_pattern_variable_program()
    self.assertSameTypes(expected_prog, typed)
    alloc = allocation_strategy.optimize(typed)
    lowered = lowering.lower_function_calls(alloc)
    self.assertAllEqual(
        outputs, self.evaluate(_execute(lowered, inputs_t, 15, TF_BACKEND)))

  @parameterized.parameters(np.int32, np.int64, np.float32, np.float64)
  def testFibonacciTypeInferenceNumpy(self, dtype):
    for inputs, outputs in ([5], [8]), ([5, 6, 8, 9], [8, 13, 34, 55]):
      inputs = np.array(inputs, dtype=dtype)
      outputs = np.array(outputs, dtype=dtype)
      tf1.logging.debug('np.fib {} {} {}'.format(
          dtype, inputs.shape, outputs.shape))
      prog = test_programs.fibonacci_function_calls(include_types=False)
      typed = type_inference.infer_types(prog, [inputs], NP_BACKEND)
      expected_prog = test_programs.fibonacci_function_calls(dtype=dtype)
      # We can only assert on the int64/float64 cases because numpy does
      # not match-cast types on arithmetic with constants.
      # i.e. (np.int32(0) - 1).dtype == np.int64
      self.assertSameTypes(
          expected_prog, typed, check_dtypes=dtype(0).nbytes == 8)
      alloc = allocation_strategy.optimize(typed)
      lowered = lowering.lower_function_calls(alloc)
      self.assertAllEqual(
          outputs, _execute(lowered, inputs, 15, NP_BACKEND))

  @parameterized.parameters(np.int32, np.int64, np.float32, np.float64)
  def testFibonacciTypeInferenceTF(self, dtype):
    for inputs, outputs in ([5], [8]), ([5, 6, 8, 9], [8, 13, 34, 55]):
      inputs = np.array(inputs, dtype=dtype)
      outputs = np.array(outputs, dtype=dtype)
      tf1.logging.debug('tf.fib {} {} {}'.format(
          dtype, inputs.shape, outputs.shape))
      inputs_t = tf.constant(inputs, dtype=dtype)
      prog = test_programs.fibonacci_function_calls(include_types=False)
      typed = type_inference.infer_types(prog, [inputs_t], TF_BACKEND)
      expected_prog = test_programs.fibonacci_function_calls(dtype=dtype)
      self.assertSameTypes(expected_prog, typed)
      alloc = allocation_strategy.optimize(typed)
      lowered = lowering.lower_function_calls(alloc)
      self.assertAllEqual(
          outputs, self.evaluate(_execute(lowered, inputs_t, 15, TF_BACKEND)))

  @parameterized.parameters(np.int32, np.int64, np.float32, np.float64)
  def testIsEvenTypeInferenceNumpy(self, dtype):
    for inputs, outputs in [([1], [False]),
                            ([5, 6, 0, 3], [False, True, True, False])]:
      inputs = np.array(inputs, dtype=dtype)
      outputs = np.array(outputs, dtype=np.bool)
      tf1.logging.debug('np.even {} {} {}'.format(
          dtype, inputs.shape, outputs.shape))
      prog = test_programs.is_even_function_calls(include_types=False)
      typed = type_inference.infer_types(prog, [inputs], NP_BACKEND)
      expected_prog = test_programs.is_even_function_calls(dtype=dtype)
      # We can only assert on the int64/float64 cases because numpy does
      # not match-cast types on arithmetic with constants.
      # i.e. (np.int32(0) - 1).dtype == np.int64
      self.assertSameTypes(
          expected_prog, typed, check_dtypes=dtype(0).nbytes == 8)
      alloc = allocation_strategy.optimize(typed)
      lowered = lowering.lower_function_calls(alloc)
      self.assertAllEqual(
          outputs,
          _execute(lowered, inputs, int(np.max(inputs)) + 3, NP_BACKEND))

  @parameterized.parameters(np.int32, np.int64, np.float32, np.float64)
  def testIsEvenTypeInferenceTF(self, dtype):
    for inputs, outputs in [([1], [False]),
                            ([5, 6, 0, 3], [False, True, True, False])]:
      inputs = np.array(inputs, dtype=dtype)
      outputs = np.array(outputs, dtype=np.bool)
      tf1.logging.debug('tf.even {} {} {}'.format(
          dtype, inputs.shape, outputs.shape))
      inputs_t = tf.constant(inputs, dtype=dtype)
      prog = test_programs.is_even_function_calls(include_types=False)
      typed = type_inference.infer_types(prog, [inputs_t], TF_BACKEND)
      expected_prog = test_programs.is_even_function_calls(dtype=dtype)
      self.assertSameTypes(expected_prog, typed)
      alloc = allocation_strategy.optimize(typed)
      lowered = lowering.lower_function_calls(alloc)
      self.assertAllEqual(
          outputs,
          self.evaluate(_execute(
              lowered, inputs_t, int(np.max(inputs)) + 3, TF_BACKEND)))

if __name__ == '__main__':
  tf.test.main()
