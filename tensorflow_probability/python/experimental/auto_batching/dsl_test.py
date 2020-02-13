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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.experimental.auto_batching import allocation_strategy
from tensorflow_probability.python.experimental.auto_batching import dsl
from tensorflow_probability.python.experimental.auto_batching import instructions
from tensorflow_probability.python.experimental.auto_batching import lowering
from tensorflow_probability.python.experimental.auto_batching import numpy_backend
from tensorflow_probability.python.experimental.auto_batching import tf_backend
from tensorflow_probability.python.experimental.auto_batching import type_inference
from tensorflow_probability.python.experimental.auto_batching import virtual_machine as vm
from tensorflow_probability.python.internal import test_util

TF_BACKEND = tf_backend.TensorFlowBackend()

NP_BACKEND = numpy_backend.NumpyBackend()


def _execute(prog, inputs, stack_depth, backend):
  return vm.execute(
      prog, [inputs], max_stack_depth=stack_depth, backend=backend)


def fibonacci_program():
  ab = dsl.ProgramBuilder()

  def fib_type(arg_types):
    return arg_types[0]

  with ab.function('fibonacci', type_inference=fib_type) as fibonacci:
    ab.param('n')
    ab.var.cond = ab.primop(lambda n: n > 1)
    with ab.if_(ab.var.cond, then_name='recur'):
      ab.var.nm1 = ab.primop(lambda n: n - 1)
      ab.var.fibm1 = ab.call(fibonacci, [ab.var.nm1])
      ab.var.nm2 = ab.primop(lambda n: n - 2)
      ab.var.fibm2 = ab.call(fibonacci, [ab.var.nm2])
      ab.var.ans = ab.primop(lambda fibm1, fibm2: fibm1 + fibm2)
    with ab.else_(else_name='base-case', continue_name='finish'):
      ab.var.ans = ab.const(1)
    ab.return_(ab.var.ans)

  prog = ab.program(main=fibonacci)
  return prog


def even_odd_program():
  ab = dsl.ProgramBuilder()

  def pred_type(_):
    return instructions.TensorType(np.bool, ())

  odd = ab.declare_function('odd', type_inference=pred_type)

  with ab.function('even', type_inference=pred_type) as even:
    ab.param('n')
    ab.var.cond = ab.primop(lambda n: n <= 0)
    with ab.if_(ab.var.cond, then_name='base-case'):
      ab.var.ans = ab.const(True)
    with ab.else_(else_name='recur', continue_name='finish'):
      ab.var.nm1 = ab.primop(lambda n: n - 1)
      ab.var.ans = ab.call(odd, [ab.var.nm1])
    ab.return_(ab.var.ans)

  with ab.define_function(odd):
    ab.param('n')
    ab.var.cond = ab.primop(lambda n: n <= 0)
    with ab.if_(ab.var.cond, then_name='base-case'):
      ab.var.ans = ab.const(False)
    with ab.else_(else_name='recur', continue_name='finish'):
      ab.var.nm1 = ab.primop(lambda n: n - 1)
      ab.var.ans = ab.call(even, [ab.var.nm1])
    ab.return_(ab.var.ans)

  prog = ab.program(main=even)
  return prog


def synthetic_pattern_program():
  ab = dsl.ProgramBuilder()
  def my_type(_):
    int_ = instructions.TensorType(np.int64, ())
    return ((int_, int_), int_, (int_, int_))

  with ab.function('synthetic', type_inference=my_type) as syn:
    ab.param('batch_size_index')
    one, three, five = ab.locals_(3)
    ab((one, (five, three))).pattern = ab.primop(lambda: (1, (2, 3)))
    ab(((ab.var.four, five), ab.var.six)).pattern = ab.primop(
        lambda: ((4, 5), 6))
    ab.return_(((one, three), ab.var.four, (five, ab.var.six)))

  prog = ab.program(main=syn)
  return prog


@test_util.test_all_tf_execution_regimes
class AutoBatchingTest(test_util.TestCase):

  def testAutoBatchingFibonacciNumpy(self):
    for inputs, outputs in ([5], [8]), ([5, 6, 8, 9], [8, 13, 34, 55]):
      # This test doesn't pass with int32 input types, because (apparently)
      # numpy can't tell the difference between an ndarray of shape () and known
      # dtype, and a scalar (literal) whose dtype needs to be inferred.
      # To wit:
      #   (np.zeros((), dtype=np.int32) - 1).dtype == np.int64
      # because that's somehow the best numpy can do, even though
      #   (np.zeros([6], dtype=np.int32) - 1).dtype == np.int32
      # Needless to say, this messes up type inference for programs like
      # Fibonacci whose unbatched input shape is scalar.
      inputs = np.array(inputs, dtype=np.int64)
      outputs = np.array(outputs, dtype=np.int64)
      prog = fibonacci_program()
      # print(prog)
      typed = type_inference.infer_types(prog, [inputs], NP_BACKEND)
      # print(typed)
      alloc = allocation_strategy.optimize(typed)
      lowered = lowering.lower_function_calls(alloc)
      # print(lowered)
      self.assertAllEqual(outputs, _execute(lowered, inputs, 15, NP_BACKEND))

  def testAutoBatchingFibonacciTF(self):
    for inputs, outputs in ([5], [8]), ([5, 6, 8, 9], [8, 13, 34, 55]):
      inputs = np.array(inputs, dtype=np.int32)
      outputs = np.array(outputs, dtype=np.int32)
      prog = fibonacci_program()
      # print(prog)
      inputs_t = tf.constant(inputs, dtype=np.int32)
      typed = type_inference.infer_types(prog, [inputs_t], TF_BACKEND)
      # print(typed)
      alloc = allocation_strategy.optimize(typed)
      lowered = lowering.lower_function_calls(alloc)
      # print(lowered)
      self.assertAllEqual(
          outputs, self.evaluate(_execute(lowered, inputs_t, 15, TF_BACKEND)))

  def testAutoBatchingEvenOddNumpy(self):
    for inputs, outputs in ([5], [False]), ([5, 6, 8, 9],
                                            [False, True, True, False]):
      inputs = np.array(inputs, dtype=np.int64)
      outputs = np.array(outputs, dtype=np.bool)
      prog = even_odd_program()
      # print(prog)
      typed = type_inference.infer_types(prog, [inputs], NP_BACKEND)
      # print(typed)
      alloc = allocation_strategy.optimize(typed)
      lowered = lowering.lower_function_calls(alloc)
      # print(lowered)
      self.assertAllEqual(outputs, _execute(lowered, inputs, 15, NP_BACKEND))

  def testAutoBatchingEvenOddTF(self):
    for inputs, outputs in ([5], [False]), ([5, 6, 8, 9],
                                            [False, True, True, False]):
      inputs = np.array(inputs, dtype=np.int32)
      outputs = np.array(outputs, dtype=np.int32)
      prog = even_odd_program()
      # print(prog)
      inputs_t = tf.constant(inputs, dtype=np.int32)
      typed = type_inference.infer_types(prog, [inputs_t], TF_BACKEND)
      # print(typed)
      alloc = allocation_strategy.optimize(typed)
      lowered = lowering.lower_function_calls(alloc)
      # print(lowered)
      self.assertAllEqual(
          outputs, self.evaluate(_execute(lowered, inputs_t, 15, TF_BACKEND)))

  def testAutoBatchingMultivalueTF(self):
    input_ = np.array([1, 1, 1], dtype=np.int64)
    output = ((np.array([1, 1, 1], dtype=np.int64),
               np.array([3, 3, 3], dtype=np.int64)),
              np.array([4, 4, 4], dtype=np.int64),
              (np.array([5, 5, 5], dtype=np.int64),
               np.array([6, 6, 6], dtype=np.int64)))
    prog = synthetic_pattern_program()
    # print(prog)
    input_t = tf.constant(input_, dtype=np.int64)
    typed = type_inference.infer_types(prog, [input_t], TF_BACKEND)
    # print(typed)
    alloc = allocation_strategy.optimize(typed)
    lowered = lowering.lower_function_calls(alloc)
    # print(lowered)
    for expected, obtained in instructions.pattern_zip(
        output, self.evaluate(_execute(lowered, input_t, 15, TF_BACKEND))):
      self.assertAllEqual(expected, obtained)

if __name__ == '__main__':
  tf.test.main()
