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

# Dependency imports
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.experimental.auto_batching import allocation_strategy
from tensorflow_probability.python.experimental.auto_batching import lowering
from tensorflow_probability.python.experimental.auto_batching import numpy_backend
from tensorflow_probability.python.experimental.auto_batching import test_programs
from tensorflow_probability.python.experimental.auto_batching import virtual_machine as vm
from tensorflow_probability.python.internal import test_util as tfp_test_util


NP_BACKEND = numpy_backend.NumpyBackend()


def _fibonacci_lowered_execute(inputs, backend):
  prog = test_programs.fibonacci_function_calls()
  alloc = allocation_strategy.optimize(prog)
  lowered = lowering.lower_function_calls(alloc)
  return list(vm.execute(
      lowered, [inputs],
      max_stack_depth=15, backend=backend))


def _is_even_lowered_execute(inputs, backend):
  prog = test_programs.is_even_function_calls()
  alloc = allocation_strategy.optimize(prog)
  lowered = lowering.lower_function_calls(alloc)
  return list(vm.execute(
      lowered, [inputs],
      max_stack_depth=int(max(inputs)) + 3, backend=backend))


class LoweringTest(tfp_test_util.TestCase):

  def testLoweringFibonacciNumpy(self):
    self.assertEqual([8], _fibonacci_lowered_execute([5], NP_BACKEND))
    self.assertEqual(
        [8, 13, 34, 55], _fibonacci_lowered_execute([5, 6, 8, 9], NP_BACKEND))

  def testLoweringIsEvenNumpy(self):
    self.assertEqual([True], _is_even_lowered_execute([0], NP_BACKEND))
    self.assertEqual([False], _is_even_lowered_execute([1], NP_BACKEND))
    self.assertEqual([True], _is_even_lowered_execute([2], NP_BACKEND))
    self.assertEqual(
        [False, True, True, False, True],
        _is_even_lowered_execute([5, 6, 8, 9, 0], NP_BACKEND))

if __name__ == '__main__':
  tf.test.main()
