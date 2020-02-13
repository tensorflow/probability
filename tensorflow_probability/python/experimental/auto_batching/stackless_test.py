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

import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.experimental.auto_batching import allocation_strategy
from tensorflow_probability.python.experimental.auto_batching import numpy_backend
from tensorflow_probability.python.experimental.auto_batching import stackless
from tensorflow_probability.python.experimental.auto_batching import test_programs
from tensorflow_probability.python.experimental.auto_batching import tf_backend
from tensorflow_probability.python.internal import test_util as tfp_test_util

TF_BACKEND = tf_backend.TensorFlowBackend()
NP_BACKEND = numpy_backend.NumpyBackend()


def _fibonacci_stackless_execute(inputs, backend):
  prog = test_programs.fibonacci_function_calls()
  alloc = allocation_strategy.optimize(prog)
  return stackless.execute(alloc, backend, None, inputs)


def _is_even_stackless_execute(inputs, backend):
  prog = test_programs.is_even_function_calls()
  alloc = allocation_strategy.optimize(prog)
  return stackless.execute(alloc, backend, None, inputs)


# Stackless autobatching doesn't work in TF graph mode.
class StacklessTest(tfp_test_util.TestCase):

  def testStacklessFibonacciNumpy(self):
    self.assertEqual([8], list(_fibonacci_stackless_execute([5], NP_BACKEND)))
    self.assertEqual(
        [8, 13, 34, 55],
        list(_fibonacci_stackless_execute([5, 6, 8, 9], NP_BACKEND)))

  def testStacklessFibonacciTF(self):
    self.assertAllEqual(
        [8], self.evaluate(_fibonacci_stackless_execute([5], TF_BACKEND)))
    self.assertAllEqual(
        [8, 13, 34, 55],
        self.evaluate(_fibonacci_stackless_execute([5, 6, 8, 9], TF_BACKEND)))

  def testStacklessIsEvenNumpy(self):
    self.assertEqual([True], list(_is_even_stackless_execute([0], NP_BACKEND)))
    self.assertEqual([False], list(_is_even_stackless_execute([1], NP_BACKEND)))
    self.assertEqual([True], list(_is_even_stackless_execute([2], NP_BACKEND)))
    self.assertEqual(
        [False, True, True, False, True],
        list(_is_even_stackless_execute([5, 6, 8, 9, 0], NP_BACKEND)))

  def testStacklessIsEvenTF(self):
    self.assertAllEqual(
        [True], self.evaluate(_is_even_stackless_execute([0], TF_BACKEND)))
    self.assertAllEqual(
        [False], self.evaluate(_is_even_stackless_execute([1], TF_BACKEND)))
    self.assertAllEqual(
        [True], self.evaluate(_is_even_stackless_execute([2], TF_BACKEND)))
    self.assertAllEqual(
        [False, True, True, False, True],
        self.evaluate(_is_even_stackless_execute([5, 6, 8, 9, 0], TF_BACKEND)))

if __name__ == '__main__':
  tf1.enable_eager_execution()
  tf.test.main()
