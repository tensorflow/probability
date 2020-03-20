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
"""Tests of the auto-batching VM."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

# Dependency imports

from absl import flags

import numpy as np

import tensorflow.compat.v1 as tf

from tensorflow_probability.python.experimental.auto_batching import numpy_backend
from tensorflow_probability.python.experimental.auto_batching import test_programs
from tensorflow_probability.python.experimental.auto_batching import tf_backend
from tensorflow_probability.python.experimental.auto_batching import virtual_machine as vm
from tensorflow_probability.python.internal import test_util

flags.DEFINE_string('test_device', None,
                    'TensorFlow device on which to place operators under test')
FLAGS = flags.FLAGS


NP_BACKEND = numpy_backend.NumpyBackend()
TF_BACKEND = tf_backend.TensorFlowBackend()
TF_BACKEND_NO_ASSERTS = tf_backend.TensorFlowBackend(safety_checks=False)


# This program always returns 2.
def _constant_execute(inputs, backend):
  # Stack depth limit is 4 to accommodate the initial value and
  # two pushes to "answer".
  with tf.compat.v2.name_scope('constant_program'):
    return vm.execute(
        test_programs.constant_program(), [inputs],
        max_stack_depth=4, backend=backend)


# This program returns n > 1 ? 2 : 0.
def _single_if_execute(inputs, backend):
  with tf.compat.v2.name_scope('single_if_program'):
    return vm.execute(
        test_programs.single_if_program(), [inputs],
        max_stack_depth=3, backend=backend)


def _product_type_execute(inputs, backend):
  with tf.compat.v2.name_scope('product_types_program'):
    return vm.execute(
        test_programs.synthetic_pattern_variable_program(), [inputs],
        max_stack_depth=4, backend=backend)


# This program returns fib(n), where fib(0) = fib(1) = 1.
def _fibonacci_execute(inputs, backend):
  with tf.compat.v2.name_scope('fibonacci_program'):
    return vm.execute(
        test_programs.fibonacci_program(), [inputs],
        max_stack_depth=15, backend=backend)


class VMTest(test_util.TestCase):

  def _tfTestHelper(self, run_asserts_fn, execute_program_fn):
    # Note: test_device is 'cpu', 'gpu', etc.

    # Various int32 and int64 kernels are missing for GPU, so we skip direct
    # tests on the GPU device, but test XLA on GPU below.
    if 'cpu' in FLAGS.test_device.lower():
      # Make sure everything works with no XLA compilation.
      with tf.device('CPU:0'):
        run_asserts_fn(
            functools.partial(execute_program_fn, backend=TF_BACKEND))

    # Force XLA compilation using tf.function.
    backend = TF_BACKEND_NO_ASSERTS
    f = functools.partial(execute_program_fn, backend=backend)
    f = tf.function(f, autograph=False, experimental_compile=True)
    with tf.device(FLAGS.test_device):
      run_asserts_fn(f)

  def testConstantNumpy(self):
    self.assertAllEqual([2], _constant_execute([5], NP_BACKEND))
    self.assertAllEqual([2, 2, 2], _constant_execute([5, 10, 15], NP_BACKEND))

  def testConstantTF(self):
    def _asserts_fn(f):
      ph = tf.compat.v1.placeholder_with_default(np.int64([8, 3]), shape=None)
      results = self.evaluate(
          [f(tf.cast([5], tf.int64)),
           f(tf.cast([5, 10, 15], tf.int64)),
           f(ph)])
      self.assertAllEqual([2], results[0])
      self.assertAllEqual([2, 2, 2], results[1])
      self.assertAllEqual([2, 2], results[2])
    self._tfTestHelper(_asserts_fn, _constant_execute)

  def testSingleIfNumpy(self):
    self.assertAllEqual([0], _single_if_execute([1], NP_BACKEND))
    self.assertAllEqual([2], _single_if_execute([3], NP_BACKEND))
    self.assertAllEqual([0, 2, 0], _single_if_execute([0, 5, -15], NP_BACKEND))

  def testSingleIfTF(self):
    def _asserts_fn(f):
      ph = tf.compat.v1.placeholder_with_default(np.int64([-3, 7]), shape=None)
      results = self.evaluate(
          [f(tf.cast([1], tf.int64)),
           f(tf.cast([3], tf.int64)),
           f(tf.cast([0, 5, -15], tf.int64)),
           f(ph)])
      self.assertAllEqual([0], results[0])
      self.assertAllEqual([2], results[1])
      self.assertAllEqual([0, 2, 0], results[2])
      self.assertAllEqual([0, 2], results[3])
    self._tfTestHelper(_asserts_fn, _single_if_execute)

  def testProductTypesNumpy(self):
    self.assertAllEqual(
        [[5, 6, 7], [6, 7, 8]], _product_type_execute([3, 4, 5], NP_BACKEND))

  def testProductTypesTF(self):
    def _asserts_fn(f):
      results = self.evaluate([f([3, 4, 5])])
      self.assertAllEqual([[5, 6, 7], [6, 7, 8]], results[0])
    self._tfTestHelper(_asserts_fn, _product_type_execute)

  def testFibonacciNumpy(self):
    self.assertAllEqual([8], _fibonacci_execute([5], NP_BACKEND))
    self.assertAllEqual(
        [8, 13, 34, 55], _fibonacci_execute([5, 6, 8, 9], NP_BACKEND))

  def testFibonacciTF(self):
    def _asserts_fn(f):
      ph = tf.compat.v1.placeholder_with_default(
          np.int64([0, 1, 3]), shape=None)
      results = self.evaluate([f([5]), f([5, 6, 8, 9]), f(ph)])
      self.assertAllEqual([8], results[0])
      self.assertAllEqual([8, 13, 34, 55], results[1])
      self.assertAllEqual([1, 1, 3], results[2])
    self._tfTestHelper(_asserts_fn, _fibonacci_execute)

  def testPeaNutsNumpy(self):
    def execute(batch_size, latent_size, data_size):
      data = np.random.normal(size=(data_size, latent_size)).astype(np.float32)
      def step_state(state):
        return state + np.sum(np.tensordot(data, state, ([1], [1])))
      state = np.random.normal(
          size=(batch_size, latent_size)).astype(np.float32)
      def choose_depth(count):
        del count
        return 3
      program = test_programs.pea_nuts_program(
          (latent_size,), choose_depth, step_state)
      input_counts = np.array([3] * batch_size)
      return vm.execute(
          program, [input_counts, state], 10, backend=NP_BACKEND)
    # Check that running the program doesn't crash
    result = execute(4, 3, 10)
    self.assertEqual((4, 3), result.shape)

  def testPeaNutsTF(self):
    batch_size = 4
    latent_size = 3
    data_size = 10
    def execute(_, backend):
      data = tf.random.normal(shape=(data_size, latent_size), dtype=np.float32)

      def step_state(state):
        return state + tf.reduce_sum(
            input_tensor=tf.tensordot(data, state, ([1], [1])))

      state = tf.random.normal(
          shape=(batch_size, latent_size), dtype=np.float32)
      def choose_depth(count):
        del count
        return 2
      program = test_programs.pea_nuts_program(
          (latent_size,), choose_depth, step_state)
      input_counts = np.array([3] * batch_size)
      return vm.execute(
          program, [input_counts, state], 10, backend=backend)
    # Check that running the program doesn't crash
    def _asserts_fn(f):
      result = self.evaluate(f(()))
      self.assertEqual((batch_size, latent_size), result.shape)
    self._tfTestHelper(_asserts_fn, execute)

if __name__ == '__main__':
  tf.test.main()
