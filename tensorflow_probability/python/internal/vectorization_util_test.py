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
"""Tests for utilities for testing distributions and/or bijectors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.internal import vectorization_util


@test_util.test_all_tf_execution_regimes
class VectorizationTest(test_util.TestCase):

  def test_docstring_example(self):
    add = lambda a, b: a + b
    add_vector_to_scalar = vectorization_util.make_rank_polymorphic(
        add, core_ndims=(1, 0))
    self.assertAllEqual(
        [[4., 5.], [5., 6.], [6., 7.]],
        self.evaluate(add_vector_to_scalar(
            tf.constant([1., 2.]), tf.constant([3., 4., 5.]))))

  def test_can_take_structured_input_and_output(self):
    # Dummy function that takes a (tuple, dict) pair
    # and returns a (dict, scalar) pair.
    def fn(x, y):
      a, b, c = x
      d, e = y['d'], y['e']
      return {'r': a * b + c}, d + e

    vectorized_fn = vectorization_util.make_rank_polymorphic(
        fn, core_ndims=0)

    x = np.array([[2.], [3.]]), np.array(2.), np.array([5., 6., 7.])
    y = {'d': np.array([[1.]]), 'e': np.array([2., 3., 4.])}
    vectorized_result = self.evaluate(vectorized_fn(x, y))
    result = tf.nest.map_structure(lambda a, b: a * np.ones(b.shape),
                                   fn(x, y), vectorized_result)
    self.assertAllClose(result, vectorized_result)

  @parameterized.named_parameters(
      ('static_shapes', True),
      ('dynamic_shapes', False))
  def tests_aligns_broadcast_dims_using_core_ndims(self, is_static):
    np.random.seed(test_util.test_seed() % 2**32)

    def matvec(a, b):
      # Throws an error if either arg has extra dimensions.
      return tf.linalg.matvec(tf.reshape(a, tf.shape(a)[-2:]),
                              tf.reshape(b, tf.shape(b)[-1:]))

    vectorized_matvec = vectorization_util.make_rank_polymorphic(
        matvec, core_ndims=(
            self.maybe_static(2, is_static=is_static),
            self.maybe_static(1, is_static=is_static)))

    for (a_shape, b_shape) in (([3, 2], [2]),
                               ([4, 3, 2], [2]),
                               ([4, 3, 2], [5, 1, 2])):
      a = self.maybe_static(np.random.randn(*a_shape), is_static=is_static)
      b = self.maybe_static(np.random.randn(*b_shape), is_static=is_static)

      c = tf.linalg.matvec(a, b)
      c_vectorized = vectorized_matvec(a, b)
      if is_static:
        self.assertAllEqual(c.shape, c_vectorized.shape)
      self.assertAllEqual(*self.evaluate((c, c_vectorized)))

  def test_can_call_with_variable_number_of_args(self):

    def scalar_sum(*args):
      return sum([tf.reshape(x, []) for x in args])
    vectorized_sum = vectorization_util.make_rank_polymorphic(
        scalar_sum, core_ndims=0)

    xs = [1.,
          np.array([3., 2.]).astype(np.float32),
          np.array([[1., 2.], [-4., 3.]]).astype(np.float32)]
    self.assertAllEqual(self.evaluate(vectorized_sum(*xs)), sum(xs))

  def test_raises_error_on_insufficient_rank_input(self):
    def matvec(a, b):
      # Throws an error if either arg has extra dimensions.
      return tf.linalg.matvec(tf.reshape(a, tf.shape(a)[-2:]),
                              tf.reshape(b, tf.shape(b)[-1:]))
    vectorized_matvec = vectorization_util.make_rank_polymorphic(
        matvec, core_ndims=(2, 1), validate_args=True)

    # Static check.
    with self.assertRaisesRegexp(
        ValueError, 'Cannot broadcast a Tensor having lower rank'):
      vectorized_matvec(tf.zeros([5]), tf.zeros([2, 1, 5]))

    # Runtime check.
    if not tf.executing_eagerly():
      with self.assertRaisesOpError(
          'Condition x >= 0 did not hold element-wise'):
        self.evaluate(vectorized_matvec(
            self.maybe_static(tf.zeros([5]), is_static=False),
            self.maybe_static(tf.zeros([2, 1, 5]), is_static=False)))

  def test_can_escape_vectorization_with_none_ndims(self):

    # Suppose the original fn supports `None` as an input.
    fn = lambda x, y: (tf.reduce_sum(x, axis=0), y[0] if y is not None else y)

    polymorphic_fn = vectorization_util.make_rank_polymorphic(
        fn, core_ndims=[1, None])
    rx, ry = polymorphic_fn([[1., 2., 4.], [3., 5., 7.]], None)
    self.assertAllEqual(rx.shape, [2])
    self.assertIsNone(ry)

    single_arg_polymorphic_fn = vectorization_util.make_rank_polymorphic(
        lambda y: fn(tf.convert_to_tensor([1., 2., 3.]), y), core_ndims=None)
    rx, ry = self.evaluate(single_arg_polymorphic_fn(
        tf.convert_to_tensor([[1., 3.], [2., 4.]])))
    self.assertAllEqual(ry, [1., 3.])

  def test_docstring_example_passing_fn_arg(self):
    def apply_binop(fn, a, b):
      return fn(a, b)
    apply_binop_to_vector_and_scalar = vectorization_util.make_rank_polymorphic(
        apply_binop, core_ndims=(None, 1, 0))
    r = self.evaluate(apply_binop_to_vector_and_scalar(
        lambda a, b: a * b, tf.constant([1., 2.]), tf.constant([3., 4., 5.])))
    self.assertAllEqual(r, np.array(
        [[3., 6.], [4., 8.], [5., 10.]], dtype=np.float32))


if __name__ == '__main__':
  tf.test.main()
