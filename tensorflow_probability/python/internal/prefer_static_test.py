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
"""Tests for tensorflow_probability internal prefer_static."""

from absl.testing import parameterized
import numpy as np

import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import test_util


def raise_exception():
  raise RuntimeError('Did not expect to be called')


def raise_exception_in_eager_mode(value):
  def f():
    if tf.executing_eagerly():
      raise_exception()
    return value
  return f


@test_util.test_all_tf_execution_regimes
class GetStaticValueTest(test_util.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name='_True',
           predicate_fn=lambda: True,
           expected_truth_value=True,
           expect_static_graph_evaluation=True),
      dict(testcase_name='_False',
           predicate_fn=lambda: False,
           expected_truth_value=False,
           expect_static_graph_evaluation=True),
      dict(testcase_name='_tf.constant(True)',
           predicate_fn=lambda: tf.constant(True),
           expected_truth_value=True,
           expect_static_graph_evaluation=True),
      dict(testcase_name='_tf.constant(1)',
           predicate_fn=lambda: tf.constant(1),
           expected_truth_value=True,
           expect_static_graph_evaluation=True),
      dict(testcase_name='_tf.constant(1)>0',
           predicate_fn=lambda: tf.constant(1) > 0,
           expected_truth_value=True,
           expect_static_graph_evaluation=True),
      )
  def testStaticEvaluation(self,
                           predicate_fn,
                           expected_truth_value,
                           expect_static_graph_evaluation):
    predicate = predicate_fn()
    static_predicate = ps._get_static_predicate(predicate)

    if tf.executing_eagerly() or expect_static_graph_evaluation:
      # If we are in eager mode, we always expect static evaluation.
      self.assertIsNotNone(static_predicate)
      self.assertEqual(expected_truth_value, static_predicate)
    else:
      self.assertIsNone(static_predicate)


@test_util.test_all_tf_execution_regimes
class PredicatesTest(test_util.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name='_greater_true',
           predicate=ps.greater,
           args_fn=lambda: [tf.constant(1), tf.constant(0)],
           kwargs=dict(),
           expected=True),
      dict(testcase_name='_greater_false',
           predicate=ps.greater,
           args_fn=lambda: [tf.constant(-.1), tf.constant(0.)],
           kwargs=dict(),
           expected=False),
      dict(testcase_name='_greater_none',
           predicate=ps.greater,
           args_fn=lambda: [tf.constant(1) + tf.constant(0), tf.constant(0)],
           kwargs=dict(),
           expected=True),
      dict(testcase_name='_less_true',
           predicate=ps.less,
           args_fn=lambda: [tf.constant(-1), tf.constant(0)],
           kwargs=dict(),
           expected=True),
      dict(testcase_name='_log',
           predicate=ps.log,
           args_fn=lambda: [tf.constant(1.)],
           kwargs=dict(),
           expected=0.),
      dict(testcase_name='_equal_true',
           predicate=ps.equal,
           args_fn=lambda: [tf.constant(0), 0],
           kwargs=dict(),
           expected=True),
      dict(testcase_name='_equal_false',
           predicate=ps.equal,
           args_fn=lambda: [tf.constant(1), tf.constant(0)],
           kwargs=dict(),
           expected=False),
      dict(testcase_name='_and_true',
           predicate=ps.logical_and,
           args_fn=lambda: [True, tf.constant(True)],
           kwargs=dict(),
           expected=True),
      dict(testcase_name='_and_none',
           predicate=ps.logical_and,
           args_fn=lambda: [tf.constant(True), tf.equal(1, 1)],
           kwargs=dict(),
           expected=True),
      dict(testcase_name='_or_true',
           predicate=ps.logical_or,
           args_fn=lambda: [tf.constant(True), tf.constant(False)],
           kwargs=dict(),
           expected=True),
      dict(testcase_name='_all_true',
           predicate=ps.reduce_all,
           args_fn=lambda: [[tf.constant(True)] * 10],
           kwargs=dict(),
           expected=True),
      dict(testcase_name='_all_false',
           predicate=ps.reduce_all,
           args_fn=lambda: [[tf.constant(True),  # pylint: disable=g-long-lambda
                             True,
                             tf.constant(False)]],
           kwargs=dict(),
           expected=False),
      dict(testcase_name='_all_with_axis',
           predicate=ps.reduce_all,
           args_fn=lambda: ([[True, tf.constant(True)],  # pylint: disable=g-long-lambda
                             [False, True]],),
           kwargs=dict(axis=1),
           expected=[True, False]),
      dict(testcase_name='_all_with_name',
           predicate=ps.reduce_all,
           args_fn=lambda: ([[True, tf.constant(True)],  # pylint: disable=g-long-lambda
                             [False, True]],),
           kwargs=dict(axis=1, name='my_name'),
           expected=[True, False]),
      dict(testcase_name='_any_true',
           predicate=ps.reduce_any,
           args_fn=lambda: [[tf.constant(True),  # pylint: disable=g-long-lambda
                             tf.constant(False),
                             tf.constant(False)]],
           kwargs=dict(),
           expected=True),
      dict(testcase_name='_any_false',
           predicate=ps.reduce_any,
           args_fn=lambda: [[tf.constant(False)] * 23],
           kwargs=dict(),
           expected=False),
      dict(testcase_name='_any_keepdims',
           predicate=ps.reduce_any,
           args_fn=lambda: ([[True, tf.constant(True)],  # pylint: disable=g-long-lambda
                             [False, True]],),
           kwargs=dict(keepdims=True),
           expected=[[True]]),
      )
  def test_static_predicate(self, predicate, args_fn, kwargs, expected):
    actual = predicate(*args_fn(), **kwargs)
    self.assertAllCloseAccordingToType(expected, actual)


@test_util.test_all_tf_execution_regimes
class CondTest(test_util.TestCase):

  def test_true(self):
    x = tf.constant(2)
    y = tf.constant(5)
    z = ps.cond(True, lambda: tf.multiply(x, 16), lambda: tf.multiply(y, 5))
    self.assertEqual(self.evaluate(z), 32)

  def test_false(self):
    x = tf.constant(4)
    y = tf.constant(3)
    z = ps.cond(False, lambda: tf.multiply(x, 16), lambda: tf.multiply(y, 3))
    self.assertEqual(self.evaluate(z), 9)

  def test_missing_arg1(self):
    x = tf.constant(1)
    with self.assertRaises(TypeError):
      ps.cond(True, false_fn=lambda: x)

  def test_missing_arg2(self):
    x = tf.constant(1)
    with self.assertRaises(TypeError):
      ps.cond(True, lambda: x)


@test_util.test_all_tf_execution_regimes
class CaseTest(test_util.TestCase):

  def test_true(self):
    x = tf.constant(0)
    conditions = [(True, lambda: tf.constant(1)),
                  (tf.equal(x, 1), raise_exception)]
    y = ps.case(conditions, default=raise_exception, exclusive=False)
    z = ps.case(conditions, default=raise_exception, exclusive=True)

    self.assertEqual(self.evaluate(y), 1)
    self.assertEqual(self.evaluate(z), 1)

  def test_false(self):
    conditions = [(False, raise_exception)]
    y = ps.case(conditions,
                default=lambda: tf.constant(1),
                exclusive=False)
    z = ps.case(conditions,
                default=lambda: tf.constant(1),
                exclusive=True)
    self.assertEqual(self.evaluate(y), 1)
    self.assertEqual(self.evaluate(z), 1)

  def test_mix(self):
    x = tf.constant(0)
    y = tf.constant(10)
    conditions = [(x > 1, lambda: tf.constant(1)),
                  (y < 1, raise_exception_in_eager_mode(tf.constant(2))),
                  (tf.constant(False), raise_exception),
                  (tf.constant(True), lambda: tf.constant(3))]
    z = ps.case(conditions, default=lambda: raise_exception)
    self.assertEqual(self.evaluate(z), 3)


@test_util.test_all_tf_execution_regimes
class ShapeTest(test_util.TestCase):

  def test_shape(self):
    vector_value = [0., 1.]

    # case: numpy input
    self.assertAllEqual(ps.shape(np.array(vector_value)), [2])

    # case: tensor input with static shape
    self.assertAllEqual(ps.shape(tf.constant(vector_value)), [2])

    # case: tensor input with dynamic shape
    if not tf.executing_eagerly():
      shape = ps.shape(tf1.placeholder_with_default(vector_value, shape=None))
      self.assertAllEqual(self.evaluate(shape), [2])

  def test_rank_from_shape_scalar(self):
    self.assertEqual(1, ps.rank_from_shape(5))
    v = tf.Variable(4, shape=tf.TensorShape(None))
    self.evaluate(v.initializer)
    self.assertEqual(1, self.evaluate(ps.rank_from_shape(v)))

  def test_convert_dimension_to_tensor(self):
    v = ps.constant(tf1.Dimension(1))
    self.assertEqual(1, v)

  def test_dimension_size(self):
    shape = [3, 2, 1]
    idx = 1
    idx_tensor = tf.convert_to_tensor(idx)

    # case: numpy input.
    self.assertEqual(ps.dimension_size(np.zeros(shape), idx), shape[idx])
    self.assertEqual(ps.dimension_size(np.zeros(shape), idx_tensor),
                     shape[idx])

    # case: static-shape Tensor input.
    self.assertEqual(ps.dimension_size(tf.zeros(shape), idx), shape[idx])
    self.assertNotIsInstance(
        ps.dimension_size(tf.zeros(shape), idx), tf1.Dimension)
    self.assertEqual(ps.dimension_size(tf.zeros(shape), idx_tensor),
                     shape[idx])
    self.assertNotIsInstance(
        ps.dimension_size(tf.zeros(shape), idx_tensor), tf1.Dimension)

    if tf.executing_eagerly():
      return

    # Case: input is Tensor with fully unknown shape.
    zeros_pl = tf1.placeholder_with_default(tf.zeros(shape), shape=None)
    idx_pl = tf1.placeholder_with_default(idx, shape=[])
    self.assertAllEqual(ps.dimension_size(zeros_pl, idx), shape[idx])
    self.assertAllEqual(ps.dimension_size(zeros_pl, idx_tensor), shape[idx])
    self.assertAllEqual(ps.dimension_size(zeros_pl, idx_pl), shape[idx])

    # Case: input is Tensor with partially known shape.
    # The result should be static if idx is.
    zeros_partial_pl = tf1.placeholder_with_default(
        tf.zeros(shape),
        shape=tf.TensorShape([None, shape[idx], None]))
    self.assertEqual(ps.dimension_size(zeros_partial_pl, idx), shape[idx])
    self.assertEqual(ps.dimension_size(zeros_partial_pl, idx_tensor),
                     shape[idx])
    self.assertAllEqual(ps.dimension_size(zeros_partial_pl, idx_pl),
                        shape[idx])

  def test_shape_slice(self):
    shape = [3, 2, 1]
    slice_ = slice(1, 2)
    slice_tensor = slice(tf.constant(1), tf.constant(2))

    # case: numpy input.
    self.assertEqual(ps.shape_slice(np.zeros(shape), slice_), shape[slice_])
    self.assertEqual(
        ps.shape_slice(np.zeros(shape), slice_tensor), shape[slice_])

    # case: static-shape Tensor input.
    self.assertEqual(ps.shape_slice(tf.zeros(shape), slice_), shape[slice_])
    self.assertNotIsInstance(
        ps.shape_slice(tf.zeros(shape), slice_), tf.TensorShape)
    self.assertEqual(
        ps.shape_slice(tf.zeros(shape), slice_tensor), shape[slice_])
    self.assertNotIsInstance(
        ps.shape_slice(tf.zeros(shape), slice_tensor), tf1.Dimension)

    if tf.executing_eagerly():
      return

    # Case: input is Tensor with fully unknown shape.
    zeros_pl = tf1.placeholder_with_default(tf.zeros(shape), shape=None)
    slice_pl = slice(
        tf1.placeholder_with_default(1, shape=[]),
        tf1.placeholder_with_default(2, shape=[]))
    self.assertAllEqual(ps.shape_slice(zeros_pl, slice_), shape[slice_])
    self.assertAllEqual(
        ps.shape_slice(zeros_pl, slice_tensor), shape[slice_])
    self.assertAllEqual(ps.shape_slice(zeros_pl, slice_pl), shape[slice_])

    # Case: input is Tensor with partially known shape.
    # The result should be static if slice_ is.
    zeros_partial_pl = tf1.placeholder_with_default(
        tf.zeros(shape), shape=tf.TensorShape([None, 2, 1]))
    self.assertEqual(ps.shape_slice(zeros_partial_pl, slice_), shape[slice_])
    self.assertEqual(
        ps.shape_slice(zeros_partial_pl, slice_tensor), shape[slice_])
    self.assertAllEqual(
        ps.shape_slice(zeros_partial_pl, slice_pl), shape[slice_])

  def test_rank_from_shape(self):
    shape = [2, 4, 3]
    expected_rank = len(shape)
    v_ndarray = np.ones(shape)

    # case: shape_tensor is tuple
    rank = ps.rank_from_shape(shape_tensor_fn=v_ndarray.shape)
    self.assertEqual(rank, expected_rank)

    # case: shape_tensor is ndarray
    rank = ps.rank_from_shape(shape_tensor_fn=ps.shape(v_ndarray))
    self.assertEqual(rank, expected_rank)

    # case: tensorshape is fully defined
    v_tensor = tf.convert_to_tensor(v_ndarray)
    rank = ps.rank_from_shape(
        shape_tensor_fn=ps.shape(v_tensor),
        tensorshape=v_tensor.shape)
    self.assertEqual(rank, expected_rank)

    if not tf.executing_eagerly():
      # case: tensorshape is unknown, rank cannot be statically inferred
      v_dynamic = tf1.placeholder_with_default(v_ndarray, shape=None)
      rank = ps.rank_from_shape(
          shape_tensor_fn=lambda: ps.shape(v_dynamic),
          tensorshape=v_dynamic.shape)
      self.assertEqual(self.evaluate(rank), expected_rank)

      # case: tensorshape is not provided, rank cannot be statically inferred
      rank = ps.rank_from_shape(shape_tensor_fn=lambda: ps.shape(v_dynamic))
      self.assertEqual(self.evaluate(rank), expected_rank)

  def test_tensorshape(self):
    self.assertAllEqual([0], ps.shape(tf.TensorShape([])))
    self.assertAllEqual([1], ps.shape(tf.TensorShape(3)))
    self.assertAllEqual([1], ps.shape(tf.TensorShape([2])))
    self.assertAllEqual([2], ps.shape(tf.TensorShape([19, 17])))
    self.assertAllEqual([3], ps.shape(tf.TensorShape([2, 1, 4])))

    with self.assertRaisesRegex(
        ValueError, 'Cannot convert a partially known TensorShape'):
      ps.shape(tf.TensorShape(None))

    with self.assertRaisesRegex(
        ValueError, 'Cannot convert a partially known TensorShape'):
      # Should we support this?  tf.shape fails on this input.
      ps.shape(tf.TensorShape([3, None, 4]))


@test_util.test_all_tf_execution_regimes
class SetDiff1DTest(test_util.TestCase):

  def test_static(self):
    self.assertAllEqual(
        [0, 3, 4],
        ps.setdiff1d(np.arange(5), [1, 2]))
    self.assertAllEqual(
        [],
        ps.setdiff1d([], [1, 2]))
    self.assertAllEqual(
        [1, 2],
        ps.setdiff1d([1, 2], []))

  def test_dynamic(self):
    if tf.executing_eagerly(): return
    x = tf1.placeholder_with_default(np.arange(5), shape=None)
    self.assertAllEqual([0, 3, 4], self.evaluate(ps.setdiff1d(x, [1, 2])))
    x = tf1.placeholder_with_default(np.array([], np.int32), shape=None)
    self.assertAllEqual([], self.evaluate(ps.setdiff1d(x, [1, 2])))
    self.assertAllEqual([1, 2], self.evaluate(ps.setdiff1d([1, 2], x)))


@test_util.test_all_tf_execution_regimes
class SizeTest(test_util.TestCase):

  def test_static(self):
    self.assertAllEqual(
        3 * 4 * 5,
        ps.size(tf.random.normal([3, 4, 5], seed=test_util.test_seed())))

  def test_dynamic(self):
    if tf.executing_eagerly(): return
    x = tf1.placeholder_with_default(
        tf.random.normal([3, 4, 5], seed=test_util.test_seed()), shape=None)
    self.assertAllEqual(3 * 4 * 5, self.evaluate(ps.size(x)))

  def test_tensorshape(self):
    self.assertAllEqual(1, ps.size(tf.TensorShape(3)))
    self.assertAllEqual(1, ps.size(tf.TensorShape([2])))
    self.assertAllEqual(2, ps.size(tf.TensorShape([19, 17])))
    self.assertAllEqual(3, ps.size(tf.TensorShape([2, 1, 4])))

    with self.assertRaises(Exception):
      ps.size(tf.TensorShape(None))

    with self.assertRaises(Exception):
      ps.size(tf.TensorShape([3, None, 4]))


@test_util.test_all_tf_execution_regimes
class NonNegativeAxisTest(test_util.TestCase):

  def test_static_scalar_positive_index(self):
    positive_axis = ps.non_negative_axis(axis=2, rank=4)
    self.assertAllEqual(2, positive_axis)

  def test_static_scalar_negative_index(self):
    positive_axis = ps.non_negative_axis(axis=-1, rank=4)
    self.assertAllEqual(3, positive_axis)

  def test_static_vector_index(self):
    positive_axis = ps.non_negative_axis(axis=[0, -2], rank=4)
    self.assertAllEqual([0, 2], positive_axis)

  @test_util.jax_disable_variable_test
  def test_dynamic_vector_index(self):
    axis = tf.Variable([0, -2])
    positive_axis = ps.non_negative_axis(axis=axis, rank=4)
    self.evaluate(axis.initializer)
    self.assertAllEqual([0, 2], self.evaluate(positive_axis))


@test_util.test_all_tf_execution_regimes
class BroadcastShapeTest(test_util.TestCase):

  def test_static(self):
    self.assertAllEqual(
        (3, 4, 2, 5),
        ps.broadcast_shape((4, 1, 1), (3, 1, 2, 5)))

    self.assertAllEqual((3, 4, 2, 5),
                        ps.broadcast_shape(
                            tf.convert_to_tensor((4, 1, 1)),
                            tf.convert_to_tensor((3, 1, 2, 5))))

  def test_dynamic(self):
    if tf.executing_eagerly():
      return

    shape = ps.broadcast_shape(
        tf.convert_to_tensor([3, 2, 1]),
        tf.shape(
            tf1.placeholder_with_default(np.zeros((1, 5)), shape=(None, 5))))
    self.assertIsNone(tf.get_static_value(shape))
    self.assertAllEqual([3, 2, 5], self.evaluate(shape))


@test_util.test_all_tf_execution_regimes
class PadTest(test_util.TestCase):

  def test_num_paddings_dynamic(self):
    n = tf1.placeholder_with_default(2, shape=None)
    x = ps.pad([2, 3], paddings=[[0, n]], constant_values=1)
    if not ps.is_numpy(x):
      x = self.evaluate(x)
    self.assertAllEqual([2, 3, 1, 1], x)

  def test_num_paddings_static(self):
    n = 2
    x = ps.pad([2, 3], paddings=[[0, n]], constant_values=1)
    self.assertAllEqual([2, 3, 1, 1], x)


@test_util.test_all_tf_execution_regimes
class SmartWhereTest(test_util.TestCase):

  def test_static_scalar_condition(self):
    fn_calls = [0, 0]
    ones = tf.ones([10])
    zeros = tf.zeros([10])
    def fn1():
      fn_calls[0] += 1
      return ones
    def fn2():
      fn_calls[1] += 1
      return zeros

    self.assertAllEqual(zeros, ps.smart_where(False, fn1, fn2))
    self.assertEqual([0, 1], fn_calls)
    self.assertAllEqual(ones, ps.smart_where(True, fn1, fn2))
    self.assertEqual([1, 1], fn_calls)
    self.assertAllEqual(zeros, ps.smart_where(tf.constant(False), fn1, fn2))
    self.assertEqual([1, 2], fn_calls)
    self.assertAllEqual(ones, ps.smart_where(tf.constant(True), fn1, fn2))
    self.assertEqual([2, 2], fn_calls)
    self.assertAllEqual(zeros, ps.smart_where(np.array(False), fn1, fn2))
    self.assertEqual([2, 3], fn_calls)
    self.assertAllEqual(ones, ps.smart_where(np.array(True), fn1, fn2))
    self.assertEqual([3, 3], fn_calls)

    self.assertAllEqual(zeros, ps.smart_where(0, fn1, fn2))
    self.assertEqual([3, 4], fn_calls)
    self.assertAllEqual(ones, ps.smart_where(1, fn1, fn2))
    self.assertEqual([4, 4], fn_calls)
    self.assertAllEqual(zeros, ps.smart_where(tf.constant(0), fn1, fn2))
    self.assertEqual([4, 5], fn_calls)
    self.assertAllEqual(ones, ps.smart_where(tf.constant(1), fn1, fn2))
    self.assertEqual([5, 5], fn_calls)
    self.assertAllEqual(zeros, ps.smart_where(np.array(0), fn1, fn2))
    self.assertEqual([5, 6], fn_calls)
    self.assertAllEqual(ones, ps.smart_where(np.array(1), fn1, fn2))
    self.assertEqual([6, 6], fn_calls)

  def test_cond_x_broadcast_error(self):
    with self.assertRaisesOpError('Incompatible shapes'):
      self.evaluate(
          ps.smart_where(
              tf.constant([True, True]), lambda: tf.zeros([3]), lambda: None))

  def test_cond_y_broadcast_error(self):
    with self.assertRaisesOpError('Incompatible shapes'):
      self.evaluate(
          ps.smart_where(
              tf.constant([False, False]), lambda: None, lambda: tf.zeros([3])))

  def test_broadcast_success(self):
    self.assertAllEqual(
        tf.zeros([10, 2]),
        ps.smart_where(
            tf.constant([True, True]), lambda: tf.zeros([10, 1]), lambda: None))
    self.assertAllEqual(
        tf.ones([2, 10]),
        ps.smart_where(
            tf.constant([[False], [False]]),
            lambda: None, lambda: tf.ones([10])))

  def test_where_fallback(self):
    self.assertAllEqual([1., 0.],
                        ps.smart_where(
                            tf.constant([True, False]),
                            lambda: tf.ones([]),
                            lambda: tf.zeros([])))


@test_util.test_all_tf_execution_regimes
class DtypeTest(test_util.TestCase):

  def test_ones_like(self):
    x = tf1.placeholder_with_default(tf.ones([2], dtype=tf.float32),
                                     shape=None)
    self.assertEqual(dtype_util.convert_to_dtype(ps.ones_like(x)),
                     tf.float32)


JAX_MODE = False
if JAX_MODE:
  class JaxTest(test_util.TestCase):

    def testShape(self):

      @tf.function(jit_compile=True)
      def f(x):
        return tf.zeros(ps.shape(x))

      x = tf.zeros([3, 5])
      self.assertEqual((3, 5), f(x).shape)

    def testConcat(self):

      @tf.function(jit_compile=True)
      def f(x):
        return tf.zeros(ps.concat([ps.shape(x), [7]], 0))

      x = tf.zeros([3, 5])
      self.assertEqual((3, 5, 7), f(x).shape)


if __name__ == '__main__':
  test_util.main()
