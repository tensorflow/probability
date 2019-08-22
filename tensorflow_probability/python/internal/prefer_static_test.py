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
"""Tests for tensorflow_probability.prefer_static."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np

import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import prefer_static
from tensorflow_probability.python.internal import test_case
from tensorflow_probability.python.internal import test_util as tfp_test_util
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top


def raise_exception():
  raise RuntimeError('Did not expect to be called')


def raise_exception_in_eager_mode(value):
  def f():
    if tf.executing_eagerly():
      raise_exception()
    return value
  return f


@test_util.run_all_in_graph_and_eager_modes
class GetStaticValueTest(test_case.TestCase, parameterized.TestCase):

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
    static_predicate = prefer_static._get_static_predicate(predicate)

    if tf.executing_eagerly() or expect_static_graph_evaluation:
      # If we are in eager mode, we always expect static evaluation.
      self.assertIsNotNone(static_predicate)
      self.assertEqual(expected_truth_value, static_predicate)
    else:
      self.assertIsNone(static_predicate)


@test_util.run_all_in_graph_and_eager_modes
class PredicatesTest(test_case.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name='_greater_true',
           predicate=prefer_static.greater,
           args_fn=lambda: [tf.constant(1), tf.constant(0)],
           kwargs=dict(),
           expected=True),
      dict(testcase_name='_greater_false',
           predicate=prefer_static.greater,
           args_fn=lambda: [tf.constant(-.1), tf.constant(0.)],
           kwargs=dict(),
           expected=False),
      dict(testcase_name='_greater_none',
           predicate=prefer_static.greater,
           args_fn=lambda: [tf.constant(1) + tf.constant(0), tf.constant(0)],
           kwargs=dict(),
           expected=True),
      dict(testcase_name='_less_true',
           predicate=prefer_static.less,
           args_fn=lambda: [tf.constant(-1), tf.constant(0)],
           kwargs=dict(),
           expected=True),
      dict(testcase_name='_log',
           predicate=prefer_static.log,
           args_fn=lambda: [tf.constant(1.)],
           kwargs=dict(),
           expected=0.),
      dict(testcase_name='_equal_true',
           predicate=prefer_static.equal,
           args_fn=lambda: [tf.constant(0), 0],
           kwargs=dict(),
           expected=True),
      dict(testcase_name='_equal_false',
           predicate=prefer_static.equal,
           args_fn=lambda: [tf.constant(1), tf.constant(0)],
           kwargs=dict(),
           expected=False),
      dict(testcase_name='_and_true',
           predicate=prefer_static.logical_and,
           args_fn=lambda: [True, tf.constant(True)],
           kwargs=dict(),
           expected=True),
      dict(testcase_name='_and_none',
           predicate=prefer_static.logical_and,
           args_fn=lambda: [tf.constant(True), tf.equal(1, 1)],
           kwargs=dict(),
           expected=True),
      dict(testcase_name='_or_true',
           predicate=prefer_static.logical_or,
           args_fn=lambda: [tf.constant(True), tf.constant(False)],
           kwargs=dict(),
           expected=True),
      dict(testcase_name='_all_true',
           predicate=prefer_static.reduce_all,
           args_fn=lambda: [[tf.constant(True)] * 10],
           kwargs=dict(),
           expected=True),
      dict(testcase_name='_all_false',
           predicate=prefer_static.reduce_all,
           args_fn=lambda: [[tf.constant(True),  # pylint: disable=g-long-lambda
                             True,
                             tf.constant(False)]],
           kwargs=dict(),
           expected=False),
      dict(testcase_name='_all_with_axis',
           predicate=prefer_static.reduce_all,
           args_fn=lambda: ([[True, tf.constant(True)],  # pylint: disable=g-long-lambda
                             [False, True]],),
           kwargs=dict(axis=1),
           expected=[True, False]),
      dict(testcase_name='_all_with_name',
           predicate=prefer_static.reduce_all,
           args_fn=lambda: ([[True, tf.constant(True)],  # pylint: disable=g-long-lambda
                             [False, True]],),
           kwargs=dict(axis=1, name='my_name'),
           expected=[True, False]),
      dict(testcase_name='_any_true',
           predicate=prefer_static.reduce_any,
           args_fn=lambda: [[tf.constant(True),  # pylint: disable=g-long-lambda
                             tf.constant(False),
                             tf.constant(False)]],
           kwargs=dict(),
           expected=True),
      dict(testcase_name='_any_false',
           predicate=prefer_static.reduce_any,
           args_fn=lambda: [[tf.constant(False)] * 23],
           kwargs=dict(),
           expected=False),
      dict(testcase_name='_any_keepdims',
           predicate=prefer_static.reduce_any,
           args_fn=lambda: ([[True, tf.constant(True)],  # pylint: disable=g-long-lambda
                             [False, True]],),
           kwargs=dict(keepdims=True),
           expected=[[True]]),
      )
  def test_static_predicate(self, predicate, args_fn, kwargs, expected):
    actual = predicate(*args_fn(), **kwargs)
    self.assertAllCloseAccordingToType(expected, actual)


@test_util.run_all_in_graph_and_eager_modes
class CondTest(test_case.TestCase, parameterized.TestCase):

  def test_true(self):
    x = tf.constant(2)
    y = tf.constant(5)
    z = prefer_static.cond(True, lambda: tf.multiply(x, 16),
                           lambda: tf.multiply(y, 5))
    self.assertEqual(self.evaluate(z), 32)

  def test_false(self):
    x = tf.constant(4)
    y = tf.constant(3)
    z = prefer_static.cond(False, lambda: tf.multiply(x, 16),
                           lambda: tf.multiply(y, 3))
    self.assertEqual(self.evaluate(z), 9)

  def test_missing_arg1(self):
    x = tf.constant(1)
    with self.assertRaises(TypeError):
      prefer_static.cond(True, false_fn=lambda: x)

  def test_missing_arg2(self):
    x = tf.constant(1)
    with self.assertRaises(TypeError):
      prefer_static.cond(True, lambda: x)


@test_util.run_all_in_graph_and_eager_modes
class CaseTest(test_case.TestCase):

  def test_true(self):
    x = tf.constant(0)
    conditions = [(True, lambda: tf.constant(1)),
                  (tf.equal(x, 1), raise_exception)]
    y = prefer_static.case(conditions, default=raise_exception,
                           exclusive=False)
    z = prefer_static.case(conditions, default=raise_exception,
                           exclusive=True)

    self.assertEqual(self.evaluate(y), 1)
    self.assertEqual(self.evaluate(z), 1)

  def test_false(self):
    conditions = [(False, raise_exception)]
    y = prefer_static.case(conditions,
                           default=lambda: tf.constant(1),
                           exclusive=False)
    z = prefer_static.case(conditions,
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
    z = prefer_static.case(conditions, default=lambda: raise_exception)
    self.assertEqual(self.evaluate(z), 3)


@test_util.run_all_in_graph_and_eager_modes
class ShapeTest(test_case.TestCase):

  def test_shape(self):
    vector_value = [0., 1.]

    # case: numpy input
    self.assertAllEqual(prefer_static.shape(np.array(vector_value)), [2])

    # case: tensor input with static shape
    self.assertAllEqual(prefer_static.shape(tf.constant(vector_value)), [2])

    # case: tensor input with dynamic shape
    if not tf.executing_eagerly():
      shape = prefer_static.shape(tf1.placeholder_with_default(
          vector_value, shape=None))
      self.assertAllEqual(self.evaluate(shape), [2])

  def test_rank_from_shape(self):
    shape = [2, 4, 3]
    expected_rank = len(shape)
    v_ndarray = np.ones(shape)

    # case: shape_tensor is tuple
    rank = prefer_static.rank_from_shape(
        shape_tensor_fn=v_ndarray.shape)
    self.assertEqual(rank, expected_rank)

    # case: shape_tensor is ndarray
    rank = prefer_static.rank_from_shape(
        shape_tensor_fn=prefer_static.shape(v_ndarray))
    self.assertEqual(rank, expected_rank)

    # case: tensorshape is fully defined
    v_tensor = tf.convert_to_tensor(v_ndarray)
    rank = prefer_static.rank_from_shape(
        shape_tensor_fn=prefer_static.shape(v_tensor),
        tensorshape=v_tensor.shape)
    self.assertEqual(rank, expected_rank)

    if not tf.executing_eagerly():
      # case: tensorshape is unknown, rank cannot be statically inferred
      v_dynamic = tf1.placeholder_with_default(v_ndarray, shape=None)
      rank = prefer_static.rank_from_shape(
          shape_tensor_fn=lambda: prefer_static.shape(v_dynamic),
          tensorshape=v_dynamic.shape)
      self.assertEqual(self.evaluate(rank), expected_rank)

      # case: tensorshape is not provided, rank cannot be statically inferred
      rank = prefer_static.rank_from_shape(
          shape_tensor_fn=lambda: prefer_static.shape(v_dynamic))
      self.assertEqual(self.evaluate(rank), expected_rank)


@test_util.run_all_in_graph_and_eager_modes
class SetDiff1DTest(test_case.TestCase):

  def test_static(self):
    self.assertAllEqual(
        [0, 3, 4],
        prefer_static.setdiff1d(np.arange(5), [1, 2]))
    self.assertAllEqual(
        [],
        prefer_static.setdiff1d([], [1, 2]))
    self.assertAllEqual(
        [1, 2],
        prefer_static.setdiff1d([1, 2], []))

  def test_dynamic(self):
    if tf.executing_eagerly(): return
    x = tf1.placeholder_with_default(np.arange(5), shape=None)
    self.assertAllEqual(
        [0, 3, 4],
        self.evaluate(prefer_static.setdiff1d(x, [1, 2])))
    x = tf1.placeholder_with_default(np.array([], np.int32), shape=None)
    self.assertAllEqual(
        [],
        self.evaluate(prefer_static.setdiff1d(x, [1, 2])))
    self.assertAllEqual(
        [1, 2],
        self.evaluate(prefer_static.setdiff1d([1, 2], x)))


@test_util.run_all_in_graph_and_eager_modes
class SizeTest(test_case.TestCase):

  def test_static(self):
    self.assertAllEqual(
        3 * 4 * 5,
        prefer_static.size(
            tf.random.normal([3, 4, 5], seed=tfp_test_util.test_seed())))

  def test_dynamic(self):
    if tf.executing_eagerly(): return
    x = tf1.placeholder_with_default(
        tf.random.normal([3, 4, 5], seed=tfp_test_util.test_seed()), shape=None)
    self.assertAllEqual(
        3 * 4 * 5,
        self.evaluate(prefer_static.size(x)))


@test_util.run_all_in_graph_and_eager_modes
class NonNegativeAxisTest(test_case.TestCase):

  def test_static_scalar_positive_index(self):
    positive_axis = prefer_static.non_negative_axis(axis=2, rank=4)
    self.assertAllEqual(2, positive_axis)

  def test_static_scalar_negative_index(self):
    positive_axis = prefer_static.non_negative_axis(axis=-1, rank=4)
    self.assertAllEqual(3, positive_axis)

  def test_static_vector_index(self):
    positive_axis = prefer_static.non_negative_axis(axis=[0, -2], rank=4)
    self.assertAllEqual([0, 2], positive_axis)

  @tfp_test_util.jax_disable_variable_test
  def test_dynamic_vector_index(self):
    axis = tf.Variable([0, -2])
    positive_axis = prefer_static.non_negative_axis(axis=axis, rank=4)
    self.evaluate(axis.initializer)
    self.assertAllEqual([0, 2], self.evaluate(positive_axis))


@test_util.run_all_in_graph_and_eager_modes
class BroadcastShapeTest(test_case.TestCase):

  def test_static(self):
    self.assertAllEqual(
        (3, 4, 2, 5),
        prefer_static.broadcast_shape((4, 1, 1), (3, 1, 2, 5)))

    self.assertAllEqual((3, 4, 2, 5), prefer_static.broadcast_shape(
        tf.convert_to_tensor((4, 1, 1)), tf.convert_to_tensor((3, 1, 2, 5))))

  def test_dynamic(self):
    if tf.executing_eagerly():
      return

    shape = prefer_static.broadcast_shape(
        tf.convert_to_tensor([3, 2, 1]),
        tf.shape(tf1.placeholder_with_default(np.zeros((1, 5)),
                                              shape=(None, 5))))
    self.assertIsNone(tf.get_static_value(shape))
    self.assertAllEqual([3, 2, 5], self.evaluate(shape))


if __name__ == '__main__':
  tf.test.main()
