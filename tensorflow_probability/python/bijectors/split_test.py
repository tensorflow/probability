# Copyright 2020 The TensorFlow Probability Authors.
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
"""Tests for Split Bijector."""

# Dependency imports
from absl.testing import parameterized

import numpy as np
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
from tensorflow_probability.python import bijectors as tfb

from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.internal import test_util


class _SplitBijectorTest(object):
  """Base class for testing the split transformation.

  Methods defined in this class call a method self.build_input() that
  is implemented by subclasses defined below, returning respectively
   SplitBijectorTestStatic: static shapes,
   SplitBijectorTestDynamic: shape placeholders of known ndims, and
  so that each test in this base class is automatically run over all
  three cases. The subclasses also implement assertRaisesError to test
  for either Python exceptions (in the case of static shapes) or
  TensorFlow op errors (dynamic shapes).
  """

  def _testBijector(  # pylint: disable=invalid-name
      self, num_or_size_splits, expected_split_sizes, shape_in, axis):
    """Do a basic sanity check of forward, inverse, jacobian."""
    num_or_size_splits = self.build_input(num_or_size_splits)
    bijector = tfb.Split(num_or_size_splits, axis=axis, validate_args=True)

    self.assertStartsWith(bijector.name, 'split')
    x = np.random.rand(*shape_in)
    y = tf.split(x, num_or_size_splits, axis=axis)

    self.assertAllClose(
        self.evaluate(y),
        self.evaluate(bijector.forward(x)), atol=0., rtol=1e-2)
    self.assertAllClose(
        x, self.evaluate(bijector.inverse(y)), atol=0., rtol=1e-4)

    shape_out = []
    for d in expected_split_sizes:
      s = shape_in[:]
      s[axis] = d
      shape_out.append(self.build_input(s))

    shape_in_ = self.evaluate(bijector.inverse_event_shape_tensor(shape_out))
    self.assertAllEqual(shape_in_, shape_in)

    shape_in = self.build_input(shape_in)
    shape_out_ = self.evaluate(bijector.forward_event_shape_tensor(shape_in))
    self.assertAllEqual(shape_out_, self.evaluate(shape_out))

    event_ndims = abs(axis)
    inverse_event_ndims = [event_ndims for _ in expected_split_sizes]
    self.assertEqual(
        0.,
        self.evaluate(bijector.inverse_log_det_jacobian(
            y, event_ndims=inverse_event_ndims)))
    self.assertEqual(
        0.,
        self.evaluate(bijector.forward_log_det_jacobian(
            x, event_ndims=event_ndims)))

  def testCompositeTensor(self):
    split_sizes = self.build_input([1, 2, 2])
    bijector = tfb.Split(split_sizes, validate_args=True)
    x = tf.ones([3, 2, 5])
    flat = tf.nest.flatten(bijector, expand_composites=True)
    unflat = tf.nest.pack_sequence_as(bijector, flat, expand_composites=True)
    self.assertAllClose(
        bijector.forward(x),
        tf.function(lambda b_: b_.forward(x))(unflat))

  def testAssertRaisesNonVectorSplitSizes(self):
    split_sizes = self.build_input([[1, 2, 2]])
    with self.assertRaisesRegexp(ValueError, 'must be an integer or 1-D'):
      tfb.Split(split_sizes, validate_args=True)

  def testAssertRaisesWrongNumberOfOutputs(self):
    split_sizes = self.build_input([5, 3, -1])
    y = [np.random.rand(2, i) for i in [5, 3, 1, 2]]
    bijector = tfb.Split(split_sizes, validate_args=True)
    with self.assertRaisesRegexp(
        ValueError, "don't have the same sequence length"):
      self.evaluate(bijector.inverse(y))

  def testAssertRaisesNumSplitsNonDivisible(self):
    num_splits = 3
    x = np.random.rand(4, 5, 6)
    bijector = tfb.Split(num_splits, axis=-2, validate_args=True)
    with self.assertRaisesRegexp(ValueError, 'number of splits'):
      self.evaluate(bijector.forward(x))

  def testAssertRaisesWrongNumSplits(self):
    num_splits = 4
    y = [np.random.rand(2, 3)] * 3
    bijector = tfb.Split(num_splits, validate_args=True)
    with self.assertRaisesRegexp(
        ValueError, "don't have the same sequence length"):
      self.evaluate(bijector.inverse(y))

  # pylint: disable=invalid-name
  def _testAssertRaisesMultipleUnknownSplitSizes(self):
    split_sizes = self.build_input([-1, 4, -1, 8])
    with self.assertRaisesError('must have at most one'):
      bijector = tfb.Split(split_sizes, validate_args=True)
      self.evaluate(bijector.forward(tf.zeros((3, 14))))

  def _testAssertRaisesNegativeSplitSizes(self):
    split_sizes = self.build_input([-2, 3, 5])
    with self.assertRaisesError('must be either non-negative integers or'):
      bijector = tfb.Split(split_sizes, validate_args=True)
      self.evaluate(bijector.forward(tf.zeros((4, 10))))

  def _testAssertRaisesMismatchedInputShape(self):
    split_sizes = self.build_input([5, 3, 4])
    x = tf.Variable(tf.zeros((3, 10)), shape=None)
    self.evaluate(x.initializer)
    bijector = tfb.Split(split_sizes, validate_args=True)
    with self.assertRaisesError('size of the input along `axis`'):
      self.evaluate(bijector.forward(x))

  def _testAssertRaisesTooSmallInputShape(self):
    split_sizes = self.build_input([-1, 2, 3])
    x = tf.Variable(tf.zeros((2, 4)), shape=None)
    self.evaluate(x.initializer)
    bijector = tfb.Split(split_sizes, validate_args=True)
    with self.assertRaisesError('size of the input along `axis`'):
      self.evaluate(bijector.forward(x))

  def _testAssertRaisesMismatchedOutputShapes(self):
    split_sizes = self.build_input([5, -1, 3])
    y = [np.random.rand(3, 1, i) for i in [6, 2, 3]]
    bijector = tfb.Split(split_sizes, validate_args=True)
    if tf.get_static_value(split_sizes) is not None:
      with self.assertRaisesError('does not match expected `split_size`'):
        self.evaluate(bijector.inverse(y))
  # pylint: enable=invalid-name


@test_util.test_all_tf_execution_regimes
class SplitBijectorTestStatic(test_util.TestCase, _SplitBijectorTest):

  def build_input(self, x):
    if isinstance(x, int):
      return x
    return tf.convert_to_tensor(x)

  def assertRaisesError(self, msg):
    return self.assertRaisesRegexp(Exception, msg)

  @parameterized.named_parameters(
      {'testcase_name': 'fully_determined_split_sizes',
       'num_or_size_splits': [1, 3, 2, 1, 5],
       'expected_split_sizes': [1, 3, 2, 1, 5],
       'shape_in': [2, 5, 12],
       'axis': -1},
      {'testcase_name': 'unknown_split_size',
       'num_or_size_splits': [2, -1],
       'expected_split_sizes': [2, 3],
       'shape_in': [2, 5, 12],
       'axis': -2},
      {'testcase_name': 'num_splits',
       'num_or_size_splits': 3,
       'expected_split_sizes': [5, 5, 5],
       'shape_in': [4, 15],
       'axis': -1})
  def testBijector(
      self, num_or_size_splits, expected_split_sizes, shape_in, axis):
    self._testBijector(
        num_or_size_splits, expected_split_sizes, shape_in, axis)

  @parameterized.named_parameters(
      {'testcase_name': 'fully_determined',
       'num_or_size_splits': [2, 3, 2],
       'expected_split_sizes': [2, 3, 2]},
      {'testcase_name': 'unknown_split_size',
       'num_or_size_splits': [-1, 3, 2],
       'expected_split_sizes': [2, 3, 2]},
      {'testcase_name': 'num_splits',
       'num_or_size_splits': 4,
       'expected_split_sizes': [2, 2, 2, 2]})
  def testEventShape(self, num_or_size_splits, expected_split_sizes):
    num_or_size_splits = self.build_input(num_or_size_splits)
    total_size = np.sum(expected_split_sizes)
    shape_in_static = tf.TensorShape([total_size, 2])
    shape_out_static = [
        tf.TensorShape([d, 2]) for d in expected_split_sizes]
    bijector = tfb.Split(
        num_or_size_splits=num_or_size_splits, axis=-2, validate_args=True)

    # Test that forward_ and inverse_event_shape are correct when
    # event_shape_in/_out are statically known, even when the input shapes
    # are only partially specified.
    self.assertAllEqual(
        bijector.forward_event_shape(shape_in_static), shape_out_static)
    self.assertEqual(
        bijector.inverse_event_shape(shape_out_static), shape_in_static)

    # Shape is always known for splitting in eager mode, so we skip these tests.
    if tf.executing_eagerly():
      return
    self.assertAllEqual(
        [s.as_list() for s in bijector.forward_event_shape(
            tf.TensorShape([total_size, None]))],
        [[d, None] for d in expected_split_sizes])

    if bijector.split_sizes is None:
      static_split_sizes = tensorshape_util.constant_value_as_shape(
          expected_split_sizes).as_list()
    else:
      static_split_sizes = tensorshape_util.constant_value_as_shape(
          num_or_size_splits).as_list()

    static_total_size = None if None in static_split_sizes else total_size

    # Test correctness with an inverse input dimension of None that coincides
    # with the `-1` element in not-fully specified `split_sizes`
    shape_with_maybe_unknown_dim = (
        [[None, 3]] + [[d, 3] for d in expected_split_sizes[1:]])
    self.assertAllEqual(
        bijector.inverse_event_shape(shape_with_maybe_unknown_dim).as_list(),
        [static_total_size, 3])

    # Test correctness with an input dimension of None that does not coincide
    # with a `-1` split_size.
    shape_with_deducable_dim = [[d, 3] for d in expected_split_sizes]
    shape_with_deducable_dim[2] = [None, 3]
    self.assertAllEqual(
        bijector.inverse_event_shape(
            shape_with_deducable_dim).as_list(), [total_size, 3])

    # Test correctness for an input shape of known rank only.
    if bijector.split_sizes is not None:
      shape_with_unknown_total = (
          [[d, None] for d in static_split_sizes])
    else:
      shape_with_unknown_total = [[None, None]] * len(expected_split_sizes)
    self.assertAllEqual(
        [s.as_list() for s in bijector.forward_event_shape(
            tf.TensorShape([None, None]))],
        shape_with_unknown_total)

  def testAssertRaisesTooSmallInputShape(self):
    self._testAssertRaisesTooSmallInputShape()

  def testAssertRaisesMultipleUnknownSplitSizes(self):
    self._testAssertRaisesMultipleUnknownSplitSizes()

  def testAssertRaisesNegativeSplitSizes(self):
    self._testAssertRaisesNegativeSplitSizes()

  def testAssertRaisesMismatchedInputShape(self):
    self._testAssertRaisesMismatchedInputShape()

  def testAssertRaisesMismatchedOutputShapes(self):
    self._testAssertRaisesMismatchedOutputShapes()


@test_util.test_graph_mode_only
class SplitBijectorTestDynamic(test_util.TestCase, _SplitBijectorTest):

  def build_input(self, x):
    if isinstance(x, int):
      return x
    x = tf.convert_to_tensor(x)
    return tf1.placeholder_with_default(x, shape=x.shape)

  def assertRaisesError(self, msg):
    return self.assertRaisesOpError(msg)

  @parameterized.named_parameters(
      {'testcase_name': 'fully_determined_split_sizes',
       'num_or_size_splits': [1, 3, 2, 1, 5],
       'expected_split_sizes': [1, 3, 2, 1, 5],
       'shape_in': [2, 5, 12],
       'axis': -1},
      {'testcase_name': 'unknown_split_size',
       'num_or_size_splits': [2, -1],
       'expected_split_sizes': [2, 3],
       'shape_in': [2, 5, 12],
       'axis': -2},
      {'testcase_name': 'num_splits',
       'num_or_size_splits': 3,
       'expected_split_sizes': [5, 5, 5],
       'shape_in': [4, 15],
       'axis': -1})
  def testBijector(
      self, num_or_size_splits, expected_split_sizes, shape_in, axis):
    self._testBijector(
        num_or_size_splits, expected_split_sizes, shape_in, axis)

  @parameterized.named_parameters(
      {'testcase_name': 'fully_determined',
       'num_or_size_splits': [2, 3, 2],
       'expected_split_sizes': [2, 3, 2]},
      {'testcase_name': 'unknown_split_size',
       'num_or_size_splits': [-1, 3, 2],
       'expected_split_sizes': [2, 3, 2]})
  def testEventShape(self, num_or_size_splits, expected_split_sizes):
    split_sizes = self.build_input(num_or_size_splits)
    total_size = np.sum(expected_split_sizes)
    shape_in_static = tf.TensorShape([total_size, 2])
    shape_out_static = [
        tf.TensorShape([d, 2]) for d in expected_split_sizes]
    bijector = tfb.Split(
        num_or_size_splits=split_sizes, axis=-2, validate_args=True)

    output_shape = [[None, 2]] * 3
    self.assertAllEqual(
        [s.as_list() for s in bijector.forward_event_shape(shape_in_static)],
        output_shape)
    self.assertEqual(
        bijector.inverse_event_shape(shape_out_static).as_list(),
        shape_in_static.as_list())

    self.assertAllEqual(
        [s.as_list() for s in
         bijector.forward_event_shape(tf.TensorShape([total_size, None]))],
        [[None, None]] * 3)
    self.assertAllEqual(
        bijector.inverse_event_shape([[None, 3], [3, 3], [2, 3]]).as_list(),
        [None, 3])
    self.assertAllEqual(
        bijector.inverse_event_shape([[2, 3], [None, 3], [2, 3]]).as_list(),
        [None, 3])

  def testAssertRaisesUnknownNumSplits(self):
    split_sizes = tf1.placeholder_with_default([-1, 2, 1], shape=[None])
    with self.assertRaisesRegexp(
        ValueError, 'must have a statically-known number of elements'):
      tfb.Split(num_or_size_splits=split_sizes, validate_args=True)

  def testAssertRaisesTooSmallInputShape(self):
    self._testAssertRaisesTooSmallInputShape()

  def testAssertRaisesMultipleUnknownSplitSizes(self):
    self._testAssertRaisesMultipleUnknownSplitSizes()

  def testAssertRaisesNegativeSplitSizes(self):
    self._testAssertRaisesNegativeSplitSizes()

  def testAssertRaisesMismatchedInputShape(self):
    self._testAssertRaisesMismatchedInputShape()

  def testAssertRaisesMismatchedOutputShapes(self):
    self._testAssertRaisesMismatchedOutputShapes()

if __name__ == '__main__':
  test_util.main()
