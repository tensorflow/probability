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
"""Tests for distribution_utility functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np

import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.distributions import Categorical
from tensorflow_probability.python.distributions import Mixture
from tensorflow_probability.python.distributions import MixtureSameFamily
from tensorflow_probability.python.distributions import MultivariateNormalDiag
from tensorflow_probability.python.distributions import Normal
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.internal import test_util


def _logit(x):
  x = np.asarray(x)
  return np.log(x) - np.log1p(-x)


@test_util.test_all_tf_execution_regimes
class ShapesFromLocAndScaleTest(test_util.TestCase):

  def test_static_loc_static_scale_non_matching_event_size_raises(self):
    loc = tf.zeros([2, 4])
    diag = tf.ones([5, 1, 3])
    with self.assertRaisesRegexp(ValueError, 'could not be broadcast'):
      distribution_util.shapes_from_loc_and_scale(
          loc, tf.linalg.LinearOperatorDiag(diag))

  def test_static_loc_static_scale(self):
    loc = tf.zeros([2, 3])
    diag = tf.ones([5, 1, 3])
    batch_shape, event_shape = distribution_util.shapes_from_loc_and_scale(
        loc, tf.linalg.LinearOperatorDiag(diag))

    if not tf.executing_eagerly():
      self.assertAllEqual([5, 2], tf.get_static_value(batch_shape))
      self.assertAllEqual([3], tf.get_static_value(event_shape))

    batch_shape_, event_shape_ = self.evaluate([batch_shape, event_shape])
    self.assertAllEqual([5, 2], batch_shape_)
    self.assertAllEqual([3], event_shape_)

  def test_static_loc_dynamic_scale(self):
    loc = tf.zeros([2, 3])
    diag = tf1.placeholder_with_default(np.ones([5, 1, 3]), shape=None)
    batch_shape, event_shape = distribution_util.shapes_from_loc_and_scale(
        loc, tf.linalg.LinearOperatorDiag(diag))

    if not tf.executing_eagerly():
      # batch_shape depends on both args, and so is dynamic.  Since loc did not
      # have static shape, we inferred event shape entirely from scale, and this
      # is available statically.
      self.assertIsNone(tf.get_static_value(batch_shape))
      self.assertAllEqual([3], tf.get_static_value(event_shape))

    batch_shape_, event_shape_ = self.evaluate([batch_shape, event_shape])
    self.assertAllEqual([5, 2], batch_shape_)
    self.assertAllEqual([3], event_shape_)

  def test_dynamic_loc_static_scale(self):
    loc = tf1.placeholder_with_default(np.zeros([2, 3]), shape=None)
    diag = tf.ones([5, 2, 3])
    batch_shape, event_shape = distribution_util.shapes_from_loc_and_scale(
        loc, tf.linalg.LinearOperatorDiag(diag))

    if not tf.executing_eagerly():
      # batch_shape depends on both args, and so is dynamic.  Since loc did not
      # have static shape, we inferred event shape entirely from scale, and this
      # is available statically.
      self.assertIsNone(tf.get_static_value(batch_shape))
      self.assertAllEqual([3], tf.get_static_value(event_shape))

    batch_shape_, event_shape_ = self.evaluate([batch_shape, event_shape])
    self.assertAllEqual([5, 2], batch_shape_)
    self.assertAllEqual([3], event_shape_)

  def test_dynamic_loc_dynamic_scale(self):
    loc = tf1.placeholder_with_default(np.ones([2, 3]), shape=None)
    diag = tf1.placeholder_with_default(np.ones([5, 2, 3]), shape=None)
    batch_shape, event_shape = distribution_util.shapes_from_loc_and_scale(
        loc, tf.linalg.LinearOperatorDiag(diag))

    if not tf.executing_eagerly():
      self.assertIsNone(tf.get_static_value(batch_shape))
      self.assertIsNone(tf.get_static_value(event_shape))

    batch_shape_, event_shape_ = self.evaluate([batch_shape, event_shape])
    self.assertAllEqual([5, 2], batch_shape_)
    self.assertAllEqual([3], event_shape_)

  def test_none_loc_static_scale(self):
    loc = None
    diag = tf.ones([5, 1, 3])
    batch_shape, event_shape = distribution_util.shapes_from_loc_and_scale(
        loc, tf.linalg.LinearOperatorDiag(diag))

    if not tf.executing_eagerly():
      self.assertAllEqual([5, 1], tf.get_static_value(batch_shape))
      self.assertAllEqual([3], tf.get_static_value(event_shape))

    batch_shape_, event_shape_ = self.evaluate([batch_shape, event_shape])
    self.assertAllEqual([5, 1], batch_shape_)
    self.assertAllEqual([3], event_shape_)

  def test_none_loc_dynamic_scale(self):
    loc = None
    diag = tf1.placeholder_with_default(np.ones([5, 1, 3]), shape=None)
    batch_shape, event_shape = distribution_util.shapes_from_loc_and_scale(
        loc, tf.linalg.LinearOperatorDiag(diag))

    if not tf.executing_eagerly():
      self.assertIsNone(tf.get_static_value(batch_shape))
      self.assertIsNone(tf.get_static_value(event_shape))

    batch_shape_, event_shape_ = self.evaluate([batch_shape, event_shape])
    self.assertAllEqual([5, 1], batch_shape_)
    self.assertAllEqual([3], event_shape_)


@test_util.test_all_tf_execution_regimes
class GetBroadcastShapeTest(test_util.TestCase):

  def test_all_static_shapes_work(self):
    x = tf.ones((2, 1, 3))
    y = tf.ones((1, 5, 3))
    z = tf.ones(())
    self.assertAllEqual([2, 5, 3],
                        distribution_util.get_broadcast_shape(x, y, z))

  def test_with_some_dynamic_shapes_works(self):
    if tf.executing_eagerly(): return
    x = tf.ones([2, 1, 3])
    y = tf1.placeholder_with_default(
        np.ones([1, 5, 3], dtype=np.float32),
        shape=None)
    z = tf.ones([])
    bcast_shape = self.evaluate(distribution_util.get_broadcast_shape(x, y, z))
    self.assertAllEqual([2, 5, 3], bcast_shape)


@test_util.test_all_tf_execution_regimes
class MixtureStddevTest(test_util.TestCase):

  def test_mixture_dev(self):
    mixture_weights = np.array([
        [1.0/3, 1.0/3, 1.0/3],
        [0.750, 0.250, 0.000]
    ])
    component_means = np.array([
        [1.0, 1.0, 1.0],
        [-5, 0, 1.25]
    ])
    component_devs = np.array([
        [1.0, 1.0, 1.0],
        [0.01, 2.0, 0.1]
    ])

    # The first case should trivially have a standard deviation of 1.0 because
    # all components are identical and have that standard deviation.
    # The second case was computed by hand.
    expected_devs = np.array([
        1.0,
        2.3848637277
    ])

    weights_tf = tf.constant(mixture_weights)
    means_tf = tf.constant(component_means)
    sigmas_tf = tf.constant(component_devs)
    mix_dev = distribution_util.mixture_stddev(weights_tf,
                                               means_tf,
                                               sigmas_tf)

    self.assertAllClose(expected_devs, self.evaluate(mix_dev))


@test_util.test_all_tf_execution_regimes
class PadMixtureDimensionsTest(test_util.TestCase):

  def test_pad_mixture_dimensions_mixture(self):
    gm = Mixture(
        cat=Categorical(probs=[[0.3, 0.7]]),
        components=[
            Normal(loc=[-1.0], scale=[1.0]),
            Normal(loc=[1.0], scale=[0.5])
        ])

    x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    x_pad = distribution_util.pad_mixture_dimensions(
        x, gm, gm.cat, tensorshape_util.rank(gm.event_shape))
    x_out, x_pad_out = self.evaluate([x, x_pad])

    self.assertAllEqual(x_pad_out.shape, [2, 2])
    self.assertAllEqual(x_out.reshape([-1]), x_pad_out.reshape([-1]))

  def test_pad_mixture_dimensions_mixture_same_family(self):
    gm = MixtureSameFamily(
        mixture_distribution=Categorical(probs=[0.3, 0.7]),
        components_distribution=MultivariateNormalDiag(
            loc=[[-1., 1], [1, -1]], scale_identity_multiplier=[1.0, 0.5]))

    x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    x_pad = distribution_util.pad_mixture_dimensions(
        x, gm, gm.mixture_distribution, tensorshape_util.rank(gm.event_shape))
    x_out, x_pad_out = self.evaluate([x, x_pad])

    self.assertAllEqual(x_pad_out.shape, [2, 2, 1])
    self.assertAllEqual(x_out.reshape([-1]), x_pad_out.reshape([-1]))


class _PadTest(object):

  def testNegAxisCorrectness(self):
    x_ = np.float32([[1., 2, 3],
                     [4, 5, 6]])
    value_ = np.float32(0.25)
    count_ = np.int32(2)

    x = tf1.placeholder_with_default(
        x_, shape=x_.shape if self.is_static_shape else None)
    value = (
        tf.constant(value_) if self.is_static_shape else
        tf1.placeholder_with_default(value_, shape=None))
    count = (
        tf.constant(count_) if self.is_static_shape else
        tf1.placeholder_with_default(count_, shape=None))

    x0_front = distribution_util.pad(
        x, axis=-2, value=value, count=count, front=True)
    x0_back = distribution_util.pad(
        x, axis=-2, count=count, back=True)
    x0_both = distribution_util.pad(
        x, axis=-2, value=value, front=True, back=True)

    if self.is_static_shape:
      self.assertAllEqual([4, 3], x0_front.shape)
      self.assertAllEqual([4, 3], x0_back.shape)
      self.assertAllEqual([4, 3], x0_both.shape)

    [x0_front_, x0_back_, x0_both_] = self.evaluate([
        x0_front, x0_back, x0_both])

    self.assertAllClose(
        np.float32([[value_]*3,
                    [value_]*3,
                    [1, 2, 3],
                    [4, 5, 6]]),
        x0_front_, atol=0., rtol=1e-6)
    self.assertAllClose(
        np.float32([[1, 2, 3],
                    [4, 5, 6],
                    [0.]*3,
                    [0.]*3]),
        x0_back_, atol=0., rtol=1e-6)
    self.assertAllClose(
        np.float32([[value_]*3,
                    [1, 2, 3],
                    [4, 5, 6],
                    [value_]*3]),
        x0_both_, atol=0., rtol=1e-6)

  def testPosAxisCorrectness(self):
    x_ = np.float32([[1., 2, 3],
                     [4, 5, 6]])
    value_ = np.float32(0.25)
    count_ = np.int32(2)
    x = tf1.placeholder_with_default(
        x_, shape=x_.shape if self.is_static_shape else None)
    value = (
        tf.constant(value_) if self.is_static_shape else
        tf1.placeholder_with_default(value_, shape=None))
    count = (
        tf.constant(count_) if self.is_static_shape else
        tf1.placeholder_with_default(count_, shape=None))

    x1_front = distribution_util.pad(
        x, axis=1, value=value, count=count, front=True)
    x1_back = distribution_util.pad(
        x, axis=1, count=count, back=True)
    x1_both = distribution_util.pad(
        x, axis=1, value=value, front=True, back=True)

    if self.is_static_shape:
      self.assertAllEqual([2, 5], x1_front.shape)
      self.assertAllEqual([2, 5], x1_back.shape)
      self.assertAllEqual([2, 5], x1_both.shape)

    [x1_front_, x1_back_, x1_both_] = self.evaluate([
        x1_front, x1_back, x1_both])

    self.assertAllClose(
        np.float32([[value_]*2 + [1, 2, 3],
                    [value_]*2 + [4, 5, 6]]),
        x1_front_, atol=0., rtol=1e-6)
    self.assertAllClose(
        np.float32([[1, 2, 3] + [0.]*2,
                    [4, 5, 6] + [0.]*2]),
        x1_back_, atol=0., rtol=1e-6)
    self.assertAllClose(
        np.float32([[value_, 1, 2, 3, value_],
                    [value_, 4, 5, 6, value_]]),
        x1_both_, atol=0., rtol=1e-6)


@test_util.test_all_tf_execution_regimes
class PadStaticTest(_PadTest, test_util.TestCase):

  @property
  def is_static_shape(self):
    return True


@test_util.test_all_tf_execution_regimes
class PadDynamicTest(_PadTest, test_util.TestCase):

  @property
  def is_static_shape(self):
    return False


@test_util.test_all_tf_execution_regimes
class PickScalarConditionTest(test_util.TestCase):

  def test_pick_scalar_condition_static(self):

    pos = np.exp(np.random.randn(3, 2, 4)).astype(np.float32)
    neg = -np.exp(np.random.randn(3, 2, 4)).astype(np.float32)

    # Python static cond
    self.assertAllEqual(
        distribution_util.pick_scalar_condition(True, pos, neg), pos)
    self.assertAllEqual(
        distribution_util.pick_scalar_condition(False, pos, neg), neg)

    # TF static cond
    self.assertAllEqual(distribution_util.pick_scalar_condition(
        tf.constant(True), pos, neg), pos)
    self.assertAllEqual(distribution_util.pick_scalar_condition(
        tf.constant(False), pos, neg), neg)

  # Dynamic tests don't need to (/can't) run in Eager mode.
  def test_pick_scalar_condition_dynamic(self):
    pos = np.exp(np.random.randn(3, 2, 4)).astype(np.float32)
    neg = -np.exp(np.random.randn(3, 2, 4)).astype(np.float32)

    # TF dynamic cond
    dynamic_true = tf1.placeholder_with_default(True, shape=None)
    dynamic_false = tf1.placeholder_with_default(
        False, shape=None)
    pos_ = self.evaluate(distribution_util.pick_scalar_condition(
        dynamic_true, pos, neg))
    neg_ = self.evaluate(distribution_util.pick_scalar_condition(
        dynamic_false, pos, neg))
    self.assertAllEqual(pos_, pos)
    self.assertAllEqual(neg_, neg)

    # TF dynamic everything
    pos_dynamic = tf1.placeholder_with_default(pos, shape=None)
    neg_dynamic = tf1.placeholder_with_default(neg, shape=None)
    pos_ = self.evaluate(distribution_util.pick_scalar_condition(
        dynamic_true, pos_dynamic, neg_dynamic))
    neg_ = self.evaluate(distribution_util.pick_scalar_condition(
        dynamic_false, pos_dynamic, neg_dynamic))
    self.assertAllEqual(pos_, pos)
    self.assertAllEqual(neg_, neg)


@test_util.test_all_tf_execution_regimes
class TestMoveDimension(test_util.TestCase):

  def test_move_dimension_static_shape(self):

    x = tf.random.normal(
        shape=[200, 30, 4, 1, 6], seed=test_util.test_seed())

    x_perm = distribution_util.move_dimension(x, 1, 1)
    self.assertAllEqual(
        tensorshape_util.as_list(x_perm.shape), [200, 30, 4, 1, 6])

    x_perm = distribution_util.move_dimension(x, 0, 3)
    self.assertAllEqual(
        tensorshape_util.as_list(x_perm.shape), [30, 4, 1, 200, 6])

    x_perm = distribution_util.move_dimension(x, 0, -2)
    self.assertAllEqual(
        tensorshape_util.as_list(x_perm.shape), [30, 4, 1, 200, 6])

    x_perm = distribution_util.move_dimension(x, 4, 2)
    self.assertAllEqual(
        tensorshape_util.as_list(x_perm.shape), [200, 30, 6, 4, 1])

  def test_move_dimension_dynamic_shape(self):

    x_ = tf.random.normal(
        shape=[200, 30, 4, 1, 6], seed=test_util.test_seed())
    x = tf1.placeholder_with_default(x_, shape=None)

    x_perm1 = distribution_util.move_dimension(x, 1, 1)
    x_perm2 = distribution_util.move_dimension(x, 0, 3)
    x_perm3 = distribution_util.move_dimension(x, 0, -2)
    x_perm4 = distribution_util.move_dimension(x, 4, 2)
    x_perm5 = distribution_util.move_dimension(x, -1, 2)

    x_perm1_, x_perm2_, x_perm3_, x_perm4_, x_perm5_ = self.evaluate([
        tf.shape(x_perm1),
        tf.shape(x_perm2),
        tf.shape(x_perm3),
        tf.shape(x_perm4),
        tf.shape(x_perm5)
    ])

    self.assertAllEqual(x_perm1_, [200, 30, 4, 1, 6])

    self.assertAllEqual(x_perm2_, [30, 4, 1, 200, 6])

    self.assertAllEqual(x_perm3_, [30, 4, 1, 200, 6])

    self.assertAllEqual(x_perm4_, [200, 30, 6, 4, 1])

    self.assertAllEqual(x_perm5_, [200, 30, 6, 4, 1])

  def test_move_dimension_dynamic_indices(self):

    x_ = tf.random.normal(
        shape=[200, 30, 4, 1, 6], seed=test_util.test_seed())
    x = tf1.placeholder_with_default(x_, shape=None)

    x_perm1 = distribution_util.move_dimension(
        x, tf1.placeholder_with_default(1, shape=[]),
        tf1.placeholder_with_default(1, shape=[]))

    x_perm2 = distribution_util.move_dimension(
        x, tf1.placeholder_with_default(0, shape=[]),
        tf1.placeholder_with_default(3, shape=[]))

    x_perm3 = distribution_util.move_dimension(
        x, tf1.placeholder_with_default(0, shape=[]),
        tf1.placeholder_with_default(-2, shape=[]))

    x_perm4 = distribution_util.move_dimension(
        x, tf1.placeholder_with_default(4, shape=[]),
        tf1.placeholder_with_default(2, shape=[]))

    x_perm5 = distribution_util.move_dimension(
        x, tf1.placeholder_with_default(-1, shape=[]),
        tf1.placeholder_with_default(2, shape=[]))

    x_perm1_, x_perm2_, x_perm3_, x_perm4_, x_perm5_ = self.evaluate([
        tf.shape(x_perm1),
        tf.shape(x_perm2),
        tf.shape(x_perm3),
        tf.shape(x_perm4),
        tf.shape(x_perm5)
    ])

    self.assertAllEqual(x_perm1_, [200, 30, 4, 1, 6])

    self.assertAllEqual(x_perm2_, [30, 4, 1, 200, 6])

    self.assertAllEqual(x_perm3_, [30, 4, 1, 200, 6])

    self.assertAllEqual(x_perm4_, [200, 30, 6, 4, 1])

    self.assertAllEqual(x_perm5_, [200, 30, 6, 4, 1])


@test_util.test_all_tf_execution_regimes
class AssertCloseTest(test_util.TestCase):

  def testAssertIntegerForm(self):
    # This should only be detected as an integer.
    x = tf1.placeholder_with_default(
        np.array([1., 5, 10, 15, 20], dtype=np.float32), shape=None)
    y = tf1.placeholder_with_default(
        np.array([1.1, 5, 10, 15, 20], dtype=np.float32), shape=None)
    # First component isn't less than float32.eps = 1e-7
    z = tf1.placeholder_with_default(
        np.array([1.0001, 5, 10, 15, 20], dtype=np.float32), shape=None)
    # This shouldn't be detected as an integer.
    w = tf1.placeholder_with_default(
        np.array([1e-8, 5, 10, 15, 20], dtype=np.float32), shape=None)

    with tf.control_dependencies([distribution_util.assert_integer_form(x)]):
      self.evaluate(tf.identity(x))

    with self.assertRaisesOpError('has non-integer components'):
      with tf.control_dependencies(
          [distribution_util.assert_integer_form(y)]):
        self.evaluate(tf.identity(y))

    with self.assertRaisesOpError('has non-integer components'):
      with tf.control_dependencies(
          [distribution_util.assert_integer_form(z)]):
        self.evaluate(tf.identity(z))

    with self.assertRaisesOpError('has non-integer components'):
      with tf.control_dependencies(
          [distribution_util.assert_integer_form(w)]):
        self.evaluate(tf.identity(w))


@test_util.test_all_tf_execution_regimes
class MaybeGetStaticTest(test_util.TestCase):

  def testGetStaticInt(self):
    x = 2
    self.assertEqual(x, distribution_util.maybe_get_static_value(x))
    self.assertAllClose(
        np.array(2.),
        distribution_util.maybe_get_static_value(x, dtype=np.float64))

  def testGetStaticNumpyArray(self):
    x = np.array(2, dtype=np.int32)
    self.assertEqual(x, distribution_util.maybe_get_static_value(x))
    self.assertAllClose(
        np.array(2.),
        distribution_util.maybe_get_static_value(x, dtype=np.float64))

  def testGetStaticConstant(self):
    x = tf.constant(2, dtype=tf.int32)
    self.assertEqual(np.array(2, dtype=np.int32),
                     distribution_util.maybe_get_static_value(x))
    self.assertAllClose(
        np.array(2.),
        distribution_util.maybe_get_static_value(x, dtype=np.float64))

  def testGetStaticPlaceholder(self):
    if tf.executing_eagerly(): return
    x = tf1.placeholder_with_default(
        np.array([2.], dtype=np.int32), shape=[1])
    self.assertEqual(None, distribution_util.maybe_get_static_value(x))
    self.assertEqual(
        None, distribution_util.maybe_get_static_value(x, dtype=np.float64))


@test_util.test_all_tf_execution_regimes
class EmbedCheckCategoricalEventShapeTest(test_util.TestCase):

  def testTooSmall(self):
    with self.assertRaises(ValueError):
      param = tf.ones([1], dtype=np.float16)
      checked_param = distribution_util.embed_check_categorical_event_shape(
          param)

    if tf.executing_eagerly(): return
    with self.assertRaisesOpError(
        'must have at least 2 events'):
      param = tf1.placeholder_with_default(
          np.ones([1], dtype=np.float16), shape=None)
      checked_param = distribution_util.embed_check_categorical_event_shape(
          param)
      self.evaluate(checked_param)

  def testTooLarge(self):
    with self.assertRaises(ValueError):
      param = tf.ones([int(2**11+1)], dtype=tf.float16)
      checked_param = distribution_util.embed_check_categorical_event_shape(
          param)

    if tf.executing_eagerly(): return
    with self.assertRaisesOpError(
        'Number of classes exceeds `dtype` precision'):
      param = tf1.placeholder_with_default(
          np.ones([int(2**11+1)], dtype=np.float16), shape=None)
      checked_param = distribution_util.embed_check_categorical_event_shape(
          param)
      self.evaluate(checked_param)

  def testUnsupportedDtype(self):
    param = tf.convert_to_tensor(
        value=np.ones([2**11 + 1]).astype(tf.qint16.as_numpy_dtype),
        dtype=tf.qint16)
    with self.assertRaises(TypeError):
      distribution_util.embed_check_categorical_event_shape(param)


@test_util.test_all_tf_execution_regimes
class EmbedCheckIntegerCastingClosedTest(test_util.TestCase):

  def testCorrectlyAssertsNonnegative(self):
    with self.assertRaisesOpError('Elements must be non-negative'):
      x = tf1.placeholder_with_default(
          np.array([1, -1], dtype=np.float16), shape=None)
      x_checked = distribution_util.embed_check_integer_casting_closed(
          x, target_dtype=tf.int16)
      self.evaluate(x_checked)

  def testCorrectlyAssertsPositive(self):
    with self.assertRaisesOpError('Elements must be positive'):
      x = tf1.placeholder_with_default(
          np.array([1, 0], dtype=np.float16), shape=None)
      x_checked = distribution_util.embed_check_integer_casting_closed(
          x, target_dtype=tf.int16, assert_positive=True)
      self.evaluate(x_checked)

  def testCorrectlyAssersIntegerForm(self):
    with self.assertRaisesOpError('Elements must be int16-equivalent.'):
      x = tf1.placeholder_with_default(
          np.array([1, 1.5], dtype=np.float16), shape=None)
      x_checked = distribution_util.embed_check_integer_casting_closed(
          x, target_dtype=tf.int16)
      self.evaluate(x_checked)

  def testCorrectlyAssertsLargestPossibleInteger(self):
    with self.assertRaisesOpError('Elements cannot exceed 32767.'):
      x = tf1.placeholder_with_default(
          np.array([1, 2**15], dtype=np.int32), shape=None)
      x_checked = distribution_util.embed_check_integer_casting_closed(
          x, target_dtype=tf.int16)
      self.evaluate(x_checked)

  def testCorrectlyAssertsSmallestPossibleInteger(self):
    with self.assertRaisesOpError('Elements cannot be smaller than 0.'):
      x = tf1.placeholder_with_default(
          np.array([1, -1], dtype=np.int32), shape=None)
      x_checked = distribution_util.embed_check_integer_casting_closed(
          x, target_dtype=tf.uint16, assert_nonnegative=False)
      self.evaluate(x_checked)


@test_util.test_all_tf_execution_regimes
class DynamicShapeTest(test_util.TestCase):

  def testSameDynamicShape(self):
    scalar = tf.constant(2.)
    scalar1 = tf1.placeholder_with_default(
        np.array(2., dtype=np.float32), shape=None)

    vector = tf.constant([0.3, 0.4, 0.5])
    vector1 = tf1.placeholder_with_default(
        np.array([2., 3., 4.], dtype=np.float32), shape=[None])
    vector2 = tf1.placeholder_with_default(
        np.array([2., 3.5, 6.], dtype=np.float32), shape=[None])

    multidimensional = tf.constant([[0.3, 0.4], [0.2, 0.6]])
    multidimensional1 = tf1.placeholder_with_default(
        np.array([[2., 3.], [3., 4.]], dtype=np.float32),
        shape=[None, None])
    multidimensional2 = tf1.placeholder_with_default(
        np.array([[1., 3.5], [6.3, 2.3]], dtype=np.float32),
        shape=[None, None])
    multidimensional3 = tf1.placeholder_with_default(
        np.array([[1., 3.5, 5.], [6.3, 2.3, 7.1]], dtype=np.float32),
        shape=[None, None])

    # Scalar
    self.assertTrue(self.evaluate(
        distribution_util.same_dynamic_shape(scalar, scalar1)))

    # Vector
    self.assertTrue(self.evaluate(
        distribution_util.same_dynamic_shape(vector, vector1)))
    self.assertTrue(self.evaluate(
        distribution_util.same_dynamic_shape(vector1, vector2)))

    # Multidimensional
    self.assertTrue(self.evaluate(
        distribution_util.same_dynamic_shape(
            multidimensional, multidimensional1)))
    self.assertTrue(self.evaluate(
        distribution_util.same_dynamic_shape(
            multidimensional1, multidimensional2)))

    # Scalar, X
    self.assertFalse(self.evaluate(
        distribution_util.same_dynamic_shape(scalar, vector1)))
    self.assertFalse(self.evaluate(
        distribution_util.same_dynamic_shape(scalar1, vector1)))
    self.assertFalse(self.evaluate(
        distribution_util.same_dynamic_shape(scalar, multidimensional1)))
    self.assertFalse(self.evaluate(
        distribution_util.same_dynamic_shape(scalar1, multidimensional1)))

    # Vector, X
    self.assertFalse(self.evaluate(
        distribution_util.same_dynamic_shape(vector, vector1[:2])))
    self.assertFalse(self.evaluate(
        distribution_util.same_dynamic_shape(vector1, vector2[-1:])))
    self.assertFalse(self.evaluate(
        distribution_util.same_dynamic_shape(vector, multidimensional1)))
    self.assertFalse(self.evaluate(
        distribution_util.same_dynamic_shape(vector1, multidimensional1)))

    # Multidimensional, X
    self.assertFalse(self.evaluate(
        distribution_util.same_dynamic_shape(
            multidimensional, multidimensional3)))
    self.assertFalse(self.evaluate(
        distribution_util.same_dynamic_shape(
            multidimensional1, multidimensional3)))


@test_util.test_all_tf_execution_regimes
class RotateTransposeTest(test_util.TestCase):

  def _np_rotate_transpose(self, x, shift):
    if not isinstance(x, np.ndarray):
      x = np.array(x)
    return np.transpose(x, np.roll(np.arange(len(x.shape)), shift))

  def testRollStatic(self):
    with self.assertRaisesRegexp(Exception, 'None'):
      distribution_util.rotate_transpose(None, 1)
    for x in (np.ones(1), np.ones((2, 1)), np.ones((3, 2, 1))):
      for shift in np.arange(-5, 5):
        y = distribution_util.rotate_transpose(x, shift)
        self.assertAllEqual(
            self._np_rotate_transpose(x, shift), self.evaluate(y))
        self.assertAllEqual(
            np.roll(x.shape, shift), tensorshape_util.as_list(y.shape))

  def testRollDynamic(self):
    for x_value in (np.ones(1, dtype=np.float32),
                    np.ones([2, 1], dtype=np.float32),
                    np.ones([3, 2, 1], dtype=np.float32)):
      for shift_value in np.arange(-5, 5).astype(np.int32):
        x = tf1.placeholder_with_default(x_value, shape=None)
        shift = tf1.placeholder_with_default(shift_value, shape=None)
        self.assertAllEqual(
            self._np_rotate_transpose(x_value, shift_value),
            self.evaluate(distribution_util.rotate_transpose(x, shift)))


@test_util.test_all_tf_execution_regimes
class PickVectorTest(test_util.TestCase):

  def testCorrectlyPicksVector(self):
    x = np.arange(10, 12)
    y = np.arange(15, 18)
    self.assertAllEqual(
        x, self.evaluate(distribution_util.pick_vector(tf.less(0, 5), x, y)))
    self.assertAllEqual(
        y, self.evaluate(distribution_util.pick_vector(tf.less(5, 0), x, y)))
    self.assertAllEqual(x,
                        distribution_util.pick_vector(
                            tf.constant(True), x, y))  # No eval.
    self.assertAllEqual(y,
                        distribution_util.pick_vector(
                            tf.constant(False), x, y))  # No eval.


@test_util.test_all_tf_execution_regimes
class PreferStaticRankTest(test_util.TestCase):

  def testNonEmptyConstantTensor(self):
    x = tf.zeros([2, 3, 4])
    rank = distribution_util.prefer_static_rank(x)
    if not tf.executing_eagerly():
      self.assertIsInstance(rank, np.ndarray)
    self.assertEqual(3, rank)

  def testEmptyConstantTensor(self):
    x = tf.constant([])
    rank = distribution_util.prefer_static_rank(x)
    if not tf.executing_eagerly():
      self.assertIsInstance(rank, np.ndarray)
    self.assertEqual(1, rank)

  def testScalarTensor(self):
    x = tf.constant(1.)
    rank = distribution_util.prefer_static_rank(x)
    if not tf.executing_eagerly():
      self.assertIsInstance(rank, np.ndarray)
    self.assertEqual(0, rank)

  def testDynamicRankEndsUpBeingNonEmpty(self):
    if tf.executing_eagerly(): return
    x = tf1.placeholder_with_default(
        np.zeros([2, 3], dtype=np.float64), shape=None)
    rank = distribution_util.prefer_static_rank(x)
    self.assertAllEqual(2, self.evaluate(rank))

  def testDynamicRankEndsUpBeingEmpty(self):
    if tf.executing_eagerly(): return
    x = tf1.placeholder_with_default(
        np.array([], dtype=np.int32), shape=None)
    rank = distribution_util.prefer_static_rank(x)
    self.assertAllEqual(1, self.evaluate(rank))

  def testDynamicRankEndsUpBeingScalar(self):
    if tf.executing_eagerly(): return
    x = tf1.placeholder_with_default(
        np.array(1, dtype=np.int32), shape=None)
    rank = distribution_util.prefer_static_rank(x)
    self.assertAllEqual(0, self.evaluate(rank))


@test_util.test_all_tf_execution_regimes
class PreferStaticShapeTest(test_util.TestCase):

  def testNonEmptyConstantTensor(self):
    x = tf.zeros((2, 3, 4))
    shape = distribution_util.prefer_static_shape(x)
    self.assertIsInstance(shape, np.ndarray)
    self.assertAllEqual([2, 3, 4], shape)

  def testEmptyConstantTensor(self):
    x = tf.constant([])
    shape = distribution_util.prefer_static_shape(x)
    self.assertIsInstance(shape, np.ndarray)
    self.assertAllEqual([0], shape)

  def testScalarTensor(self):
    x = tf.constant(1.)
    shape = distribution_util.prefer_static_shape(x)
    self.assertIsInstance(shape, np.ndarray)
    self.assertAllEqual([], shape)

  def testDynamicShapeEndsUpBeingNonEmpty(self):
    if tf.executing_eagerly(): return
    x = tf1.placeholder_with_default(
        np.zeros([2, 3], dtype=np.float64), shape=None)
    shape = distribution_util.prefer_static_shape(x)
    self.assertAllEqual([2, 3], self.evaluate(shape))

  def testDynamicShapeEndsUpBeingEmpty(self):
    if tf.executing_eagerly(): return
    x = tf1.placeholder_with_default(
        np.array([], dtype=np.int32), shape=None)
    shape = distribution_util.prefer_static_shape(x)
    self.assertAllEqual([0], self.evaluate(shape))

  def testDynamicShapeEndsUpBeingScalar(self):
    if tf.executing_eagerly(): return
    x = tf1.placeholder_with_default(
        np.array(1, dtype=np.int32), shape=None)
    shape = distribution_util.prefer_static_shape(x)
    self.assertAllEqual([], self.evaluate(shape))


@test_util.test_all_tf_execution_regimes
class PreferStaticValueTest(test_util.TestCase):

  def testNonEmptyConstantTensor(self):
    x = tf.zeros((2, 3, 4))
    value = distribution_util.prefer_static_value(x)
    self.assertIsInstance(value, np.ndarray)
    self.assertAllEqual(np.zeros((2, 3, 4)), value)

  def testEmptyConstantTensor(self):
    x = tf.constant([])
    value = distribution_util.prefer_static_value(x)
    self.assertIsInstance(value, np.ndarray)
    self.assertAllEqual(np.array([]), value)

  def testScalarTensor(self):
    x = tf.constant(1.)
    value = distribution_util.prefer_static_value(x)
    if not tf.executing_eagerly():
      self.assertIsInstance(value, np.ndarray)
    self.assertAllEqual(np.array(1.), value)

  def testDynamicValueEndsUpBeingNonEmpty(self):
    if tf.executing_eagerly(): return
    x = tf1.placeholder_with_default(
        np.zeros((2, 3), dtype=np.float64), shape=None)
    value = distribution_util.prefer_static_value(x)
    self.assertAllEqual(np.zeros((2, 3)),
                        self.evaluate(value))

  def testDynamicValueEndsUpBeingEmpty(self):
    if tf.executing_eagerly(): return
    x = tf1.placeholder_with_default(
        np.array([], dtype=np.int32), shape=None)
    value = distribution_util.prefer_static_value(x)
    self.assertAllEqual(np.array([]), self.evaluate(value))

  def testDynamicValueEndsUpBeingScalar(self):
    if tf.executing_eagerly(): return
    x = tf1.placeholder_with_default(
        np.array(1, dtype=np.int32), shape=None)
    value = distribution_util.prefer_static_value(x)
    self.assertAllEqual(np.array(1), self.evaluate(value))


# No need for eager tests; this function doesn't depend on TF.
class GenNewSeedTest(test_util.TestCase):

  def testOnlyNoneReturnsNone(self):
    self.assertIsNotNone(distribution_util.gen_new_seed(0, 'salt'))
    self.assertIsNone(distribution_util.gen_new_seed(None, 'salt'))


@test_util.test_all_tf_execution_regimes
class ArgumentsTest(test_util.TestCase):

  def testNoArguments(self):
    def foo():
      return distribution_util.parent_frame_arguments()

    self.assertEqual({}, foo())

  def testPositionalArguments(self):
    def foo(a, b, c, d):  # pylint: disable=unused-argument
      return distribution_util.parent_frame_arguments()

    self.assertEqual({'a': 1, 'b': 2, 'c': 3, 'd': 4}, foo(1, 2, 3, 4))

    # Tests that it does not matter where this function is called, and
    # no other local variables are returned back.
    def bar(a, b, c):
      unused_x = a * b
      unused_y = c * 3
      return distribution_util.parent_frame_arguments()

    self.assertEqual({'a': 1, 'b': 2, 'c': 3}, bar(1, 2, 3))

  def testOverloadedArgumentValues(self):
    def foo(a, b, c):  # pylint: disable=unused-argument
      a = 42
      b = 31
      c = 42
      return distribution_util.parent_frame_arguments()
    self.assertEqual({'a': 42, 'b': 31, 'c': 42}, foo(1, 2, 3))

  def testKeywordArguments(self):
    def foo(**kwargs):  # pylint: disable=unused-argument
      return distribution_util.parent_frame_arguments()

    self.assertEqual({'a': 1, 'b': 2, 'c': 3, 'd': 4}, foo(a=1, b=2, c=3, d=4))

  def testPositionalKeywordArgs(self):
    def foo(a, b, c, **kwargs):  # pylint: disable=unused-argument
      return distribution_util.parent_frame_arguments()

    self.assertEqual({'a': 1, 'b': 2, 'c': 3}, foo(a=1, b=2, c=3))
    self.assertEqual({'a': 1, 'b': 2, 'c': 3, 'unicorn': None},
                     foo(a=1, b=2, c=3, unicorn=None))

  def testNoVarargs(self):
    def foo(a, b, c, *varargs, **kwargs):  # pylint: disable=unused-argument
      return distribution_util.parent_frame_arguments()

    self.assertEqual({'a': 1, 'b': 2, 'c': 3}, foo(a=1, b=2, c=3))
    self.assertEqual({'a': 1, 'b': 2, 'c': 3}, foo(1, 2, 3, *[1, 2, 3]))
    self.assertEqual({'a': 1, 'b': 2, 'c': 3, 'unicorn': None},
                     foo(1, 2, 3, unicorn=None))
    self.assertEqual({'a': 1, 'b': 2, 'c': 3, 'unicorn': None},
                     foo(1, 2, 3, *[1, 2, 3], unicorn=None))


@test_util.test_all_tf_execution_regimes
class ExpandToVectorTest(test_util.TestCase):

  def _check_static(self, expected, actual, dtype=np.int32):
    const_actual = tf.get_static_value(actual)
    self.assertAllEqual(expected, const_actual)
    self.assertEqual(dtype, const_actual.dtype)

  def _check(self, expected, actual, expected_dtype=np.int32):
    self.assertAllEqual(expected, actual)
    self.assertEquals(expected_dtype, actual.dtype)

  def test_expand_to_vector_on_literals(self):
    self._check_static([1], distribution_util.expand_to_vector(1))
    self._check_static(
        [3.5], distribution_util.expand_to_vector(3.5), dtype=np.float32)

    self._check_static([3], distribution_util.expand_to_vector((3,)))
    self._check_static([0, 0], distribution_util.expand_to_vector((0, 0)))
    self._check_static(
        [1.25, 2.75, 3.0],
        distribution_util.expand_to_vector((1.25, 2.75, 3.0)),
        dtype=np.float32)

    self._check_static([3], distribution_util.expand_to_vector([3,]))
    self._check_static([0, 0], distribution_util.expand_to_vector([0, 0]))
    self._check_static(
        [1.25, 2.75, 3.0],
        distribution_util.expand_to_vector([1.25, 2.75, 3.0]),
        dtype=np.float32)

    # Empty lists and tuples are converted to `tf.float32`.
    self._check_static(
        [], distribution_util.expand_to_vector(()), dtype=np.float32)
    self._check_static(
        [], distribution_util.expand_to_vector([]), dtype=np.float32)

    # Test for error on input with rank >= 2.
    with self.assertRaises(ValueError):
      distribution_util.expand_to_vector([[1, 2], [3, 4]])

  def test_expand_to_vector_on_constants(self):
    # Helper to construct a const Tensor and call expand_to_tensor on it.
    def _expand_tensor(x, dtype=tf.int32):
      return distribution_util.expand_to_vector(
          tf.convert_to_tensor(value=x, dtype=dtype), op_name='test')

    self._check_static([], _expand_tensor([]))
    self._check_static([], _expand_tensor(()))

    self._check_static([17], _expand_tensor(17))
    self._check_static([1.125], _expand_tensor(1.125, np.float32), np.float32)

    self._check_static([314], _expand_tensor([314]))
    self._check_static(
        [3.75, 0], _expand_tensor([3.75, 0], np.float64), np.float64)
    self._check_static([1, 2, 3], _expand_tensor([1, 2, 3], np.int64), np.int64)

    # Test for error on input with rank >= 2.
    with self.assertRaises(ValueError):
      _expand_tensor([[[]]], tf.float32)

  def test_expand_to_vector_on_tensors(self):
    # Helper to construct a placeholder and call expand_to_tensor on it.
    def _expand_tensor(x, shape=None, dtype=np.int32, validate_args=False):
      return distribution_util.expand_to_vector(
          tf1.placeholder_with_default(
              np.array(x, dtype=dtype), shape=shape),
          tensor_name='name_for_tensor',
          validate_args=validate_args)

    for dtype in [np.int64, np.float32, np.float64, np.int64]:

      self._check([], _expand_tensor([], shape=[0], dtype=dtype), dtype)
      self._check([], _expand_tensor([], shape=[None], dtype=dtype), dtype)
      self._check([], _expand_tensor([], shape=None, dtype=dtype), dtype)

      self._check([7], _expand_tensor(7, shape=[], dtype=dtype), dtype)

      self._check(
          [1, 2, 3], _expand_tensor([1, 2, 3], shape=[3], dtype=dtype), dtype)
      self._check(
          [1, 2, 3],
          _expand_tensor([1, 2, 3], shape=[None], dtype=dtype), dtype)
      self._check(
          [1, 2, 3], _expand_tensor([1, 2, 3], shape=None, dtype=dtype), dtype)

    # Test for error on input with rank >= 2.
    with self.assertRaises(ValueError):
      _expand_tensor([[1, 2]], shape=[1, 2])
    with self.assertRaises(ValueError):
      _expand_tensor([[1, 2]], shape=[None, None])
    if tf.executing_eagerly():
      with self.assertRaises(ValueError):
        _expand_tensor([[1, 2]], shape=None)
    else:
      with self.assertRaises(tf.errors.InvalidArgumentError):
        self.evaluate(_expand_tensor([[1, 2]], shape=None, validate_args=True))


@test_util.test_all_tf_execution_regimes
class WithDependenciesTestCase(test_util.TestCase):

  def testTupleDependencies(self):
    counter = tf.Variable(0, name='my_counter')
    const_with_dep = distribution_util.with_dependencies(
        (counter.assign_add(1), tf.constant(42)),
        tf.constant(7))

    self.evaluate(tf1.global_variables_initializer())
    self.assertEqual(1 if tf.executing_eagerly() else 0,
                     self.evaluate(counter))
    self.assertEqual(7, self.evaluate(const_with_dep))
    self.assertEqual(1, self.evaluate(counter))

  def testListDependencies(self):
    counter = tf.Variable(0, name='my_counter')
    const_with_dep = distribution_util.with_dependencies(
        [counter.assign_add(1), tf.constant(42)],
        tf.constant(7))

    self.evaluate(tf1.global_variables_initializer())
    self.assertEqual(1 if tf.executing_eagerly() else 0,
                     self.evaluate(counter))
    self.assertEqual(7, self.evaluate(const_with_dep))
    self.assertEqual(1, self.evaluate(counter))


if __name__ == '__main__':
  tf.test.main()
