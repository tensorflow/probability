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

import importlib
import itertools

# Dependency imports
import numpy as np
import tensorflow as tf

from tensorflow_probability.python.distributions import Categorical
from tensorflow_probability.python.distributions import Mixture
from tensorflow_probability.python.distributions import MixtureSameFamily
from tensorflow_probability.python.distributions import MultivariateNormalDiag
from tensorflow_probability.python.distributions import Normal

from tensorflow_probability.python.internal import distribution_util
tfe = tf.contrib.eager


def try_import(name):  # pylint: disable=invalid-name
  module = None
  try:
    module = importlib.import_module(name)
  except ImportError as e:
    tf.logging.warning("Could not import %s: %s" % (name, str(e)))
  return module


special = try_import("scipy.special")


def _logit(x):
  x = np.asarray(x)
  return np.log(x) - np.log1p(-x)


def _powerset(x):
  s = list(x)
  return itertools.chain.from_iterable(
      itertools.combinations(s, r) for r in range(len(s) + 1))


def _matrix_diag(d):
  """Batch version of np.diag."""
  orig_shape = d.shape
  d = np.reshape(d, (int(np.prod(d.shape[:-1])), d.shape[-1]))
  diag_list = []
  for i in range(d.shape[0]):
    diag_list.append(np.diag(d[i, ...]))
  return np.reshape(diag_list, orig_shape + (d.shape[-1],))


def _make_tril_scale(
    loc=None,
    scale_tril=None,
    scale_diag=None,
    scale_identity_multiplier=None,
    shape_hint=None):
  if scale_tril is not None:
    scale_tril = np.tril(scale_tril)
    if scale_diag is not None:
      scale_tril += _matrix_diag(np.array(scale_diag, dtype=np.float32))
    if scale_identity_multiplier is not None:
      scale_tril += (
          scale_identity_multiplier * _matrix_diag(np.ones(
              [scale_tril.shape[-1]], dtype=np.float32)))
    return scale_tril
  return _make_diag_scale(
      loc, scale_diag, scale_identity_multiplier, shape_hint)


def _make_diag_scale(
    loc=None,
    scale_diag=None,
    scale_identity_multiplier=None,
    shape_hint=None):
  if scale_diag is not None:
    scale_diag = np.asarray(scale_diag)
    if scale_identity_multiplier is not None:
      scale_diag += scale_identity_multiplier
    return _matrix_diag(scale_diag)

  if loc is None and shape_hint is None:
    return None

  if shape_hint is None:
    shape_hint = loc.shape[-1]
  if scale_identity_multiplier is None:
    scale_identity_multiplier = 1.
  return scale_identity_multiplier * np.diag(np.ones(shape_hint))


class MakeTrilScaleTest(tf.test.TestCase):

  def _testLegalInputs(
      self, loc=None, shape_hint=None, scale_params=None):
    for args in _powerset(scale_params.items()):
      with self.cached_session():
        args = dict(args)

        scale_args = dict({
            "loc": loc,
            "shape_hint": shape_hint}, **args)
        expected_scale = _make_tril_scale(**scale_args)
        if expected_scale is None:
          # Not enough shape information was specified.
          with self.assertRaisesRegexp(ValueError, ("is specified.")):
            scale = distribution_util.make_tril_scale(**scale_args)
            self.evaluate(scale.to_dense())
        else:
          scale = distribution_util.make_tril_scale(**scale_args)
          self.assertAllClose(expected_scale, self.evaluate(scale.to_dense()))

  def testLegalInputs(self):
    self._testLegalInputs(
        loc=np.array([-1., -1.], dtype=np.float32),
        shape_hint=2,
        scale_params={
            "scale_identity_multiplier": 2.,
            "scale_diag": [2., 3.],
            "scale_tril": [[1., 0.],
                           [-3., 3.]],
        })

  def testLegalInputsMultidimensional(self):
    self._testLegalInputs(
        loc=np.array([[[-1., -1., 2.], [-2., -3., 4.]]], dtype=np.float32),
        shape_hint=3,
        scale_params={
            "scale_identity_multiplier": 2.,
            "scale_diag": [[[2., 3., 4.], [3., 4., 5.]]],
            "scale_tril": [[[[1., 0., 0.],
                             [-3., 3., 0.],
                             [1., -2., 1.]],
                            [[2., 1., 0.],
                             [-4., 7., 0.],
                             [1., -1., 1.]]]]
        })

  def testZeroTriU(self):
    with self.cached_session():
      scale = distribution_util.make_tril_scale(scale_tril=[[1., 1], [1., 1.]])
      self.assertAllClose([[1., 0], [1., 1.]], self.evaluate(scale.to_dense()))

  def testValidateArgs(self):
    with self.cached_session():
      with self.assertRaisesOpError("diagonal part must be non-zero"):
        scale = distribution_util.make_tril_scale(
            scale_tril=[[0., 1], [1., 1.]], validate_args=True)
        self.evaluate(scale.to_dense())

  def testAssertPositive(self):
    with self.cached_session():
      with self.assertRaisesOpError("diagonal part must be positive"):
        scale = distribution_util.make_tril_scale(
            scale_tril=[[-1., 1], [1., 1.]],
            validate_args=True,
            assert_positive=True)
        self.evaluate(scale.to_dense())


class MakeDiagScaleTest(tf.test.TestCase):

  def _testLegalInputs(
      self, loc=None, shape_hint=None, scale_params=None):
    for args in _powerset(scale_params.items()):
      with self.cached_session():
        args = dict(args)

        scale_args = dict({
            "loc": loc,
            "shape_hint": shape_hint}, **args)
        expected_scale = _make_diag_scale(**scale_args)
        if expected_scale is None:
          # Not enough shape information was specified.
          with self.assertRaisesRegexp(ValueError, ("is specified.")):
            scale = distribution_util.make_diag_scale(**scale_args)
            self.evaluate(scale.to_dense())
        else:
          scale = distribution_util.make_diag_scale(**scale_args)
          self.assertAllClose(expected_scale, self.evaluate(scale.to_dense()))

  def testLegalInputs(self):
    self._testLegalInputs(
        loc=np.array([-1., -1.], dtype=np.float32),
        shape_hint=2,
        scale_params={
            "scale_identity_multiplier": 2.,
            "scale_diag": [2., 3.]
        })

  def testLegalInputsMultidimensional(self):
    self._testLegalInputs(
        loc=np.array([[[-1., -1., 2.], [-2., -3., 4.]]], dtype=np.float32),
        shape_hint=3,
        scale_params={
            "scale_identity_multiplier": 2.,
            "scale_diag": [[[2., 3., 4.], [3., 4., 5.]]]
        })

  def testValidateArgs(self):
    with self.cached_session():
      with self.assertRaisesOpError("diagonal part must be non-zero"):
        scale = distribution_util.make_diag_scale(
            scale_diag=[[0., 1], [1., 1.]], validate_args=True)
        self.evaluate(scale.to_dense())

  def testAssertPositive(self):
    with self.cached_session():
      with self.assertRaisesOpError("diagonal part must be positive"):
        scale = distribution_util.make_diag_scale(
            scale_diag=[[-1., 1], [1., 1.]],
            validate_args=True,
            assert_positive=True)
        self.evaluate(scale.to_dense())


class ShapesFromLocAndScaleTest(tf.test.TestCase):

  def test_static_loc_static_scale_non_matching_event_size_raises(self):
    loc = tf.constant(np.zeros((2, 4)))
    scale = tf.linalg.LinearOperatorDiag(np.ones((5, 1, 3)))
    with self.assertRaisesRegexp(ValueError, "could not be broadcast"):
      distribution_util.shapes_from_loc_and_scale(loc, scale)

  def test_static_loc_static_scale(self):
    loc = tf.constant(np.zeros((2, 3)))
    scale = tf.linalg.LinearOperatorDiag(np.ones((5, 1, 3)))
    batch_shape, event_shape = distribution_util.shapes_from_loc_and_scale(
        loc, scale)

    self.assertEqual(tf.TensorShape([5, 2]), batch_shape)
    self.assertEqual(tf.TensorShape([3]), event_shape)

  def test_static_loc_dynamic_scale(self):
    loc = tf.constant(np.zeros((2, 3)))
    diag = tf.placeholder(tf.float64)
    scale = tf.linalg.LinearOperatorDiag(diag)
    with self.cached_session() as sess:
      batch_shape, event_shape = sess.run(
          distribution_util.shapes_from_loc_and_scale(loc, scale),
          feed_dict={diag: np.ones((5, 1, 3))})
      self.assertAllEqual([5, 2], batch_shape)
      self.assertAllEqual([3], event_shape)

  def test_dynamic_loc_static_scale(self):
    loc = tf.placeholder(tf.float64)
    diag = tf.constant(np.ones((5, 2, 3)))
    scale = tf.linalg.LinearOperatorDiag(diag)
    with self.cached_session():
      batch_shape, event_shape = distribution_util.shapes_from_loc_and_scale(
          loc, scale)
      # batch_shape depends on both args, and so is dynamic.  Since loc did not
      # have static shape, we inferred event shape entirely from scale, and this
      # is available statically.
      self.assertAllEqual(
          [5, 2], batch_shape.eval(feed_dict={loc: np.zeros((2, 3))}))
      self.assertAllEqual([3], event_shape)

  def test_dynamic_loc_dynamic_scale(self):
    loc = tf.placeholder(tf.float64)
    diag = tf.placeholder(tf.float64)
    scale = tf.linalg.LinearOperatorDiag(diag)
    with self.cached_session() as sess:
      batch_shape, event_shape = sess.run(
          distribution_util.shapes_from_loc_and_scale(loc, scale),
          feed_dict={diag: np.ones((5, 2, 3)), loc: np.zeros((2, 3))})
      self.assertAllEqual([5, 2], batch_shape)
      self.assertAllEqual([3], event_shape)

  def test_none_loc_static_scale(self):
    loc = None
    scale = tf.linalg.LinearOperatorDiag(np.ones((5, 1, 3)))
    batch_shape, event_shape = distribution_util.shapes_from_loc_and_scale(
        loc, scale)

    self.assertEqual(tf.TensorShape([5, 1]), batch_shape)
    self.assertEqual(tf.TensorShape([3]), event_shape)

  def test_none_loc_dynamic_scale(self):
    loc = None
    diag = tf.placeholder(tf.float64)
    scale = tf.linalg.LinearOperatorDiag(diag)
    with self.cached_session() as sess:
      batch_shape, event_shape = sess.run(
          distribution_util.shapes_from_loc_and_scale(loc, scale),
          feed_dict={diag: np.ones((5, 1, 3))})
      self.assertAllEqual([5, 1], batch_shape)
      self.assertAllEqual([3], event_shape)


class GetBroadcastShapeTest(tf.test.TestCase):

  def test_all_static_shapes_work(self):
    x = tf.ones((2, 1, 3))
    y = tf.ones((1, 5, 3))
    z = tf.ones(())
    self.assertAllEqual([2, 5, 3],
                        distribution_util.get_broadcast_shape(x, y, z))

  def test_with_some_dynamic_shapes_works(self):
    x = tf.ones((2, 1, 3))
    y = tf.placeholder(x.dtype)
    z = tf.ones(())
    with self.cached_session() as sess:
      bcast_shape = sess.run(
          distribution_util.get_broadcast_shape(x, y, z),
          feed_dict={y: np.ones((1, 5, 3)).astype(np.float32)})
      self.assertAllEqual([2, 5, 3], bcast_shape)


class TridiagTest(tf.test.TestCase):

  def testWorksCorrectlyNoBatches(self):
    with self.cached_session():
      self.assertAllEqual(
          [[4., 8., 0., 0.],
           [1., 5., 9., 0.],
           [0., 2., 6., 10.],
           [0., 0., 3, 7.]],
          self.evaluate(distribution_util.tridiag(
              [1., 2., 3.],
              [4., 5., 6., 7.],
              [8., 9., 10.])))

  def testWorksCorrectlyBatches(self):
    with self.cached_session():
      self.assertAllClose(
          [[[4., 8., 0., 0.],
            [1., 5., 9., 0.],
            [0., 2., 6., 10.],
            [0., 0., 3, 7.]],
           [[0.7, 0.1, 0.0, 0.0],
            [0.8, 0.6, 0.2, 0.0],
            [0.0, 0.9, 0.5, 0.3],
            [0.0, 0.0, 1.0, 0.4]]],
          self.evaluate(distribution_util.tridiag(
              [[1., 2., 3.],
               [0.8, 0.9, 1.]],
              [[4., 5., 6., 7.],
               [0.7, 0.6, 0.5, 0.4]],
              [[8., 9., 10.],
               [0.1, 0.2, 0.3]])),
          rtol=1e-5, atol=0.)

  def testHandlesNone(self):
    with self.cached_session():
      self.assertAllClose(
          [[[4., 0., 0., 0.],
            [0., 5., 0., 0.],
            [0., 0., 6., 0.],
            [0., 0., 0, 7.]],
           [[0.7, 0.0, 0.0, 0.0],
            [0.0, 0.6, 0.0, 0.0],
            [0.0, 0.0, 0.5, 0.0],
            [0.0, 0.0, 0.0, 0.4]]],
          self.evaluate(distribution_util.tridiag(
              diag=[[4., 5., 6., 7.],
                    [0.7, 0.6, 0.5, 0.4]])),
          rtol=1e-5, atol=0.)


class MixtureStddevTest(tf.test.TestCase):

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

    with self.cached_session() as sess:
      actual_devs = sess.run(mix_dev)

    self.assertAllClose(actual_devs, expected_devs)


class PadMixtureDimensionsTest(tf.test.TestCase):

  def test_pad_mixture_dimensions_mixture(self):
    with self.cached_session() as sess:
      gm = Mixture(
          cat=Categorical(probs=[[0.3, 0.7]]),
          components=[
              Normal(loc=[-1.0], scale=[1.0]),
              Normal(loc=[1.0], scale=[0.5])
          ])

      x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
      x_pad = distribution_util.pad_mixture_dimensions(
          x, gm, gm.cat, gm.event_shape.ndims)
      x_out, x_pad_out = sess.run([x, x_pad])

    self.assertAllEqual(x_pad_out.shape, [2, 2])
    self.assertAllEqual(x_out.reshape([-1]), x_pad_out.reshape([-1]))

  def test_pad_mixture_dimensions_mixture_same_family(self):
    with self.cached_session() as sess:
      gm = MixtureSameFamily(
          mixture_distribution=Categorical(probs=[0.3, 0.7]),
          components_distribution=MultivariateNormalDiag(
              loc=[[-1., 1], [1, -1]], scale_identity_multiplier=[1.0, 0.5]))

      x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
      x_pad = distribution_util.pad_mixture_dimensions(
          x, gm, gm.mixture_distribution, gm.event_shape.ndims)
      x_out, x_pad_out = sess.run([x, x_pad])

    self.assertAllEqual(x_pad_out.shape, [2, 2, 1])
    self.assertAllEqual(x_out.reshape([-1]), x_pad_out.reshape([-1]))


class _PadTest(object):

  def testNegAxisCorrectness(self):
    x_ = np.float32([[1., 2, 3],
                     [4, 5, 6]])
    value_ = np.float32(0.25)
    count_ = np.int32(2)
    with self.cached_session() as sess:
      x = tf.placeholder_with_default(
          x_, shape=x_.shape if self.is_static_shape else None)
      value = (
          tf.constant(value_)
          if self.is_static_shape else tf.placeholder_with_default(
              value_, shape=None))
      count = (
          tf.constant(count_)
          if self.is_static_shape else tf.placeholder_with_default(
              count_, shape=None))

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

      [x0_front_, x0_back_, x0_both_] = sess.run([
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
    with self.cached_session() as sess:
      x = tf.placeholder_with_default(
          x_, shape=x_.shape if self.is_static_shape else None)
      value = (
          tf.constant(value_)
          if self.is_static_shape else tf.placeholder_with_default(
              value_, shape=None))
      count = (
          tf.constant(count_)
          if self.is_static_shape else tf.placeholder_with_default(
              count_, shape=None))

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

      [x1_front_, x1_back_, x1_both_] = sess.run([
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


class PadStaticTest(_PadTest, tf.test.TestCase):

  @property
  def is_static_shape(self):
    return True


class PadDynamicTest(_PadTest, tf.test.TestCase):

  @property
  def is_static_shape(self):
    return False


class PickScalarConditionTest(tf.test.TestCase):

  @tfe.run_test_in_graph_and_eager_modes
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
    dynamic_true = tf.placeholder_with_default(input=True, shape=None)
    dynamic_false = tf.placeholder_with_default(input=False, shape=None)
    pos_ = self.evaluate(distribution_util.pick_scalar_condition(
        dynamic_true, pos, neg))
    neg_ = self.evaluate(distribution_util.pick_scalar_condition(
        dynamic_false, pos, neg))
    self.assertAllEqual(pos_, pos)
    self.assertAllEqual(neg_, neg)

    # TF dynamic everything
    pos_dynamic = tf.placeholder_with_default(input=pos, shape=None)
    neg_dynamic = tf.placeholder_with_default(input=neg, shape=None)
    pos_ = self.evaluate(distribution_util.pick_scalar_condition(
        dynamic_true, pos_dynamic, neg_dynamic))
    neg_ = self.evaluate(distribution_util.pick_scalar_condition(
        dynamic_false, pos_dynamic, neg_dynamic))
    self.assertAllEqual(pos_, pos)
    self.assertAllEqual(neg_, neg)


@tfe.run_all_tests_in_graph_and_eager_modes
class TestNonNegativeAxis(tf.test.TestCase):

  def test_static_scalar_positive_index(self):
    positive_axis = distribution_util.make_non_negative_axis(axis=2, rank=4)
    self.assertAllEqual(2, self.evaluate(positive_axis))

  def test_static_scalar_negative_index(self):
    positive_axis = distribution_util.make_non_negative_axis(axis=-1, rank=4)
    self.assertAllEqual(3, self.evaluate(positive_axis))

  def test_static_vector_index(self):
    positive_axis = distribution_util.make_non_negative_axis(
        axis=[0, -2], rank=4)
    self.assertAllEqual([0, 2], self.evaluate(positive_axis))

  def test_dynamic_vector_index(self):
    positive_axis = distribution_util.make_non_negative_axis(
        axis=tf.placeholder_with_default(input=[0, -2], shape=None),
        rank=4)
    self.assertAllEqual([0, 2], self.evaluate(positive_axis))


class TestMoveDimension(tf.test.TestCase):

  @tfe.run_test_in_graph_and_eager_modes
  def test_move_dimension_static_shape(self):

    x = tf.random_normal(shape=[200, 30, 4, 1, 6])

    x_perm = distribution_util.move_dimension(x, 1, 1)
    self.assertAllEqual(x_perm.shape.as_list(), [200, 30, 4, 1, 6])

    x_perm = distribution_util.move_dimension(x, 0, 3)
    self.assertAllEqual(x_perm.shape.as_list(), [30, 4, 1, 200, 6])

    x_perm = distribution_util.move_dimension(x, 0, -2)
    self.assertAllEqual(x_perm.shape.as_list(), [30, 4, 1, 200, 6])

    x_perm = distribution_util.move_dimension(x, 4, 2)
    self.assertAllEqual(x_perm.shape.as_list(), [200, 30, 6, 4, 1])

  @tfe.run_test_in_graph_and_eager_modes
  def test_move_dimension_dynamic_shape(self):

    x_ = tf.random_normal(shape=[200, 30, 4, 1, 6])
    x = tf.placeholder_with_default(input=x_, shape=None)

    x_perm1 = distribution_util.move_dimension(x, 1, 1)
    x_perm2 = distribution_util.move_dimension(x, 0, 3)
    x_perm3 = distribution_util.move_dimension(x, 0, -2)
    x_perm4 = distribution_util.move_dimension(x, 4, 2)
    x_perm5 = distribution_util.move_dimension(x, -1, 2)

    x_perm1_, x_perm2_, x_perm3_, x_perm4_, x_perm5_ = self.evaluate(
        [tf.shape(x_perm1),
         tf.shape(x_perm2),
         tf.shape(x_perm3),
         tf.shape(x_perm4),
         tf.shape(x_perm5)])

    self.assertAllEqual(x_perm1_, [200, 30, 4, 1, 6])

    self.assertAllEqual(x_perm2_, [30, 4, 1, 200, 6])

    self.assertAllEqual(x_perm3_, [30, 4, 1, 200, 6])

    self.assertAllEqual(x_perm4_, [200, 30, 6, 4, 1])

    self.assertAllEqual(x_perm5_, [200, 30, 6, 4, 1])

  @tfe.run_test_in_graph_and_eager_modes
  def test_move_dimension_dynamic_indices(self):

    x_ = tf.random_normal(shape=[200, 30, 4, 1, 6])
    x = tf.placeholder_with_default(input=x_, shape=None)

    x_perm1 = distribution_util.move_dimension(
        x,
        tf.placeholder_with_default(input=1, shape=[]),
        tf.placeholder_with_default(input=1, shape=[]))

    x_perm2 = distribution_util.move_dimension(
        x,
        tf.placeholder_with_default(input=0, shape=[]),
        tf.placeholder_with_default(input=3, shape=[]))

    x_perm3 = distribution_util.move_dimension(
        x,
        tf.placeholder_with_default(input=0, shape=[]),
        tf.placeholder_with_default(input=-2, shape=[]))

    x_perm4 = distribution_util.move_dimension(
        x,
        tf.placeholder_with_default(input=4, shape=[]),
        tf.placeholder_with_default(input=2, shape=[]))

    x_perm5 = distribution_util.move_dimension(
        x,
        tf.placeholder_with_default(input=-1, shape=[]),
        tf.placeholder_with_default(input=2, shape=[]))

    x_perm1_, x_perm2_, x_perm3_, x_perm4_, x_perm5_ = self.evaluate(
        [tf.shape(x_perm1),
         tf.shape(x_perm2),
         tf.shape(x_perm3),
         tf.shape(x_perm4),
         tf.shape(x_perm5)])

    self.assertAllEqual(x_perm1_, [200, 30, 4, 1, 6])

    self.assertAllEqual(x_perm2_, [30, 4, 1, 200, 6])

    self.assertAllEqual(x_perm3_, [30, 4, 1, 200, 6])

    self.assertAllEqual(x_perm4_, [200, 30, 6, 4, 1])

    self.assertAllEqual(x_perm5_, [200, 30, 6, 4, 1])


class AssertCloseTest(tf.test.TestCase):

  def testAssertIntegerForm(self):
    # This should only be detected as an integer.
    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)
    # First component isn't less than float32.eps = 1e-7
    z = tf.placeholder(tf.float32)
    # This shouldn"t be detected as an integer.
    w = tf.placeholder(tf.float32)
    feed_dict = {x: [1., 5, 10, 15, 20], y: [1.1, 5, 10, 15, 20],
                 z: [1.0001, 5, 10, 15, 20], w: [1e-8, 5, 10, 15, 20]}
    with self.cached_session():
      with tf.control_dependencies([distribution_util.assert_integer_form(x)]):
        tf.identity(x).eval(feed_dict=feed_dict)

      with self.assertRaisesOpError("has non-integer components"):
        with tf.control_dependencies(
            [distribution_util.assert_integer_form(y)]):
          tf.identity(y).eval(feed_dict=feed_dict)

      with self.assertRaisesOpError("has non-integer components"):
        with tf.control_dependencies(
            [distribution_util.assert_integer_form(z)]):
          tf.identity(z).eval(feed_dict=feed_dict)

      with self.assertRaisesOpError("has non-integer components"):
        with tf.control_dependencies(
            [distribution_util.assert_integer_form(w)]):
          tf.identity(w).eval(feed_dict=feed_dict)


class MaybeGetStaticTest(tf.test.TestCase):

  @tfe.run_test_in_graph_and_eager_modes
  def testGetStaticInt(self):
    x = 2
    self.assertEqual(x, distribution_util.maybe_get_static_value(x))
    self.assertAllClose(
        np.array(2.),
        distribution_util.maybe_get_static_value(x, dtype=np.float64))

  @tfe.run_test_in_graph_and_eager_modes
  def testGetStaticNumpyArray(self):
    x = np.array(2, dtype=np.int32)
    self.assertEqual(x, distribution_util.maybe_get_static_value(x))
    self.assertAllClose(
        np.array(2.),
        distribution_util.maybe_get_static_value(x, dtype=np.float64))

  @tfe.run_test_in_graph_and_eager_modes
  def testGetStaticConstant(self):
    x = tf.constant(2, dtype=tf.int32)
    self.assertEqual(np.array(2, dtype=np.int32),
                     distribution_util.maybe_get_static_value(x))
    self.assertAllClose(
        np.array(2.),
        distribution_util.maybe_get_static_value(x, dtype=np.float64))

  def testGetStaticPlaceholder(self):
    x = tf.placeholder(dtype=tf.int32, shape=[1])
    self.assertEqual(None, distribution_util.maybe_get_static_value(x))
    self.assertEqual(
        None, distribution_util.maybe_get_static_value(x, dtype=np.float64))


class GetLogitsAndProbsTest(tf.test.TestCase):

  @tfe.run_test_in_graph_and_eager_modes
  def testImproperArguments(self):
    with self.assertRaises(ValueError):
      distribution_util.get_logits_and_probs(logits=None, probs=None)

    with self.assertRaises(ValueError):
      distribution_util.get_logits_and_probs(logits=[0.1], probs=[0.1])

  @tfe.run_test_in_graph_and_eager_modes
  def testLogits(self):
    p = np.array([0.01, 0.2, 0.5, 0.7, .99], dtype=np.float32)
    logits = _logit(p)

    new_logits, new_p = distribution_util.get_logits_and_probs(
        logits=logits, validate_args=True)

    self.assertAllClose(p, self.evaluate(new_p), rtol=1e-5, atol=0.)
    self.assertAllClose(logits, self.evaluate(new_logits), rtol=1e-5, atol=0.)

  @tfe.run_test_in_graph_and_eager_modes
  def testLogitsMultidimensional(self):
    p = np.array([0.2, 0.3, 0.5], dtype=np.float32)
    logits = np.log(p)

    new_logits, new_p = distribution_util.get_logits_and_probs(
        logits=logits, multidimensional=True, validate_args=True)

    self.assertAllClose(self.evaluate(new_p), p)
    self.assertAllClose(self.evaluate(new_logits), logits)

  @tfe.run_test_in_graph_and_eager_modes
  def testProbability(self):
    p = np.array([0.01, 0.2, 0.5, 0.7, .99], dtype=np.float32)

    new_logits, new_p = distribution_util.get_logits_and_probs(
        probs=p, validate_args=True)

    self.assertAllClose(_logit(p), self.evaluate(new_logits))
    self.assertAllClose(p, self.evaluate(new_p))

  @tfe.run_test_in_graph_and_eager_modes
  def testProbabilityMultidimensional(self):
    p = np.array([[0.3, 0.4, 0.3], [0.1, 0.5, 0.4]], dtype=np.float32)

    new_logits, new_p = distribution_util.get_logits_and_probs(
        probs=p, multidimensional=True, validate_args=True)

    self.assertAllClose(np.log(p), self.evaluate(new_logits))
    self.assertAllClose(p, self.evaluate(new_p))

  @tfe.run_test_in_graph_and_eager_modes
  def testProbabilityValidateArgs(self):
    p = [0.01, 0.2, 0.5, 0.7, .99]
    # Component less than 0.
    p2 = [-1, 0.2, 0.5, 0.3, .2]
    # Component greater than 1.
    p3 = [2, 0.2, 0.5, 0.3, .2]

    _, prob = distribution_util.get_logits_and_probs(
        probs=p, validate_args=True)
    self.evaluate(prob)

    with self.assertRaisesOpError("Condition x >= 0"):
      _, prob = distribution_util.get_logits_and_probs(
          probs=p2, validate_args=True)
      self.evaluate(prob)

    _, prob = distribution_util.get_logits_and_probs(
        probs=p2, validate_args=False)
    self.evaluate(prob)

    with self.assertRaisesOpError("probs has components greater than 1"):
      _, prob = distribution_util.get_logits_and_probs(
          probs=p3, validate_args=True)
      self.evaluate(prob)

    _, prob = distribution_util.get_logits_and_probs(
        probs=p3, validate_args=False)
    self.evaluate(prob)

  @tfe.run_test_in_graph_and_eager_modes
  def testProbabilityValidateArgsMultidimensional(self):
    p = np.array([[0.3, 0.4, 0.3], [0.1, 0.5, 0.4]], dtype=np.float32)
    # Component less than 0. Still sums to 1.
    p2 = np.array([[-.3, 0.4, 0.9], [0.1, 0.5, 0.4]], dtype=np.float32)
    # Component greater than 1. Does not sum to 1.
    p3 = np.array([[1.3, 0.0, 0.0], [0.1, 0.5, 0.4]], dtype=np.float32)
    # Does not sum to 1.
    p4 = np.array([[1.1, 0.3, 0.4], [0.1, 0.5, 0.4]], dtype=np.float32)

    _, prob = distribution_util.get_logits_and_probs(
        probs=p, multidimensional=True)
    self.evaluate(prob)

    with self.assertRaisesOpError("Condition x >= 0"):
      _, prob = distribution_util.get_logits_and_probs(
          probs=p2, multidimensional=True, validate_args=True)
      self.evaluate(prob)

    _, prob = distribution_util.get_logits_and_probs(
        probs=p2, multidimensional=True, validate_args=False)
    self.evaluate(prob)

    with self.assertRaisesOpError(
        "(probs has components greater than 1|probs does not sum to 1)"):
      _, prob = distribution_util.get_logits_and_probs(
          probs=p3, multidimensional=True, validate_args=True)
      self.evaluate(prob)

    _, prob = distribution_util.get_logits_and_probs(
        probs=p3, multidimensional=True, validate_args=False)
    self.evaluate(prob)

    with self.assertRaisesOpError("probs does not sum to 1"):
      _, prob = distribution_util.get_logits_and_probs(
          probs=p4, multidimensional=True, validate_args=True)
      self.evaluate(prob)

    _, prob = distribution_util.get_logits_and_probs(
        probs=p4, multidimensional=True, validate_args=False)
    self.evaluate(prob)

  def testProbsMultidimShape(self):
    with self.cached_session():
      with self.assertRaises(ValueError):
        p = tf.ones([int(2**11+1)], dtype=np.float16)
        distribution_util.get_logits_and_probs(
            probs=p, multidimensional=True, validate_args=True)

      with self.assertRaisesOpError(
          "Number of classes exceeds `dtype` precision"):
        p = tf.placeholder(dtype=tf.float16)
        _, prob = distribution_util.get_logits_and_probs(
            probs=p, multidimensional=True, validate_args=True)
        prob.eval(feed_dict={p: np.ones([int(2**11+1)])})

  def testLogitsMultidimShape(self):
    with self.cached_session():
      with self.assertRaises(ValueError):
        l = tf.ones([int(2**11+1)], dtype=np.float16)
        distribution_util.get_logits_and_probs(
            logits=l, multidimensional=True, validate_args=True)

      with self.assertRaisesOpError(
          "Number of classes exceeds `dtype` precision"):
        l = tf.placeholder(dtype=tf.float16)
        logit, _ = distribution_util.get_logits_and_probs(
            logits=l, multidimensional=True, validate_args=True)
        logit.eval(feed_dict={l: np.ones([int(2**11+1)])})


class EmbedCheckCategoricalEventShapeTest(tf.test.TestCase):

  def testTooSmall(self):
    with self.cached_session():
      with self.assertRaises(ValueError):
        param = tf.ones([1], dtype=np.float16)
        checked_param = distribution_util.embed_check_categorical_event_shape(
            param)

      with self.assertRaisesOpError(
          "must have at least 2 events"):
        param = tf.placeholder(dtype=tf.float16)
        checked_param = distribution_util.embed_check_categorical_event_shape(
            param)
        checked_param.eval(feed_dict={param: np.ones([1])})

  def testTooLarge(self):
    with self.cached_session():
      with self.assertRaises(ValueError):
        param = tf.ones([int(2**11+1)], dtype=tf.float16)
        checked_param = distribution_util.embed_check_categorical_event_shape(
            param)

      with self.assertRaisesOpError(
          "Number of classes exceeds `dtype` precision"):
        param = tf.placeholder(dtype=tf.float16)
        checked_param = distribution_util.embed_check_categorical_event_shape(
            param)
        checked_param.eval(feed_dict={param: np.ones([int(2**11+1)])})

  @tfe.run_test_in_graph_and_eager_modes
  def testUnsupportedDtype(self):
    param = tf.convert_to_tensor(
        np.ones([2**11 + 1]).astype(tf.qint16.as_numpy_dtype),
        dtype=tf.qint16)
    with self.assertRaises(TypeError):
      distribution_util.embed_check_categorical_event_shape(param)


class EmbedCheckIntegerCastingClosedTest(tf.test.TestCase):

  def testCorrectlyAssertsNonnegative(self):
    with self.cached_session():
      with self.assertRaisesOpError("Elements must be non-negative"):
        x = tf.placeholder(dtype=tf.float16)
        x_checked = distribution_util.embed_check_integer_casting_closed(
            x, target_dtype=tf.int16)
        x_checked.eval(feed_dict={x: np.array([1, -1], dtype=np.float16)})

  def testCorrectlyAssertsPositive(self):
    with self.cached_session():
      with self.assertRaisesOpError("Elements must be positive"):
        x = tf.placeholder(dtype=tf.float16)
        x_checked = distribution_util.embed_check_integer_casting_closed(
            x, target_dtype=tf.int16, assert_positive=True)
        x_checked.eval(feed_dict={x: np.array([1, 0], dtype=np.float16)})

  def testCorrectlyAssersIntegerForm(self):
    with self.cached_session():
      with self.assertRaisesOpError("Elements must be int16-equivalent."):
        x = tf.placeholder(dtype=tf.float16)
        x_checked = distribution_util.embed_check_integer_casting_closed(
            x, target_dtype=tf.int16)
        x_checked.eval(feed_dict={x: np.array([1, 1.5], dtype=np.float16)})

  def testCorrectlyAssertsLargestPossibleInteger(self):
    with self.cached_session():
      with self.assertRaisesOpError("Elements cannot exceed 32767."):
        x = tf.placeholder(dtype=tf.int32)
        x_checked = distribution_util.embed_check_integer_casting_closed(
            x, target_dtype=tf.int16)
        x_checked.eval(feed_dict={x: np.array([1, 2**15], dtype=np.int32)})

  def testCorrectlyAssertsSmallestPossibleInteger(self):
    with self.cached_session():
      with self.assertRaisesOpError("Elements cannot be smaller than 0."):
        x = tf.placeholder(dtype=tf.int32)
        x_checked = distribution_util.embed_check_integer_casting_closed(
            x, target_dtype=tf.uint16, assert_nonnegative=False)
        x_checked.eval(feed_dict={x: np.array([1, -1], dtype=np.int32)})


@tfe.run_all_tests_in_graph_and_eager_modes
class LogCombinationsTest(tf.test.TestCase):

  def testLogCombinationsBinomial(self):
    n = [2, 5, 12, 15]
    k = [1, 2, 4, 11]

    if not special:
      return

    log_combs = np.log(special.binom(n, k))

    n = np.array(n, dtype=np.float32)
    counts = [[1., 1], [2., 3], [4., 8], [11, 4]]
    log_binom = distribution_util.log_combinations(n, counts)
    self.assertEqual([4], log_binom.shape)
    self.assertAllClose(log_combs, self.evaluate(log_binom))

  def testLogCombinationsShape(self):
    # Shape [2, 2]
    n = [[2, 5], [12, 15]]

    n = np.array(n, dtype=np.float32)
    # Shape [2, 2, 4]
    counts = [[[1., 1, 0, 0], [2., 2, 1, 0]], [[4., 4, 1, 3], [10, 1, 1, 4]]]
    log_binom = distribution_util.log_combinations(n, counts)
    self.assertEqual([2, 2], log_binom.shape)


class DynamicShapeTest(tf.test.TestCase):

  def testSameDynamicShape(self):
    with self.cached_session():
      scalar = tf.constant(2.0)
      scalar1 = tf.placeholder(dtype=tf.float32)

      vector = [0.3, 0.4, 0.5]
      vector1 = tf.placeholder(dtype=tf.float32, shape=[None])
      vector2 = tf.placeholder(dtype=tf.float32, shape=[None])

      multidimensional = [[0.3, 0.4], [0.2, 0.6]]
      multidimensional1 = tf.placeholder(
          dtype=tf.float32, shape=[None, None])
      multidimensional2 = tf.placeholder(
          dtype=tf.float32, shape=[None, None])

      # Scalar
      self.assertTrue(
          distribution_util.same_dynamic_shape(scalar, scalar1).eval({
              scalar1: 2.0
          }))

      # Vector

      self.assertTrue(
          distribution_util.same_dynamic_shape(vector, vector1).eval({
              vector1: [2.0, 3.0, 4.0]
          }))
      self.assertTrue(
          distribution_util.same_dynamic_shape(vector1, vector2).eval({
              vector1: [2.0, 3.0, 4.0],
              vector2: [2.0, 3.5, 6.0]
          }))

      # Multidimensional
      self.assertTrue(
          distribution_util.same_dynamic_shape(
              multidimensional, multidimensional1).eval({
                  multidimensional1: [[2.0, 3.0], [3.0, 4.0]]
              }))
      self.assertTrue(
          distribution_util.same_dynamic_shape(
              multidimensional1, multidimensional2).eval({
                  multidimensional1: [[2.0, 3.0], [3.0, 4.0]],
                  multidimensional2: [[1.0, 3.5], [6.3, 2.3]]
              }))

      # Scalar, X
      self.assertFalse(
          distribution_util.same_dynamic_shape(scalar, vector1).eval({
              vector1: [2.0, 3.0, 4.0]
          }))
      self.assertFalse(
          distribution_util.same_dynamic_shape(scalar1, vector1).eval({
              scalar1: 2.0,
              vector1: [2.0, 3.0, 4.0]
          }))
      self.assertFalse(
          distribution_util.same_dynamic_shape(scalar, multidimensional1).eval({
              multidimensional1: [[2.0, 3.0], [3.0, 4.0]]
          }))
      self.assertFalse(
          distribution_util.same_dynamic_shape(scalar1, multidimensional1).eval(
              {
                  scalar1: 2.0,
                  multidimensional1: [[2.0, 3.0], [3.0, 4.0]]
              }))

      # Vector, X
      self.assertFalse(
          distribution_util.same_dynamic_shape(vector, vector1).eval({
              vector1: [2.0, 3.0]
          }))
      self.assertFalse(
          distribution_util.same_dynamic_shape(vector1, vector2).eval({
              vector1: [2.0, 3.0, 4.0],
              vector2: [6.0]
          }))
      self.assertFalse(
          distribution_util.same_dynamic_shape(vector, multidimensional1).eval({
              multidimensional1: [[2.0, 3.0], [3.0, 4.0]]
          }))
      self.assertFalse(
          distribution_util.same_dynamic_shape(vector1, multidimensional1).eval(
              {
                  vector1: [2.0, 3.0, 4.0],
                  multidimensional1: [[2.0, 3.0], [3.0, 4.0]]
              }))

      # Multidimensional, X
      self.assertFalse(
          distribution_util.same_dynamic_shape(
              multidimensional, multidimensional1).eval({
                  multidimensional1: [[1.0, 3.5, 5.0], [6.3, 2.3, 7.1]]
              }))
      self.assertFalse(
          distribution_util.same_dynamic_shape(
              multidimensional1, multidimensional2).eval({
                  multidimensional1: [[2.0, 3.0], [3.0, 4.0]],
                  multidimensional2: [[1.0, 3.5, 5.0], [6.3, 2.3, 7.1]]
              }))


class RotateTransposeTest(tf.test.TestCase):

  def _np_rotate_transpose(self, x, shift):
    if not isinstance(x, np.ndarray):
      x = np.array(x)
    return np.transpose(x, np.roll(np.arange(len(x.shape)), shift))

  @tfe.run_test_in_graph_and_eager_modes
  def testRollStatic(self):
    if tf.executing_eagerly():
      error_message = r"Attempt to convert a value \(None\)"
    else:
      error_message = "None values not supported."
    with self.assertRaisesRegexp(ValueError, error_message):
      distribution_util.rotate_transpose(None, 1)
    for x in (np.ones(1), np.ones((2, 1)), np.ones((3, 2, 1))):
      for shift in np.arange(-5, 5):
        y = distribution_util.rotate_transpose(x, shift)
        self.assertAllEqual(
            self._np_rotate_transpose(x, shift), self.evaluate(y))
        self.assertAllEqual(np.roll(x.shape, shift), y.shape.as_list())

  def testRollDynamic(self):
    with self.cached_session() as sess:
      x = tf.placeholder(tf.float32)
      shift = tf.placeholder(tf.int32)
      for x_value in (np.ones(
          1, dtype=x.dtype.as_numpy_dtype()), np.ones(
              (2, 1), dtype=x.dtype.as_numpy_dtype()), np.ones(
                  (3, 2, 1), dtype=x.dtype.as_numpy_dtype())):
        for shift_value in np.arange(-5, 5):
          self.assertAllEqual(
              self._np_rotate_transpose(x_value, shift_value),
              sess.run(distribution_util.rotate_transpose(x, shift),
                       feed_dict={x: x_value,
                                  shift: shift_value}))


class PickVectorTest(tf.test.TestCase):

  def testCorrectlyPicksVector(self):
    with self.cached_session():
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


class PreferStaticRankTest(tf.test.TestCase):

  def testNonEmptyConstantTensor(self):
    x = tf.zeros((2, 3, 4))
    rank = distribution_util.prefer_static_rank(x)
    self.assertIsInstance(rank, np.ndarray)
    self.assertEqual(3, rank)

  def testEmptyConstantTensor(self):
    x = tf.constant([])
    rank = distribution_util.prefer_static_rank(x)
    self.assertIsInstance(rank, np.ndarray)
    self.assertEqual(1, rank)

  def testScalarTensor(self):
    x = tf.constant(1.)
    rank = distribution_util.prefer_static_rank(x)
    self.assertIsInstance(rank, np.ndarray)
    self.assertEqual(0, rank)

  def testDynamicRankEndsUpBeingNonEmpty(self):
    x = tf.placeholder(np.float64, shape=None)
    rank = distribution_util.prefer_static_rank(x)
    with self.cached_session():
      self.assertAllEqual(2, rank.eval(feed_dict={x: np.zeros((2, 3))}))

  def testDynamicRankEndsUpBeingEmpty(self):
    x = tf.placeholder(np.int32, shape=None)
    rank = distribution_util.prefer_static_rank(x)
    with self.cached_session():
      self.assertAllEqual(1, rank.eval(feed_dict={x: []}))

  def testDynamicRankEndsUpBeingScalar(self):
    x = tf.placeholder(np.int32, shape=None)
    rank = distribution_util.prefer_static_rank(x)
    with self.cached_session():
      self.assertAllEqual(0, rank.eval(feed_dict={x: 1}))


class PreferStaticShapeTest(tf.test.TestCase):

  def testNonEmptyConstantTensor(self):
    x = tf.zeros((2, 3, 4))
    shape = distribution_util.prefer_static_shape(x)
    self.assertIsInstance(shape, np.ndarray)
    self.assertAllEqual(np.array([2, 3, 4]), shape)

  def testEmptyConstantTensor(self):
    x = tf.constant([])
    shape = distribution_util.prefer_static_shape(x)
    self.assertIsInstance(shape, np.ndarray)
    self.assertAllEqual(np.array([0]), shape)

  def testScalarTensor(self):
    x = tf.constant(1.)
    shape = distribution_util.prefer_static_shape(x)
    self.assertIsInstance(shape, np.ndarray)
    self.assertAllEqual(np.array([]), shape)

  def testDynamicShapeEndsUpBeingNonEmpty(self):
    x = tf.placeholder(np.float64, shape=None)
    shape = distribution_util.prefer_static_shape(x)
    with self.cached_session():
      self.assertAllEqual((2, 3), shape.eval(feed_dict={x: np.zeros((2, 3))}))

  def testDynamicShapeEndsUpBeingEmpty(self):
    x = tf.placeholder(np.int32, shape=None)
    shape = distribution_util.prefer_static_shape(x)
    with self.cached_session():
      self.assertAllEqual(np.array([0]), shape.eval(feed_dict={x: []}))

  def testDynamicShapeEndsUpBeingScalar(self):
    x = tf.placeholder(np.int32, shape=None)
    shape = distribution_util.prefer_static_shape(x)
    with self.cached_session():
      self.assertAllEqual(np.array([]), shape.eval(feed_dict={x: 1}))


class PreferStaticValueTest(tf.test.TestCase):

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
    self.assertIsInstance(value, np.ndarray)
    self.assertAllEqual(np.array(1.), value)

  def testDynamicValueEndsUpBeingNonEmpty(self):
    x = tf.placeholder(np.float64, shape=None)
    value = distribution_util.prefer_static_value(x)
    with self.cached_session():
      self.assertAllEqual(np.zeros((2, 3)),
                          value.eval(feed_dict={x: np.zeros((2, 3))}))

  def testDynamicValueEndsUpBeingEmpty(self):
    x = tf.placeholder(np.int32, shape=None)
    value = distribution_util.prefer_static_value(x)
    with self.cached_session():
      self.assertAllEqual(np.array([]), value.eval(feed_dict={x: []}))

  def testDynamicValueEndsUpBeingScalar(self):
    x = tf.placeholder(np.int32, shape=None)
    value = distribution_util.prefer_static_value(x)
    with self.cached_session():
      self.assertAllEqual(np.array(1), value.eval(feed_dict={x: 1}))


class FillTriangularTest(tf.test.TestCase):

  def setUp(self):
    self._rng = np.random.RandomState(42)

  def _fill_triangular(self, x, upper=False):
    """Numpy implementation of `fill_triangular`."""
    x = np.asarray(x)
    # Formula derived by solving for n: m = n(n+1)/2.
    m = np.int32(x.shape[-1])
    n = np.sqrt(0.25 + 2. * m) - 0.5
    if n != np.floor(n):
      raise ValueError("Invalid shape.")
    n = np.int32(n)
    # We can't do: `x[..., -(n**2-m):]` because this doesn't correctly handle
    # `m == n == 1`. Hence, we do absolute indexing.
    x_tail = x[..., (m - (n * n - m)):]
    y = np.concatenate(
        [x, x_tail[..., ::-1]] if upper else [x_tail, x[..., ::-1]],
        axis=-1)
    y = y.reshape(np.concatenate([
        np.int32(x.shape[:-1]),
        np.int32([n, n]),
    ], axis=0))
    return np.triu(y) if upper else np.tril(y)

  def _run_test(self, x_, use_deferred_shape=False, **kwargs):
    x_ = np.asarray(x_)
    with self.cached_session() as sess:
      static_shape = None if use_deferred_shape else x_.shape
      x_pl = tf.placeholder_with_default(x_, shape=static_shape)
      # Add `zeros_like(x)` such that x's value and gradient are identical. We
      # do this so we can ensure each gradient value is mapped to the right
      # gradient location.  (Not doing this means the gradient wrt `x` is simple
      # `ones_like(x)`.)
      # Note:
      #   zeros_like_x_pl == zeros_like(x_pl)
      #   gradient(zeros_like_x_pl, x_pl) == x_pl - 1
      zeros_like_x_pl = (x_pl * tf.stop_gradient(x_pl - 1.)
                         - tf.stop_gradient(x_pl * (x_pl - 1.)))
      x = x_pl + zeros_like_x_pl
      actual = distribution_util.fill_triangular(x, **kwargs)
      grad_actual = tf.gradients(actual, x_pl)[0]
      [actual_, grad_actual_] = sess.run([actual, grad_actual],
                                         feed_dict={x_pl: x_})
    expected = self._fill_triangular(x_, **kwargs)
    if use_deferred_shape:
      self.assertEqual(None, actual.shape)
    else:
      self.assertAllEqual(expected.shape, actual.shape)
    self.assertAllClose(expected, actual_, rtol=1e-8, atol=1e-9)
    self.assertAllClose(x_, grad_actual_, rtol=1e-8, atol=1e-9)

  def testCorrectlyMakes1x1TriLower(self):
    self._run_test(self._rng.randn(3, int(1*2/2)))

  def testCorrectlyMakesNoBatchTriLower(self):
    self._run_test(self._rng.randn(int(4*5/2)))

  def testCorrectlyMakesBatchTriLower(self):
    self._run_test(self._rng.randn(2, 3, int(3*4/2)))

  def testCorrectlyMakesBatchTriLowerUnknownShape(self):
    self._run_test(self._rng.randn(2, 3, int(3*4/2)), use_deferred_shape=True)

  def testCorrectlyMakesBatch7x7TriLowerUnknownShape(self):
    self._run_test(self._rng.randn(2, 3, int(7*8/2)), use_deferred_shape=True)

  def testCorrectlyMakesBatch7x7TriLower(self):
    self._run_test(self._rng.randn(2, 3, int(7*8/2)))

  def testCorrectlyMakes1x1TriUpper(self):
    self._run_test(self._rng.randn(3, int(1*2/2)), upper=True)

  def testCorrectlyMakesNoBatchTriUpper(self):
    self._run_test(self._rng.randn(int(4*5/2)), upper=True)

  def testCorrectlyMakesBatchTriUpper(self):
    self._run_test(self._rng.randn(2, 2, int(3*4/2)), upper=True)

  def testCorrectlyMakesBatchTriUpperUnknownShape(self):
    self._run_test(self._rng.randn(2, 2, int(3*4/2)),
                   use_deferred_shape=True,
                   upper=True)

  def testCorrectlyMakesBatch7x7TriUpperUnknownShape(self):
    self._run_test(self._rng.randn(2, 3, int(7*8/2)),
                   use_deferred_shape=True,
                   upper=True)

  def testCorrectlyMakesBatch7x7TriUpper(self):
    self._run_test(self._rng.randn(2, 3, int(7*8/2)), upper=True)


class FillTriangularInverseTest(FillTriangularTest):

  def _run_test(self, x_, use_deferred_shape=False, **kwargs):
    x_ = np.asarray(x_)
    with self.cached_session() as sess:
      static_shape = None if use_deferred_shape else x_.shape
      x_pl = tf.placeholder_with_default(x_, shape=static_shape)
      zeros_like_x_pl = (x_pl * tf.stop_gradient(x_pl - 1.)
                         - tf.stop_gradient(x_pl * (x_pl - 1.)))
      x = x_pl + zeros_like_x_pl
      actual = distribution_util.fill_triangular(x, **kwargs)
      inverse_actual = distribution_util.fill_triangular_inverse(
          actual, **kwargs)

      inverse_actual_ = sess.run(
          inverse_actual,
          feed_dict={x_pl: x_})

    if use_deferred_shape:
      self.assertEqual(None, inverse_actual.shape)
    else:
      self.assertAllEqual(x_.shape, inverse_actual.shape)
    self.assertAllEqual(x_, inverse_actual_)


class ReduceWeightedLogSumExp(tf.test.TestCase):

  def _reduce_weighted_logsumexp(self, logx, w, axis, keep_dims=False):
    m = np.max(logx, axis=axis, keepdims=True)
    sum_ = np.sum(w * np.exp(logx - m), axis=axis, keepdims=keep_dims)
    sgn = np.sign(sum_)
    if not keep_dims:
      m = np.squeeze(m, axis=axis)
    return m + np.log(sgn * sum_), sgn

  def testNoWeights(self):
    logx_ = np.array([[0., -1, 1000.],
                      [0, 1, -1000.],
                      [-5, 0, 5]])
    with self.cached_session() as sess:
      logx = tf.constant(logx_)
      expected = tf.reduce_logsumexp(logx, axis=-1)
      grad_expected = tf.gradients(expected, logx)[0]
      actual, actual_sgn = distribution_util.reduce_weighted_logsumexp(
          logx, axis=-1, return_sign=True)
      grad_actual = tf.gradients(actual, logx)[0]
      [actual_, actual_sgn_, grad_actual_,
       expected_, grad_expected_] = sess.run([
           actual, actual_sgn, grad_actual,
           expected, grad_expected])
    self.assertAllEqual(expected_, actual_)
    self.assertAllEqual(grad_expected_, grad_actual_)
    self.assertAllEqual([1., 1, 1], actual_sgn_)

  def testNegativeWeights(self):
    logx_ = np.array([[0., -1, 1000.],
                      [0, 1, -1000.],
                      [-5, 0, 5]])
    w_ = np.array([[1., 1, -1],
                   [1, -2, 1],
                   [1, 0, 1]])
    expected, _ = self._reduce_weighted_logsumexp(logx_, w_, axis=-1)
    with self.cached_session() as sess:
      logx = tf.constant(logx_)
      w = tf.constant(w_)
      actual, actual_sgn = distribution_util.reduce_weighted_logsumexp(
          logx, w, axis=-1, return_sign=True)
      [actual_, actual_sgn_] = sess.run([actual, actual_sgn])
    self.assertAllEqual(expected, actual_)
    self.assertAllEqual([-1., -1, 1], actual_sgn_)

  def testKeepDims(self):
    logx_ = np.array([[0., -1, 1000.],
                      [0, 1, -1000.],
                      [-5, 0, 5]])
    w_ = np.array([[1., 1, -1],
                   [1, -2, 1],
                   [1, 0, 1]])
    expected, _ = self._reduce_weighted_logsumexp(
        logx_, w_, axis=-1, keep_dims=True)
    with self.cached_session() as sess:
      logx = tf.constant(logx_)
      w = tf.constant(w_)
      actual, actual_sgn = distribution_util.reduce_weighted_logsumexp(
          logx, w, axis=-1, return_sign=True, keep_dims=True)
      [actual_, actual_sgn_] = sess.run([actual, actual_sgn])
    self.assertAllEqual(expected, actual_)
    self.assertAllEqual([[-1.], [-1], [1]], actual_sgn_)

  def testDocString(self):
    """This test verifies the correctness of the docstring examples."""

    with self.cached_session():
      x = tf.constant([[0., 0, 0],
                       [0, 0, 0]])

      w = tf.constant([[-1., 1, 1],
                       [1, 1, 1]])

      self.assertAllClose(
          np.log(4),
          self.evaluate(distribution_util.reduce_weighted_logsumexp(x, w)))

      with np.errstate(divide="ignore"):
        self.assertAllClose(
            np.log([0, 2, 2]),
            self.evaluate(
                distribution_util.reduce_weighted_logsumexp(x, w, axis=0)))

      self.assertAllClose(
          np.log([1, 3]),
          self.evaluate(
              distribution_util.reduce_weighted_logsumexp(x, w, axis=1)))

      self.assertAllClose(
          np.log([[1], [3]]),
          self.evaluate(
              distribution_util.reduce_weighted_logsumexp(
                  x, w, axis=1, keep_dims=True)))

      self.assertAllClose(
          np.log(4),
          self.evaluate(
              distribution_util.reduce_weighted_logsumexp(x, w, axis=[0, 1])))


class GenNewSeedTest(tf.test.TestCase):

  def testOnlyNoneReturnsNone(self):
    self.assertFalse(distribution_util.gen_new_seed(0, "salt") is None)
    self.assertTrue(distribution_util.gen_new_seed(None, "salt") is None)


# TODO(jvdillon): Merge this test back into:
# tensorflow/python/kernel_tests/softplus_op_test.py
# once TF core is accepting new ops.
class SoftplusTest(tf.test.TestCase):

  def _npSoftplus(self, np_features):
    np_features = np.asarray(np_features)
    zero = np.asarray(0).astype(np_features.dtype)
    return np.logaddexp(zero, np_features)

  def _testSoftplus(self, np_features, use_gpu=False):
    np_features = np.asarray(np_features)
    np_softplus = self._npSoftplus(np_features)
    with self.cached_session(use_gpu=use_gpu) as sess:
      softplus = tf.nn.softplus(np_features)
      softplus_inverse = distribution_util.softplus_inverse(softplus)
      [tf_softplus, tf_softplus_inverse] = sess.run([
          softplus, softplus_inverse])
    self.assertAllCloseAccordingToType(np_softplus, tf_softplus)
    rtol = {"float16": 0.07, "float32": 0.003, "float64": 0.002}.get(
        str(np_features.dtype), 1e-6)
    # This will test that we correctly computed the inverse by verifying we
    # recovered the original input.
    self.assertAllCloseAccordingToType(
        np_features, tf_softplus_inverse,
        atol=0., rtol=rtol)
    self.assertAllEqual(np.ones_like(tf_softplus).astype(np.bool),
                        tf_softplus > 0)

    self.assertShapeEqual(np_softplus, softplus)
    self.assertShapeEqual(np_softplus, softplus_inverse)

    self.assertAllEqual(np.ones_like(tf_softplus).astype(np.bool),
                        np.isfinite(tf_softplus))
    self.assertAllEqual(np.ones_like(tf_softplus_inverse).astype(np.bool),
                        np.isfinite(tf_softplus_inverse))

  def testNumbers(self):
    for t in [np.float16, np.float32, np.float64]:
      lower = {np.float16: -15, np.float32: -50, np.float64: -50}.get(t, -100)
      upper = {np.float16: 50, np.float32: 50, np.float64: 50}.get(t, 100)
      self._testSoftplus(
          np.array(np.linspace(lower, upper, int(1e3)).astype(t)).reshape(
              [2, -1]),
          use_gpu=False)
      self._testSoftplus(
          np.array(np.linspace(lower, upper, int(1e3)).astype(t)).reshape(
              [2, -1]),
          use_gpu=True)
      log_eps = np.log(np.finfo(t).eps)
      one = t(1)
      ten = t(10)
      self._testSoftplus(
          [
              log_eps, log_eps - one, log_eps + one, log_eps - ten,
              log_eps + ten, -log_eps, -log_eps - one, -log_eps + one,
              -log_eps - ten, -log_eps + ten
          ],
          use_gpu=False)
      self._testSoftplus(
          [
              log_eps, log_eps - one, log_eps + one, log_eps - ten,
              log_eps + ten - log_eps, -log_eps - one, -log_eps + one,
              -log_eps - ten, -log_eps + ten
          ],
          use_gpu=True)

  def testGradient(self):
    with self.cached_session():
      x = tf.constant(
          [-0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9],
          shape=[2, 5],
          name="x")
      y = tf.nn.softplus(x, name="softplus")
      x_init = np.asarray(
          [[-0.9, -0.7, -0.5, -0.3, -0.1], [0.1, 0.3, 0.5, 0.7, 0.9]],
          dtype=np.float32,
          order="F")
      err = tf.test.compute_gradient_error(
          x, [2, 5], y, [2, 5], x_init_value=x_init)
    tf.logging.vlog(2, "softplus (float) gradient err = ", err)
    self.assertLess(err, 1e-4)

  def testInverseSoftplusGradientNeverNan(self):
    with self.cached_session():
      # Note that this range contains both zero and inf.
      x = tf.constant(np.logspace(-8, 6).astype(np.float16))
      y = distribution_util.softplus_inverse(x)
      grads = self.evaluate(tf.gradients(y, x)[0])
      # Equivalent to `assertAllFalse` (if it existed).
      self.assertAllEqual(np.zeros_like(grads).astype(np.bool), np.isnan(grads))

  def testInverseSoftplusGradientFinite(self):
    with self.cached_session():
      # This range of x is all finite, and so is 1 / x.  So the
      # gradient and its approximations should be finite as well.
      x = tf.constant(np.logspace(-4.8, 4.5).astype(np.float16))
      y = distribution_util.softplus_inverse(x)
      grads = self.evaluate(tf.gradients(y, x)[0])
      # Equivalent to `assertAllTrue` (if it existed).
      self.assertAllEqual(
          np.ones_like(grads).astype(np.bool), np.isfinite(grads))


@tfe.run_all_tests_in_graph_and_eager_modes
class ArgumentsTest(tf.test.TestCase):

  def testNoArguments(self):
    def foo():
      return distribution_util.parent_frame_arguments()

    self.assertEqual({}, foo())

  def testPositionalArguments(self):
    def foo(a, b, c, d):  # pylint: disable=unused-argument
      return distribution_util.parent_frame_arguments()

    self.assertEqual({"a": 1, "b": 2, "c": 3, "d": 4}, foo(1, 2, 3, 4))

    # Tests that it does not matter where this function is called, and
    # no other local variables are returned back.
    def bar(a, b, c):
      unused_x = a * b
      unused_y = c * 3
      return distribution_util.parent_frame_arguments()

    self.assertEqual({"a": 1, "b": 2, "c": 3}, bar(1, 2, 3))

  def testOverloadedArgumentValues(self):
    def foo(a, b, c):  # pylint: disable=unused-argument
      a = 42
      b = 31
      c = 42
      return distribution_util.parent_frame_arguments()
    self.assertEqual({"a": 42, "b": 31, "c": 42}, foo(1, 2, 3))

  def testKeywordArguments(self):
    def foo(**kwargs):  # pylint: disable=unused-argument
      return distribution_util.parent_frame_arguments()

    self.assertEqual({"a": 1, "b": 2, "c": 3, "d": 4}, foo(a=1, b=2, c=3, d=4))

  def testPositionalKeywordArgs(self):
    def foo(a, b, c, **kwargs):  # pylint: disable=unused-argument
      return distribution_util.parent_frame_arguments()

    self.assertEqual({"a": 1, "b": 2, "c": 3}, foo(a=1, b=2, c=3))
    self.assertEqual({"a": 1, "b": 2, "c": 3, "unicorn": None},
                     foo(a=1, b=2, c=3, unicorn=None))

  def testNoVarargs(self):
    def foo(a, b, c, *varargs, **kwargs):  # pylint: disable=unused-argument
      return distribution_util.parent_frame_arguments()

    self.assertEqual({"a": 1, "b": 2, "c": 3}, foo(a=1, b=2, c=3))
    self.assertEqual({"a": 1, "b": 2, "c": 3}, foo(1, 2, 3, *[1, 2, 3]))
    self.assertEqual({"a": 1, "b": 2, "c": 3, "unicorn": None},
                     foo(1, 2, 3, unicorn=None))
    self.assertEqual({"a": 1, "b": 2, "c": 3, "unicorn": None},
                     foo(1, 2, 3, *[1, 2, 3], unicorn=None))


@tfe.run_all_tests_in_graph_and_eager_modes
class ExpandToVectorTest(tf.test.TestCase):

  def _check_static(self, expected, actual, dtype=np.int32):
    const_actual = tf.contrib.util.constant_value(actual)
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
          tf.convert_to_tensor(x, dtype=dtype), op_name="test")

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
          tf.placeholder_with_default(np.array(x, dtype=dtype), shape=shape),
          tensor_name="name_for_tensor",
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


if __name__ == "__main__":
  tf.test.main()
