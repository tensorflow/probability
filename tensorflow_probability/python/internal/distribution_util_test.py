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
"""Tests for utility functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

# Dependency imports
import numpy as np
import tensorflow as tf


from tensorflow_probability.python.distributions import Mixture
from tensorflow_probability.python.distributions import MixtureSameFamily
from tensorflow_probability.python.distributions import MultivariateNormalDiag

from tensorflow_probability.python.internal import distribution_util
from tensorflow.python.framework import test_util


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
      with self.test_session():
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
    with self.test_session():
      scale = distribution_util.make_tril_scale(scale_tril=[[1., 1], [1., 1.]])
      self.assertAllClose([[1., 0], [1., 1.]], self.evaluate(scale.to_dense()))

  def testValidateArgs(self):
    with self.test_session():
      with self.assertRaisesOpError("diagonal part must be non-zero"):
        scale = distribution_util.make_tril_scale(
            scale_tril=[[0., 1], [1., 1.]], validate_args=True)
        self.evaluate(scale.to_dense())

  def testAssertPositive(self):
    with self.test_session():
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
      with self.test_session():
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
    with self.test_session():
      with self.assertRaisesOpError("diagonal part must be non-zero"):
        scale = distribution_util.make_diag_scale(
            scale_diag=[[0., 1], [1., 1.]], validate_args=True)
        self.evaluate(scale.to_dense())

  def testAssertPositive(self):
    with self.test_session():
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
    with self.test_session() as sess:
      batch_shape, event_shape = sess.run(
          distribution_util.shapes_from_loc_and_scale(loc, scale),
          feed_dict={diag: np.ones((5, 1, 3))})
      self.assertAllEqual([5, 2], batch_shape)
      self.assertAllEqual([3], event_shape)

  def test_dynamic_loc_static_scale(self):
    loc = tf.placeholder(tf.float64)
    diag = tf.constant(np.ones((5, 2, 3)))
    scale = tf.linalg.LinearOperatorDiag(diag)
    with self.test_session():
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
    with self.test_session() as sess:
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
    with self.test_session() as sess:
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
    with self.test_session() as sess:
      bcast_shape = sess.run(
          distribution_util.get_broadcast_shape(x, y, z),
          feed_dict={y: np.ones((1, 5, 3)).astype(np.float32)})
      self.assertAllEqual([2, 5, 3], bcast_shape)


class TridiagTest(tf.test.TestCase):

  def testWorksCorrectlyNoBatches(self):
    with self.test_session():
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
    with self.test_session():
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
    with self.test_session():
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

    with self.test_session() as sess:
      actual_devs = sess.run(mix_dev)

    self.assertAllClose(actual_devs, expected_devs)


class PadMixtureDimensionsTest(tf.test.TestCase):

  def test_pad_mixture_dimensions_mixture(self):
    with self.test_session() as sess:
      gm = Mixture(
          cat=tf.distributions.Categorical(probs=[[0.3, 0.7]]),
          components=[
              tf.distributions.Normal(loc=[-1.0], scale=[1.0]),
              tf.distributions.Normal(loc=[1.0], scale=[0.5])
          ])

      x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
      x_pad = distribution_util.pad_mixture_dimensions(
          x, gm, gm.cat, gm.event_shape.ndims)
      x_out, x_pad_out = sess.run([x, x_pad])

    self.assertAllEqual(x_pad_out.shape, [2, 2])
    self.assertAllEqual(x_out.reshape([-1]), x_pad_out.reshape([-1]))

  def test_pad_mixture_dimensions_mixture_same_family(self):
    with self.test_session() as sess:
      gm = MixtureSameFamily(
          mixture_distribution=tf.distributions.Categorical(probs=[0.3, 0.7]),
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
    with self.test_session() as sess:
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
    with self.test_session() as sess:
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


class TestMoveDimension(tf.test.TestCase):

  @test_util.run_in_graph_and_eager_modes()
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

  @test_util.run_in_graph_and_eager_modes()
  def test_move_dimension_dynamic_shape(self):

    x_ = tf.random_normal(shape=[200, 30, 4, 1, 6])
    x = tf.placeholder_with_default(input=x_, shape=None)

    x_perm = distribution_util.move_dimension(x, 1, 1)
    self.assertAllEqual(self.evaluate(tf.shape(x_perm)), [200, 30, 4, 1, 6])

    x_perm = distribution_util.move_dimension(x, 0, 3)
    self.assertAllEqual(self.evaluate(tf.shape(x_perm)), [30, 4, 1, 200, 6])

    x_perm = distribution_util.move_dimension(x, 0, -2)
    self.assertAllEqual(self.evaluate(tf.shape(x_perm)), [30, 4, 1, 200, 6])

    x_perm = distribution_util.move_dimension(x, 4, 2)
    self.assertAllEqual(self.evaluate(tf.shape(x_perm)), [200, 30, 6, 4, 1])

    x_perm = distribution_util.move_dimension(x, -1, 2)
    self.assertAllEqual(self.evaluate(tf.shape(x_perm)), [200, 30, 6, 4, 1])


if __name__ == "__main__":
  tf.test.main()
