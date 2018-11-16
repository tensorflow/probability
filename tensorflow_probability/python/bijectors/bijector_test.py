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
"""Tests for Bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

# Dependency imports
import numpy as np
import six

import tensorflow as tf

from tensorflow_probability.python import bijectors as tfb
tfe = tf.contrib.eager


@tfe.run_all_tests_in_graph_and_eager_modes
class BaseBijectorTest(tf.test.TestCase):
  """Tests properties of the Bijector base-class."""

  def testIsAbstract(self):
    with self.assertRaisesRegexp(TypeError,
                                 ("Can't instantiate abstract class Bijector "
                                  "with abstract methods __init__")):
      tfb.Bijector()  # pylint: disable=abstract-class-instantiated

  def testDefaults(self):

    class _BareBonesBijector(tfb.Bijector):
      """Minimal specification of a `Bijector`."""

      def __init__(self):
        super(_BareBonesBijector, self).__init__(forward_min_event_ndims=0)

    bij = _BareBonesBijector()
    self.assertEqual([], bij.graph_parents)
    self.assertEqual(False, bij.is_constant_jacobian)
    self.assertEqual(False, bij.validate_args)
    self.assertEqual(None, bij.dtype)
    self.assertEqual("bare_bones_bijector", bij.name)

    for shape in [[], [1, 2], [1, 2, 3]]:
      forward_event_shape_ = self.evaluate(
          bij.inverse_event_shape_tensor(shape))
      inverse_event_shape_ = self.evaluate(
          bij.forward_event_shape_tensor(shape))
      self.assertAllEqual(shape, forward_event_shape_)
      self.assertAllEqual(shape, bij.forward_event_shape(shape))
      self.assertAllEqual(shape, inverse_event_shape_)
      self.assertAllEqual(shape, bij.inverse_event_shape(shape))

    with self.assertRaisesRegexp(NotImplementedError,
                                 "inverse not implemented"):
      bij.inverse(0)

    with self.assertRaisesRegexp(NotImplementedError,
                                 "forward not implemented"):
      bij.forward(0)

    with self.assertRaisesRegexp(NotImplementedError,
                                 "inverse_log_det_jacobian not implemented"):
      bij.inverse_log_det_jacobian(0, event_ndims=0)

    with self.assertRaisesRegexp(NotImplementedError,
                                 "forward_log_det_jacobian not implemented"):
      bij.forward_log_det_jacobian(0, event_ndims=0)


class IntentionallyMissingError(Exception):
  pass


class BrokenBijector(tfb.Bijector):
  """Forward and inverse are not inverses of each other."""

  def __init__(self,
               forward_missing=False,
               inverse_missing=False,
               validate_args=False):
    super(BrokenBijector, self).__init__(
        validate_args=validate_args, forward_min_event_ndims=0, name="broken")
    self._forward_missing = forward_missing
    self._inverse_missing = inverse_missing

  def _forward(self, x):
    if self._forward_missing:
      raise IntentionallyMissingError
    return 2 * x

  def _inverse(self, y):
    if self._inverse_missing:
      raise IntentionallyMissingError
    return y / 2.

  def _inverse_log_det_jacobian(self, y):  # pylint:disable=unused-argument
    if self._inverse_missing:
      raise IntentionallyMissingError
    return -tf.log(2.)

  def _forward_log_det_jacobian(self, x):  # pylint:disable=unused-argument
    if self._forward_missing:
      raise IntentionallyMissingError
    return tf.log(2.)


@tfe.run_all_tests_in_graph_and_eager_modes
class BijectorTestEventNdims(tf.test.TestCase):

  def assertRaisesError(self, msg):
    return self.assertRaisesRegexp(Exception, msg)

  def testBijectorNonIntegerEventNdims(self):
    bij = BrokenBijector()
    with self.assertRaisesRegexp(ValueError, "Expected integer"):
      bij.forward_log_det_jacobian(1., event_ndims=1.5)
    with self.assertRaisesRegexp(ValueError, "Expected integer"):
      bij.inverse_log_det_jacobian(1., event_ndims=1.5)

  def testBijectorArrayEventNdims(self):
    bij = BrokenBijector()
    with self.assertRaisesRegexp(ValueError, "Expected scalar"):
      bij.forward_log_det_jacobian(1., event_ndims=(1, 2))
    with self.assertRaisesRegexp(ValueError, "Expected scalar"):
      bij.inverse_log_det_jacobian(1., event_ndims=(1, 2))

  def testBijectorDynamicEventNdims(self):
    with self.assertRaisesError("Expected scalar"):
      bij = BrokenBijector(validate_args=True)
      event_ndims = tf.placeholder_with_default((1, 2), shape=None)
      self.evaluate(
          bij.forward_log_det_jacobian(1., event_ndims=event_ndims))
    with self.assertRaisesError("Expected scalar"):
      bij = BrokenBijector(validate_args=True)
      event_ndims = tf.placeholder_with_default((1, 2), shape=None)
      self.evaluate(
          bij.inverse_log_det_jacobian(1., event_ndims=event_ndims))


@six.add_metaclass(abc.ABCMeta)
class BijectorCachingTestBase(object):

  @abc.abstractproperty
  def broken_bijector_cls(self):
    # return a BrokenBijector type Bijector, since this will test the caching.
    raise IntentionallyMissingError("Not implemented")

  def testCachingOfForwardResults(self):
    broken_bijector = self.broken_bijector_cls(inverse_missing=True)
    x = tf.constant(1.1)

    # Call forward and forward_log_det_jacobian one-by-one (not together).
    y = broken_bijector.forward(x)
    _ = broken_bijector.forward_log_det_jacobian(x, event_ndims=0)

    # Now, everything should be cached if the argument is y.
    broken_bijector.inverse_log_det_jacobian(y, event_ndims=0)
    try:
      broken_bijector.inverse(y)
      broken_bijector.inverse_log_det_jacobian(y, event_ndims=0)
    except IntentionallyMissingError:
      raise AssertionError("Tests failed! Cached values not used.")

    # Different event_ndims should not be cached.
    with self.assertRaises(IntentionallyMissingError):
      broken_bijector.inverse_log_det_jacobian(y, event_ndims=1)

  def testCachingOfInverseResults(self):
    broken_bijector = self.broken_bijector_cls(forward_missing=True)
    y = tf.constant(1.1)

    # Call inverse and inverse_log_det_jacobian one-by-one (not together).
    x = broken_bijector.inverse(y)
    _ = broken_bijector.inverse_log_det_jacobian(y, event_ndims=0)

    # Now, everything should be cached if the argument is x.
    try:
      broken_bijector.forward(x)
      broken_bijector.forward_log_det_jacobian(x, event_ndims=0)
    except IntentionallyMissingError:
      raise AssertionError("Tests failed! Cached values not used.")

    # Different event_ndims should not be cached.
    with self.assertRaises(IntentionallyMissingError):
      broken_bijector.forward_log_det_jacobian(x, event_ndims=1)


@tfe.run_all_tests_in_graph_and_eager_modes
class BijectorCachingTest(BijectorCachingTestBase, tf.test.TestCase):
  """Test caching with BrokenBijector."""

  @property
  def broken_bijector_cls(self):
    return BrokenBijector


class ExpOnlyJacobian(tfb.Bijector):
  """Only used for jacobian calculations."""

  def __init__(self, forward_min_event_ndims=0):
    super(ExpOnlyJacobian, self).__init__(
        validate_args=False,
        is_constant_jacobian=False,
        forward_min_event_ndims=forward_min_event_ndims,
        name="exp")

  def _inverse_log_det_jacobian(self, y):
    return -tf.log(y)

  def _forward_log_det_jacobian(self, x):
    return tf.log(x)


class ConstantJacobian(tfb.Bijector):
  """Only used for jacobian calculations."""

  def __init__(self, forward_min_event_ndims=0):
    super(ConstantJacobian, self).__init__(
        validate_args=False,
        is_constant_jacobian=True,
        forward_min_event_ndims=forward_min_event_ndims,
        name="c")

  def _inverse_log_det_jacobian(self, y):
    return tf.constant(2., y.dtype)

  def _forward_log_det_jacobian(self, x):
    return tf.constant(-2., x.dtype)


@tfe.run_all_tests_in_graph_and_eager_modes
class BijectorReduceEventDimsTest(tf.test.TestCase):
  """Test caching with BrokenBijector."""

  def testReduceEventNdimsForward(self):
    x = [[[1., 2.], [3., 4.]]]
    bij = ExpOnlyJacobian()
    self.assertAllClose(
        np.log(x), self.evaluate(
            bij.forward_log_det_jacobian(x, event_ndims=0)))
    self.assertAllClose(
        np.sum(np.log(x), axis=-1),
        self.evaluate(bij.forward_log_det_jacobian(x, event_ndims=1)))
    self.assertAllClose(
        np.sum(np.log(x), axis=(-1, -2)),
        self.evaluate(bij.forward_log_det_jacobian(x, event_ndims=2)))

  def testReduceEventNdimsForwardRaiseError(self):
    x = [[[1., 2.], [3., 4.]]]
    bij = ExpOnlyJacobian(forward_min_event_ndims=1)
    with self.assertRaisesRegexp(ValueError, "must be larger than"):
      bij.forward_log_det_jacobian(x, event_ndims=0)

  def testReduceEventNdimsInverse(self):
    x = [[[1., 2.], [3., 4.]]]
    bij = ExpOnlyJacobian()
    self.assertAllClose(
        -np.log(x), self.evaluate(
            bij.inverse_log_det_jacobian(x, event_ndims=0)))
    self.assertAllClose(
        np.sum(-np.log(x), axis=-1),
        self.evaluate(bij.inverse_log_det_jacobian(x, event_ndims=1)))
    self.assertAllClose(
        np.sum(-np.log(x), axis=(-1, -2)),
        self.evaluate(bij.inverse_log_det_jacobian(x, event_ndims=2)))

  def testReduceEventNdimsInverseRaiseError(self):
    x = [[[1., 2.], [3., 4.]]]
    bij = ExpOnlyJacobian(forward_min_event_ndims=1)
    with self.assertRaisesRegexp(ValueError, "must be larger than"):
      bij.inverse_log_det_jacobian(x, event_ndims=0)

  def testReduceEventNdimsForwardConstJacobian(self):
    x = [[[1., 2.], [3., 4.]]]
    bij = ConstantJacobian()
    self.assertAllClose(
        -2., self.evaluate(bij.forward_log_det_jacobian(x, event_ndims=0)))
    self.assertAllClose(
        -4., self.evaluate(bij.forward_log_det_jacobian(x, event_ndims=1)))
    self.assertAllClose(
        -8., self.evaluate(bij.forward_log_det_jacobian(x, event_ndims=2)))

  def testReduceEventNdimsInverseConstJacobian(self):
    x = [[[1., 2.], [3., 4.]]]
    bij = ConstantJacobian()
    self.assertAllClose(
        2., self.evaluate(bij.inverse_log_det_jacobian(x, event_ndims=0)))
    self.assertAllClose(
        4., self.evaluate(bij.inverse_log_det_jacobian(x, event_ndims=1)))
    self.assertAllClose(
        8., self.evaluate(bij.inverse_log_det_jacobian(x, event_ndims=2)))

  def testHandlesNonStaticEventNdims(self):
    x_ = [[[1., 2.], [3., 4.]]]
    x = tf.placeholder_with_default(x_, shape=None)
    event_ndims = tf.placeholder_with_default(1, shape=None)
    bij = ExpOnlyJacobian(forward_min_event_ndims=1)
    bij.inverse_log_det_jacobian(x, event_ndims=event_ndims)
    ildj = self.evaluate(
        bij.inverse_log_det_jacobian(x, event_ndims=event_ndims))
    self.assertAllClose(-np.log(x_), ildj)


if __name__ == "__main__":
  tf.test.main()
