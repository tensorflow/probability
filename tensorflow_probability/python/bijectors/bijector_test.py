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

import functools
import importlib
import weakref

# Dependency imports
import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top

tfb = tfp.bijectors
tfd = tfp.distributions


def try_import(name):
  try:
    return importlib.import_module(name)
  except ImportError as e:
    tf.compat.v1.logging.warning(
        "Could not import {}: {}.".format(name, str(e)))
    return None


mock = try_import("mock")


@test_util.run_all_in_graph_and_eager_modes
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

    with self.assertRaisesRegexp(
        NotImplementedError,
        "Neither _forward_log_det_jacobian nor _inverse_log_det_jacobian.*"):
      bij.inverse_log_det_jacobian(0, event_ndims=0)

    with self.assertRaisesRegexp(
        NotImplementedError,
        "Neither _forward_log_det_jacobian nor _inverse_log_det_jacobian.*"):
      bij.forward_log_det_jacobian(0, event_ndims=0)


class IntentionallyMissingError(Exception):
  pass


class ForwardOnlyBijector(tfb.Bijector):
  """Bijector with no inverse methods at all."""

  def __init__(self, validate_args=False):
    super(ForwardOnlyBijector, self).__init__(
        validate_args=validate_args,
        forward_min_event_ndims=0,
        name="forward_only")

  def _forward(self, x):
    return 2 * x

  def _forward_log_det_jacobian(self, _):
    return tf.math.log(2.)


class InverseOnlyBijector(tfb.Bijector):
  """Bijector with no forward methods at all."""

  def __init__(self, validate_args=False):
    super(InverseOnlyBijector, self).__init__(
        validate_args=validate_args,
        forward_min_event_ndims=0,
        name="inverse_only")

  def _inverse(self, y):
    return y / 2.

  def _inverse_log_det_jacobian(self, _):
    return -tf.math.log(2.)


class ExpOnlyJacobian(tfb.Bijector):
  """Only used for jacobian calculations."""

  def __init__(self, validate_args=False, forward_min_event_ndims=0):
    super(ExpOnlyJacobian, self).__init__(
        validate_args=validate_args,
        is_constant_jacobian=False,
        forward_min_event_ndims=forward_min_event_ndims,
        name="exp")

  def _inverse_log_det_jacobian(self, y):
    return -tf.math.log(y)

  def _forward_log_det_jacobian(self, x):
    return tf.math.log(x)


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


@test_util.run_all_in_graph_and_eager_modes
class BijectorTestEventNdims(tf.test.TestCase):

  def assertRaisesError(self, msg):
    return self.assertRaisesRegexp(Exception, msg)

  def testBijectorNonIntegerEventNdims(self):
    bij = ExpOnlyJacobian()
    with self.assertRaisesRegexp(ValueError, "Expected integer"):
      bij.forward_log_det_jacobian(1., event_ndims=1.5)
    with self.assertRaisesRegexp(ValueError, "Expected integer"):
      bij.inverse_log_det_jacobian(1., event_ndims=1.5)

  def testBijectorArrayEventNdims(self):
    bij = ExpOnlyJacobian()
    with self.assertRaisesRegexp(ValueError, "Expected scalar"):
      bij.forward_log_det_jacobian(1., event_ndims=(1, 2))
    with self.assertRaisesRegexp(ValueError, "Expected scalar"):
      bij.inverse_log_det_jacobian(1., event_ndims=(1, 2))

  def testBijectorDynamicEventNdims(self):
    with self.assertRaisesError("Expected scalar"):
      bij = ExpOnlyJacobian(validate_args=True)
      event_ndims = tf.compat.v1.placeholder_with_default((1, 2), shape=None)
      self.evaluate(
          bij.forward_log_det_jacobian(1., event_ndims=event_ndims))
    with self.assertRaisesError("Expected scalar"):
      bij = ExpOnlyJacobian(validate_args=True)
      event_ndims = tf.compat.v1.placeholder_with_default((1, 2), shape=None)
      self.evaluate(
          bij.inverse_log_det_jacobian(1., event_ndims=event_ndims))


@test_util.run_all_in_graph_and_eager_modes
class BijectorCachingTest(tf.test.TestCase):

  def testCachingOfForwardResults(self):
    forward_only_bijector = ForwardOnlyBijector()
    x = tf.constant(1.1)
    y = tf.constant(2.2)

    with self.assertRaises(NotImplementedError):
      forward_only_bijector.inverse(y)

    with self.assertRaises(NotImplementedError):
      forward_only_bijector.inverse_log_det_jacobian(y, event_ndims=0)

    # Call forward and forward_log_det_jacobian one-by-one (not together).
    y = forward_only_bijector.forward(x)
    _ = forward_only_bijector.forward_log_det_jacobian(x, event_ndims=0)
    if tf.executing_eagerly():
      self.assertIsNot(y, forward_only_bijector.forward(x))
    else:
      self.assertIs(y, forward_only_bijector.forward(x))

    # Now, everything should be cached if the argument is y, so these are ok.
    forward_only_bijector.inverse(y)
    forward_only_bijector.inverse_log_det_jacobian(y, event_ndims=0)

  def testCachingOfInverseResults(self):
    inverse_only_bijector = InverseOnlyBijector()
    x = tf.constant(1.1)
    y = tf.constant(2.2)

    with self.assertRaises(NotImplementedError):
      inverse_only_bijector.forward(x)

    with self.assertRaises(NotImplementedError):
      inverse_only_bijector.forward_log_det_jacobian(x, event_ndims=0)

    # Call inverse and inverse_log_det_jacobian one-by-one (not together).
    x = inverse_only_bijector.inverse(y)
    _ = inverse_only_bijector.inverse_log_det_jacobian(y, event_ndims=0)
    if tf.executing_eagerly():
      self.assertIsNot(x, inverse_only_bijector.inverse(y))
    else:
      self.assertIs(x, inverse_only_bijector.inverse(y))

    # Now, everything should be cached if the argument is x.
    inverse_only_bijector.forward(x)
    inverse_only_bijector.forward_log_det_jacobian(x, event_ndims=0)

  def testCachingGarbageCollection(self):
    bijector = ForwardOnlyBijector()
    refs = []
    niters = 3
    for _ in range(niters):
      y = bijector.forward(tf.zeros([10]))
      refs.append(weakref.ref(y))

    # We tolerate leaking tensor references in graph mode only.
    expected_live = 1 if tf.executing_eagerly() else niters
    self.assertEqual(expected_live, sum(ref() is not None for ref in refs))


@test_util.run_all_in_graph_and_eager_modes
class BijectorReduceEventDimsTest(tf.test.TestCase):
  """Test reducing of event dims."""

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
    x = tf.compat.v1.placeholder_with_default(x_, shape=None)
    event_ndims = tf.compat.v1.placeholder_with_default(1, shape=None)
    bij = ExpOnlyJacobian(forward_min_event_ndims=1)
    bij.inverse_log_det_jacobian(x, event_ndims=event_ndims)
    ildj = self.evaluate(
        bij.inverse_log_det_jacobian(x, event_ndims=event_ndims))
    self.assertAllClose(-np.log(x_), ildj)


@test_util.run_all_in_graph_and_eager_modes
class BijectorCompositionTest(tf.test.TestCase):

  def testComposeFromChainBijector(self):
    x = tf.constant([-5., 0., 5.])
    sigmoid = functools.reduce(lambda chain, f: chain(f), [
        tfb.Reciprocal(),
        tfb.AffineScalar(shift=1.),
        tfb.Exp(),
        tfb.AffineScalar(scale=-1.),
    ])
    self.assertTrue(isinstance(sigmoid, tfb.Chain))
    self.assertAllClose(
        *self.evaluate([tf.nn.sigmoid(x), sigmoid.forward(x)]),
        atol=0, rtol=1e-3)

  def testComposeFromTransformedDistribution(self):
    actual_log_normal = tfb.Exp()(tfd.TransformedDistribution(
        distribution=tfd.Normal(0, 1),
        bijector=tfb.AffineScalar(shift=0.5, scale=2.)))
    expected_log_normal = tfd.LogNormal(0.5, 2.)
    x = tf.constant([0.1, 1., 5.])
    self.assertAllClose(
        *self.evaluate([actual_log_normal.log_prob(x),
                        expected_log_normal.log_prob(x)]),
        atol=0, rtol=1e-3)

  def testComposeFromTDSubclassWithAlternateCtorArgs(self):
    # This line used to raise an exception.
    tfb.Identity()(tfd.Chi(df=1., allow_nan_stats=True))

  def testComposeFromNonTransformedDistribution(self):
    actual_log_normal = tfb.Exp()(tfd.Normal(0.5, 2.))
    expected_log_normal = tfd.LogNormal(0.5, 2.)
    x = tf.constant([0.1, 1., 5.])
    self.assertAllClose(
        *self.evaluate([actual_log_normal.log_prob(x),
                        expected_log_normal.log_prob(x)]),
        atol=0, rtol=1e-3)

  def testComposeFromTensor(self):
    x = tf.constant([-5., 0., 5.])
    self.assertAllClose(
        *self.evaluate([tf.exp(x), tfb.Exp()(x)]),
        atol=0, rtol=1e-3)

  def testHandlesKwargs(self):
    x = tfb.Exp()(tfd.Normal(0, 1), event_shape=[4])
    y = tfd.Independent(tfd.LogNormal(tf.zeros(4), 1), 1)
    z = tf.constant([[1., 2, 3, 4],
                     [0.5, 1.5, 2., 2.5]])
    self.assertAllClose(
        *self.evaluate([y.log_prob(z), x.log_prob(z)]),
        atol=0, rtol=1e-3)


class BijectorLDJCachingTest(tf.test.TestCase):

  def testShapeCachingIssue(self):
    if tf.executing_eagerly(): return
    # Exercise the scenario outlined in
    # https://github.com/tensorflow/probability/issues/253 (originally reported
    # internally as b/119756336).
    x1 = tf.compat.v1.placeholder(tf.float32, shape=[None, 2], name="x1")
    x2 = tf.compat.v1.placeholder(tf.float32, shape=[None, 2], name="x2")

    bij = ConstantJacobian()

    bij.forward_log_det_jacobian(x2, event_ndims=1)
    a = bij.forward_log_det_jacobian(x1, event_ndims=1, name="a_fldj")

    x1_value = np.random.uniform(size=[10, 2])
    with self.test_session() as sess:
      sess.run(a, feed_dict={x1: x1_value})


@test_util.run_all_in_graph_and_eager_modes
class BijectorConstILDJLeakTest(tf.test.TestCase):

  # See discussion in b/129297998.
  def testConstILDJLeak(self):
    bij = ConstantJacobian()

    call_fldj = lambda: bij.forward_log_det_jacobian(0., event_ndims=0)

    @tf.function
    def func1():
      call_fldj()
      return call_fldj()

    @tf.function
    def func2():
      call_fldj()
      return call_fldj()

    func1()
    self.assertLen(bij._constant_ildj, 1)
    func2()
    self.assertLen(bij._constant_ildj, 2)

    call_fldj()
    call_fldj()
    self.assertLen(bij._constant_ildj, 3)


@test_util.run_all_in_graph_and_eager_modes
class NumpyArrayCaching(tf.test.TestCase):

  def test_caches(self):
    if mock is None:
      return

    x_ = np.array([[-0.1, 0.2], [0.3, -0.4]], np.float32)
    y_ = np.exp(x_)
    b = tfb.Exp()

    # We will intercept calls to TF to ensure np.array objects don't get
    # converted to tf.Tensor objects.

    with mock.patch.object(tf, "convert_to_tensor", return_value=x_):
      with mock.patch.object(tf.compat.v2, "exp", return_value=y_):
        y = b.forward(x_)
        self.assertIsInstance(y, np.ndarray)
        self.assertAllEqual([x_], [k() for k in b._from_x.keys()])

    with mock.patch.object(tf, "convert_to_tensor", return_value=y_):
      with mock.patch.object(tf.compat.v2.math, "log", return_value=x_):
        x = b.inverse(y_)
        self.assertIsInstance(x, np.ndarray)
        self.assertIs(x, b.inverse(y))
        self.assertAllEqual([y_], [k() for k in b._from_y.keys()])

    yt_ = y_.T
    xt_ = x_.T
    with mock.patch.object(tf, "convert_to_tensor", return_value=yt_):
      with mock.patch.object(tf.compat.v2.math, "log", return_value=xt_):
        xt = b.inverse(yt_)
        self.assertIsNot(x, xt)
        self.assertIs(xt_, xt)


if __name__ == "__main__":
  tf.test.main()
