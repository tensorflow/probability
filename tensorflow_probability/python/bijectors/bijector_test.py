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

# Dependency imports
import mock
import numpy as np

import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.internal import test_util


@test_util.test_all_tf_execution_regimes
class BaseBijectorTest(test_util.TestCase):
  """Tests properties of the Bijector base-class."""

  def testIsAbstract(self):
    with self.assertRaisesRegexp(TypeError,
                                 ('Can\'t instantiate abstract class Bijector '
                                  'with abstract methods __init__')):
      tfb.Bijector()  # pylint: disable=abstract-class-instantiated

  def testDefaults(self):

    class _BareBonesBijector(tfb.Bijector):
      """Minimal specification of a `Bijector`."""

      def __init__(self):
        super(_BareBonesBijector, self).__init__(forward_min_event_ndims=0)

    bij = _BareBonesBijector()
    self.assertEqual(False, bij.is_constant_jacobian)
    self.assertEqual(False, bij.validate_args)
    self.assertEqual(None, bij.dtype)
    self.assertStartsWith(bij.name, 'bare_bones_bijector')

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
                                 'inverse not implemented'):
      bij.inverse(0)

    with self.assertRaisesRegexp(NotImplementedError,
                                 'forward not implemented'):
      bij.forward(0)

    with self.assertRaisesRegexp(
        NotImplementedError,
        'Neither _forward_log_det_jacobian nor _inverse_log_det_jacobian.*'):
      bij.inverse_log_det_jacobian(0, event_ndims=0)

    with self.assertRaisesRegexp(
        NotImplementedError,
        'Neither _forward_log_det_jacobian nor _inverse_log_det_jacobian.*'):
      bij.forward_log_det_jacobian(0, event_ndims=0)


class IntentionallyMissingError(Exception):
  pass


class ForwardOnlyBijector(tfb.Bijector):
  """Bijector with no inverse methods at all."""

  def __init__(self, scale=2, validate_args=False, name=None):
    with tf.name_scope(name or 'forward_only') as name:
      self._scale = tensor_util.convert_nonref_to_tensor(
          scale,
          dtype_hint=tf.float32)
      super(ForwardOnlyBijector, self).__init__(
          validate_args=validate_args,
          forward_min_event_ndims=0,
          name=name)

  def _forward(self, x):
    return self._scale * x

  def _forward_log_det_jacobian(self, _):
    return tf.math.log(self._scale)


class InverseOnlyBijector(tfb.Bijector):
  """Bijector with no forward methods at all."""

  def __init__(self, scale=2., validate_args=False, name=None):
    with tf.name_scope(name or 'inverse_only') as name:
      self._scale = tensor_util.convert_nonref_to_tensor(
          scale,
          dtype_hint=tf.float32)
      super(InverseOnlyBijector, self).__init__(
          validate_args=validate_args,
          forward_min_event_ndims=0,
          name=name)

  def _inverse(self, y):
    return y / self._scale

  def _inverse_log_det_jacobian(self, _):
    return -tf.math.log(self._scale)


class ExpOnlyJacobian(tfb.Bijector):
  """Only used for jacobian calculations."""

  def __init__(self, validate_args=False, forward_min_event_ndims=0):
    super(ExpOnlyJacobian, self).__init__(
        validate_args=validate_args,
        is_constant_jacobian=False,
        forward_min_event_ndims=forward_min_event_ndims,
        name='exp')

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
        name='c')

  def _inverse_log_det_jacobian(self, y):
    return tf.constant(2., y.dtype)

  def _forward_log_det_jacobian(self, x):
    return tf.constant(-2., x.dtype)


@test_util.test_all_tf_execution_regimes
class BijectorTestEventNdims(test_util.TestCase):

  def assertRaisesError(self, msg):
    return self.assertRaisesRegexp(Exception, msg)

  def testBijectorNonIntegerEventNdims(self):
    bij = ExpOnlyJacobian()
    with self.assertRaisesRegexp(ValueError, 'Expected integer'):
      bij.forward_log_det_jacobian(1., event_ndims=1.5)
    with self.assertRaisesRegexp(ValueError, 'Expected integer'):
      bij.inverse_log_det_jacobian(1., event_ndims=1.5)

  def testBijectorArrayEventNdims(self):
    bij = ExpOnlyJacobian()
    with self.assertRaisesRegexp(ValueError, 'Expected scalar'):
      bij.forward_log_det_jacobian(1., event_ndims=(1, 2))
    with self.assertRaisesRegexp(ValueError, 'Expected scalar'):
      bij.inverse_log_det_jacobian(1., event_ndims=(1, 2))

  def testBijectorDynamicEventNdims(self):
    with self.assertRaisesError('Expected scalar'):
      bij = ExpOnlyJacobian(validate_args=True)
      event_ndims = tf1.placeholder_with_default((1, 2), shape=None)
      self.evaluate(
          bij.forward_log_det_jacobian(1., event_ndims=event_ndims))
    with self.assertRaisesError('Expected scalar'):
      bij = ExpOnlyJacobian(validate_args=True)
      event_ndims = tf1.placeholder_with_default((1, 2), shape=None)
      self.evaluate(
          bij.inverse_log_det_jacobian(1., event_ndims=event_ndims))


@test_util.test_all_tf_execution_regimes
class BijectorCachingTest(test_util.TestCase):

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
    self.assertIs(y, forward_only_bijector.forward(x))
    # Now, everything should be cached if the argument `is y`, so these are ok.
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
    self.assertIs(x, inverse_only_bijector.inverse(y))

    # Now, everything should be cached if the argument `is x`.
    inverse_only_bijector.forward(x)
    inverse_only_bijector.forward_log_det_jacobian(x, event_ndims=0)

  def testCachingGarbageCollection(self):
    bijector = ForwardOnlyBijector()
    niters = 6
    for i in range(niters):
      x = tf.constant(i, dtype=tf.float32)
      y = bijector.forward(x)  # pylint: disable=unused-variable

    # We tolerate leaking tensor references in graph mode only.
    expected_live = 1 if tf.executing_eagerly() else niters
    self.assertEqual(expected_live, len(bijector._cache.forward))


@test_util.test_all_tf_execution_regimes
class BijectorReduceEventDimsTest(test_util.TestCase):
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
    with self.assertRaisesRegexp(ValueError, 'must be larger than'):
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
    with self.assertRaisesRegexp(ValueError, 'must be larger than'):
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
    x = tf1.placeholder_with_default(x_, shape=None)
    event_ndims = tf1.placeholder_with_default(1, shape=None)
    bij = ExpOnlyJacobian(forward_min_event_ndims=1)
    bij.inverse_log_det_jacobian(x, event_ndims=event_ndims)
    ildj = self.evaluate(
        bij.inverse_log_det_jacobian(x, event_ndims=event_ndims))
    self.assertAllClose(-np.log(x_), ildj)


class BijectorLDJCachingTest(test_util.TestCase):

  def testShapeCachingIssue(self):
    if tf.executing_eagerly(): return
    # Exercise the scenario outlined in
    # https://github.com/tensorflow/probability/issues/253 (originally reported
    # internally as b/119756336).
    x1 = tf1.placeholder(tf.float32, shape=[None, 2], name='x1')
    x2 = tf1.placeholder(tf.float32, shape=[None, 2], name='x2')

    bij = ConstantJacobian()

    bij.forward_log_det_jacobian(x2, event_ndims=1)
    a = bij.forward_log_det_jacobian(x1, event_ndims=1, name='a_fldj')

    x1_value = np.random.uniform(size=[10, 2])
    with self.test_session() as sess:
      sess.run(a, feed_dict={x1: x1_value})


@test_util.test_all_tf_execution_regimes
class NumpyArrayCaching(test_util.TestCase):

  def test_caches(self):
    # We need to call convert_to_tensor on outputs to make sure scalar
    # outputs from the numpy backend are wrapped correctly. We could just
    # directly wrap numpy scalars with np.array, but it would look pretty
    # out of place, considering that the numpy backend is still private.
    if mock is None:
      return

    x_ = np.array([[-0.1, 0.2], [0.3, -0.4]], np.float32)
    y_ = np.exp(x_)
    b = tfb.Exp()

    # We will intercept calls to TF to ensure np.array objects don't get
    # converted to tf.Tensor objects.

    with mock.patch.object(tf, 'convert_to_tensor', return_value=x_):
      with mock.patch.object(tf, 'exp', return_value=y_):
        y = b.forward(x_)
        self.assertIsInstance(y, np.ndarray)
        self.assertAllEqual([x_],
                            [k() for k in b._cache.forward.weak_keys()])

    with mock.patch.object(tf, 'convert_to_tensor', return_value=y_):
      with mock.patch.object(tf.math, 'log', return_value=x_):
        x = b.inverse(y_)
        self.assertIsInstance(x, np.ndarray)
        self.assertIs(x, b.inverse(y))
        self.assertAllEqual([y_],
                            [k() for k in b._cache.inverse.weak_keys()])

    yt_ = y_.T
    xt_ = x_.T
    with mock.patch.object(tf, 'convert_to_tensor', return_value=yt_):
      with mock.patch.object(tf.math, 'log', return_value=xt_):
        xt = b.inverse(yt_)
        self.assertIsNot(x, xt)
        self.assertIs(xt_, xt)


@test_util.test_all_tf_execution_regimes
class TfModuleTest(test_util.TestCase):

  @test_util.jax_disable_variable_test
  def test_variable_tracking(self):
    x = tf.Variable(1.)
    b = ForwardOnlyBijector(scale=x, validate_args=True)
    self.assertIsInstance(b, tf.Module)
    self.assertEqual((x,), b.trainable_variables)

  @test_util.jax_disable_variable_test
  def test_gradient(self):
    x = tf.Variable(1.)
    b = InverseOnlyBijector(scale=x, validate_args=True)
    with tf.GradientTape() as tape:
      loss = b.inverse(1.)
    g = tape.gradient(loss, b.trainable_variables)
    self.evaluate(tf1.global_variables_initializer())
    self.assertEqual((-1.,), self.evaluate(g))


class _ConditionalBijector(tfb.Bijector):

  def __init__(self):
    super(_ConditionalBijector, self).__init__(
        forward_min_event_ndims=0,
        is_constant_jacobian=True,
        validate_args=False,
        dtype=tf.float32,
        name='test_bijector')

  # These are not implemented in the base class, but we need to write a stub in
  # order to mock them out.
  def _inverse_log_det_jacobian(self, _, arg1, arg2):
    pass

  def _forward_log_det_jacobian(self, _, arg1, arg2):
    pass


# Test that ensures kwargs from public methods are passed in to
# private methods.
@test_util.test_all_tf_execution_regimes
class ConditionalBijectorTest(test_util.TestCase):

  def testConditionalBijector(self):
    b = _ConditionalBijector()
    arg1 = 'b1'
    arg2 = 'b2'
    retval = tf.constant(1.)
    for name in ['forward', 'inverse']:
      method = getattr(b, name)
      with mock.patch.object(b, '_' + name, return_value=retval) as mock_method:
        method(1., arg1=arg1, arg2=arg2)
      mock_method.assert_called_once_with(mock.ANY, arg1=arg1, arg2=arg2)

    for name in ['inverse_log_det_jacobian', 'forward_log_det_jacobian']:
      method = getattr(b, name)
      with mock.patch.object(b, '_' + name, return_value=retval) as mock_method:
        method(1., event_ndims=0, arg1=arg1, arg2=arg2)
      mock_method.assert_called_once_with(mock.ANY, arg1=arg1, arg2=arg2)

  def testNestedCondition(self):
    b = _ConditionalBijector()
    arg1 = {'b1': 'c1'}
    arg2 = {'b2': 'c2'}
    retval = tf.constant(1.)
    for name in ['forward', 'inverse']:
      method = getattr(b, name)
      with mock.patch.object(b, '_' + name, return_value=retval) as mock_method:
        method(1., arg1=arg1, arg2=arg2)
      mock_method.assert_called_once_with(mock.ANY, arg1=arg1, arg2=arg2)

    for name in ['inverse_log_det_jacobian', 'forward_log_det_jacobian']:
      method = getattr(b, name)
      with mock.patch.object(b, '_' + name, return_value=retval) as mock_method:
        method(1., event_ndims=0, arg1=arg1, arg2=arg2)
      mock_method.assert_called_once_with(mock.ANY, arg1=arg1, arg2=arg2)


if __name__ == '__main__':
  tf.test.main()
