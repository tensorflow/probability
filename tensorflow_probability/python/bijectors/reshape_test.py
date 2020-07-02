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
"""Tests for Reshape Bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import numpy as np
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python.bijectors import bijector_test_util
from tensorflow_probability.python.internal import hypothesis_testlib as tfp_hps
from tensorflow_probability.python.internal import test_util


class _ReshapeBijectorTest(object):
  """Base class for testing the reshape transformation.

  Methods defined in this class call a method self.build_shapes() that
  is implemented by subclasses defined below, returning respectively
   ReshapeBijectorTestStatic: static shapes,
   ReshapeBijectorTestDynamic: shape placeholders of known ndims, and
  so that each test in this base class is automatically run over all
  three cases. The subclasses also implement assertRaisesError to test
  for either Python exceptions (in the case of static shapes) or
  TensorFlow op errors (dynamic shapes).
  """

  def testBijector(self):
    """Do a basic sanity check of forward, inverse, jacobian."""
    expected_x = np.random.randn(4, 3, 2)
    expected_y = np.reshape(expected_x, [4, 6])

    shape_in, shape_out = self.build_shapes([3, 2], [6,])
    bijector = tfb.Reshape(
        event_shape_out=shape_out,
        event_shape_in=shape_in,
        validate_args=True)
    [
        x_,
        y_,
        fldj_,
        ildj_,
        fest_,
        iest_,
    ] = self.evaluate([
        bijector.inverse(expected_y),
        bijector.forward(expected_x),
        bijector.forward_log_det_jacobian(expected_x, event_ndims=2),
        bijector.inverse_log_det_jacobian(expected_y, event_ndims=2),
        bijector.forward_event_shape_tensor(expected_x.shape),
        bijector.inverse_event_shape_tensor(expected_y.shape),
    ])
    self.assertStartsWith(bijector.name, 'reshape')
    self.assertAllClose(expected_y, y_, rtol=1e-6, atol=0)
    self.assertAllClose(expected_x, x_, rtol=1e-6, atol=0)
    self.assertAllClose(0., fldj_, rtol=1e-6, atol=0)
    self.assertAllClose(0., ildj_, rtol=1e-6, atol=0)
    # Test that event_shape_tensors match fwd/inv result shapes.
    self.assertAllEqual(y_.shape, fest_)
    self.assertAllEqual(x_.shape, iest_)

  def testEventShapeTensor(self):
    """Test event_shape_tensor methods when even ndims may be dynamic."""

    shape_in_static = [2, 3]
    shape_out_static = [6,]
    shape_in, shape_out = self.build_shapes(shape_in_static, shape_out_static)
    bijector = tfb.Reshape(
        event_shape_out=shape_out, event_shape_in=shape_in, validate_args=True)

    # using the _tensor methods, we should always get a fully-specified
    # result since these are evaluated at graph runtime.
    (shape_out_,
     shape_in_) = self.evaluate((
         bijector.forward_event_shape_tensor(shape_in),
         bijector.inverse_event_shape_tensor(shape_out),
     ))
    self.assertAllEqual(shape_out_static, shape_out_)
    self.assertAllEqual(shape_in_static, shape_in_)

  def testScalarReshape(self):
    """Test reshaping to and from a scalar shape ()."""

    expected_x = np.random.randn(4, 3, 1)
    expected_y = np.reshape(expected_x, [4, 3])

    expected_x_scalar = np.random.randn(1,)
    expected_y_scalar = expected_x_scalar[0]

    shape_in, shape_out = self.build_shapes([], [1,])
    bijector = tfb.Reshape(
        event_shape_out=shape_in,
        event_shape_in=shape_out,
        validate_args=True)
    (x_,
     y_,
     x_scalar_,
     y_scalar_
    ) = self.evaluate((
        bijector.inverse(expected_y),
        bijector.forward(expected_x),
        bijector.inverse(expected_y_scalar),
        bijector.forward(expected_x_scalar),
    ))
    self.assertAllClose(expected_y, y_, rtol=1e-6, atol=0)
    self.assertAllClose(expected_x, x_, rtol=1e-6, atol=0)
    self.assertAllClose(expected_y_scalar, y_scalar_, rtol=1e-6, atol=0)
    self.assertAllClose(expected_x_scalar, x_scalar_, rtol=1e-6, atol=0)

  def testValidButNonMatchingInputOpError(self):
    x = np.random.randn(4, 3, 2)

    shape_in, shape_out = self.build_shapes([2, 3], [1, 6, 1,])
    bijector = tfb.Reshape(
        event_shape_out=shape_out,
        event_shape_in=shape_in,
        validate_args=True)

    # Here we pass in a tensor (x) whose shape is compatible with
    # the output shape, so tf.reshape will throw no error, but
    # doesn't match the expected input shape.
    with self.assertRaisesError('Input `event_shape` does not match'):
      self.evaluate(bijector.forward(x))

  def testValidButNonMatchingInputPartiallySpecifiedOpError(self):
    x = np.random.randn(4, 3, 2)

    shape_in, shape_out = self.build_shapes([2, -1], [1, 6, 1,])
    bijector = tfb.Reshape(
        event_shape_out=shape_out,
        event_shape_in=shape_in,
        validate_args=True)

    with self.assertRaisesError('Input `event_shape` does not match'):
      self.evaluate(bijector.forward(x))

  # pylint: disable=invalid-name
  def _testInputOutputMismatchOpError(self, expected_error_message):
    x1 = np.random.randn(4, 2, 3)
    x2 = np.random.randn(4, 1, 1, 5)

    shape_in, shape_out = self.build_shapes([2, 3], [1, 1, 5])
    with self.assertRaisesError(expected_error_message):
      bijector = tfb.Reshape(
          event_shape_out=shape_out,
          event_shape_in=shape_in,
          validate_args=True)
      self.evaluate(bijector.forward(x1))
    with self.assertRaisesError(expected_error_message):
      bijector = tfb.Reshape(
          event_shape_out=shape_out,
          event_shape_in=shape_in,
          validate_args=True)
      self.evaluate(bijector.inverse(x2))
  # pylint: enable=invalid-name

  def testOneShapePartiallySpecified(self):
    expected_x = np.random.randn(4, 6)
    expected_y = np.reshape(expected_x, [4, 2, 3])

    # one of input/output shapes is partially specified
    shape_in, shape_out = self.build_shapes([-1,], [2, 3])
    bijector = tfb.Reshape(
        event_shape_out=shape_out,
        event_shape_in=shape_in,
        validate_args=True)
    x_, y_, = self.evaluate([
        bijector.inverse(expected_y),
        bijector.forward(expected_x),
    ])
    self.assertAllClose(expected_y, y_, rtol=1e-6, atol=0)
    self.assertAllClose(expected_x, x_, rtol=1e-6, atol=0)

  def testBothShapesPartiallySpecified(self):
    expected_x = np.random.randn(4, 2, 3)
    expected_y = np.reshape(expected_x, [4, 3, 2])
    shape_in, shape_out = self.build_shapes([-1, 3], [-1, 2])
    bijector = tfb.Reshape(
        event_shape_out=shape_out,
        event_shape_in=shape_in,
        validate_args=True)
    x_, y_, = self.evaluate([
        bijector.inverse(expected_y),
        bijector.forward(expected_x),
    ])
    self.assertAllClose(expected_y, y_, rtol=1e-6, atol=0)
    self.assertAllClose(expected_x, x_, rtol=1e-6, atol=0)

  def testDefaultVectorShape(self):
    expected_x = np.random.randn(4, 4)
    expected_y = np.reshape(expected_x, [4, 2, 2])
    _, shape_out = self.build_shapes([-1,], [-1, 2])
    bijector = tfb.Reshape(shape_out, validate_args=True)
    x_, y_, = self.evaluate([
        bijector.inverse(expected_y),
        bijector.forward(expected_x),
    ])
    self.assertAllClose(expected_y, y_, rtol=1e-6, atol=0)
    self.assertAllClose(expected_x, x_, rtol=1e-6, atol=0)

  def build_shapes(self, *args, **kwargs):
    raise NotImplementedError('Subclass failed to implement `build_shapes`.')


@test_util.test_all_tf_execution_regimes
class ReshapeBijectorTestStatic(test_util.TestCase, _ReshapeBijectorTest):

  def build_shapes(self, shape_in, shape_out):
    return shape_in, shape_out

  def assertRaisesError(self, msg):
    return self.assertRaisesRegexp(Exception, msg)

  def testEventShape(self):
    shape_in_static = tf.TensorShape([2, 3])
    shape_out_static = tf.TensorShape([6])
    bijector = tfb.Reshape(
        event_shape_out=shape_out_static,
        event_shape_in=shape_in_static,
        validate_args=True)

    # Test that forward_ and inverse_event_shape are correct when
    # event_shape_in/_out are statically known, even when the input shapes
    # are only partially specified.
    self.assertEqual(
        bijector.forward_event_shape(tf.TensorShape([4, 2, 3])).as_list(),
        [4, 6])
    self.assertEqual(
        bijector.inverse_event_shape(tf.TensorShape([4, 6])).as_list(),
        [4, 2, 3])

    # Shape is always known for reshaping in eager mode, so we skip these tests.
    if tf.executing_eagerly():
      return
    self.assertEqual(
        bijector.forward_event_shape(tf.TensorShape([None, 2, 3])).as_list(),
        [None, 6])
    self.assertEqual(
        bijector.inverse_event_shape(tf.TensorShape([None, 6])).as_list(),
        [None, 2, 3])
    # If the input shape is totally unknown, there's nothing we can do!
    self.assertIsNone(
        bijector.forward_event_shape(tf.TensorShape(None)).ndims)

  def testBijectiveAndFinite(self):
    x = np.random.randn(4, 2, 3)
    y = np.reshape(x, [4, 1, 2, 3])
    bijector = tfb.Reshape(
        event_shape_in=[2, 3], event_shape_out=[1, 2, 3], validate_args=True)
    bijector_test_util.assert_bijective_and_finite(
        bijector,
        x,
        y,
        eval_func=self.evaluate,
        event_ndims=2,
        inverse_event_ndims=3,
        rtol=1e-6,
        atol=0)

  def testWorksWithChain(self):
    shape_out = (4,)
    shape_in = (2, 2)
    x = np.zeros(shape_in)
    y = np.zeros(shape_out)
    bijector = tfb.Chain([
        tfb.Identity(),
        tfb.Reshape(event_shape_out=shape_out, event_shape_in=shape_in)
    ])
    new_y = self.evaluate(bijector.forward(x))
    new_x = self.evaluate(bijector.inverse(y))
    fldj = self.evaluate(
        bijector.forward_log_det_jacobian(x, event_ndims=len(shape_in)))
    ildj = self.evaluate(
        bijector.inverse_log_det_jacobian(y, event_ndims=len(shape_out)))
    self.assertEqual(shape_out, new_y.shape)
    self.assertEqual(shape_in, new_x.shape)
    self.assertEqual((), fldj.shape)
    self.assertEqual((), ildj.shape)

  def testMultipleUnspecifiedDimensionsOpError(self):
    shape_in, shape_out = self.build_shapes([2, 3], [4, -1, -1,])

    with self.assertRaises(ValueError):
      tfb.Reshape(event_shape_out=shape_out,
                  event_shape_in=shape_in,
                  validate_args=True)

  def testInvalidDimensionsOpError(self):
    shape_in, shape_out = self.build_shapes([2, 3], [1, 2, -2,])

    with self.assertRaises(ValueError):
      tfb.Reshape(event_shape_out=shape_out,
                  event_shape_in=shape_in,
                  validate_args=True)

  def testInputOutputMismatchOpError(self):
    self._testInputOutputMismatchOpError('reshape')

  def testCheckingVariableShape(self):
    shape_out = tf.Variable([-2, 10])
    self.evaluate(shape_out.initializer)
    with self.assertRaisesOpError(
        'elements must be either positive integers or `-1`'):
      self.evaluate(tfb.Reshape(shape_out, validate_args=True).forward([0]))

  def testCheckingMutatedVariableShape(self):
    shape_out = tf.Variable([1, 1])
    self.evaluate(shape_out.initializer)
    reshape = tfb.Reshape(shape_out, validate_args=True)
    self.evaluate(reshape.forward([0]))
    with self.assertRaisesOpError(
        'elements must be either positive integers or `-1`'):
      with tf.control_dependencies([shape_out.assign([-2, 10])]):
        self.evaluate(reshape.forward([0]))

  @test_util.numpy_disable_test_missing_functionality('b/142265598')
  def testConcretizationLimits(self):
    shape_out = tfp_hps.defer_and_count_usage(tf.Variable([1]))
    reshape = tfb.Reshape(shape_out, validate_args=True)
    x = [1]  # Pun: valid input or output, and valid input or output shape
    for method in ['forward', 'inverse', 'forward_event_shape',
                   'inverse_event_shape', 'forward_event_shape_tensor',
                   'inverse_event_shape_tensor']:
      with tfp_hps.assert_no_excessive_var_usage(method, max_permissible=7):
        getattr(reshape, method)(x)
    for method in ['forward_log_det_jacobian', 'inverse_log_det_jacobian']:
      with tfp_hps.assert_no_excessive_var_usage(method, max_permissible=4):
        getattr(reshape, method)(x, event_ndims=1)


class ReshapeBijectorTestDynamic(test_util.TestCase, _ReshapeBijectorTest):

  def build_shapes(self, shape_in, shape_out):
    shape_in = np.array(shape_in, np.int32)
    shape_out = np.array(shape_out, np.int32)
    return (
        tf1.placeholder_with_default(
            shape_in, shape=[len(shape_in)]),
        tf1.placeholder_with_default(
            shape_out, shape=[len(shape_out)]),
    )

  def assertRaisesError(self, msg):
    if tf.executing_eagerly():
      return self.assertRaisesRegexp(Exception, msg)
    return self.assertRaisesOpError(msg)

  def testEventShape(self):
    # Shape is always known for reshaping in eager mode, so we skip these tests.
    if tf.executing_eagerly(): return

    event_shape_in, event_shape_out = self.build_shapes([2, 3], [6])
    bijector = tfb.Reshape(
        event_shape_out=event_shape_out,
        event_shape_in=event_shape_in,
        validate_args=True)

    self.assertEqual(
        bijector.forward_event_shape(tf.TensorShape([4, 2, 3])).as_list(),
        [4, None])
    self.assertEqual(
        bijector.forward_event_shape(tf.TensorShape([None, 2, 3])).as_list(),
        [None, None])
    self.assertEqual(
        bijector.inverse_event_shape(tf.TensorShape([4, 6])).as_list(),
        [4, None, None])
    self.assertEqual(
        bijector.inverse_event_shape(tf.TensorShape([None, 6])).as_list(),
        [None, None, None])
    # If the input shape is totally unknown, there's nothing we can do!
    self.assertIsNone(
        bijector.forward_event_shape(tf.TensorShape(None)).ndims)

  def testInputOutputMismatchOpError(self):
    self._testInputOutputMismatchOpError('reshape')

  def testMultipleUnspecifiedDimensionsOpError(self):
    with self.assertRaisesError('must have at most one `-1`.'):
      shape_in, shape_out = self.build_shapes([2, 3], [4, -1, -1,])
      bijector = tfb.Reshape(
          event_shape_out=shape_out,
          event_shape_in=shape_in,
          validate_args=True)
      self.evaluate(bijector.forward_event_shape_tensor(shape_in))

  def testInvalidDimensionsOpError(self):
    shape_in, shape_out = self.build_shapes([2, 3], [1, 2, -2,])

    with self.assertRaisesError(
        'elements must be either positive integers or `-1`.'):
      bijector = tfb.Reshape(
          event_shape_out=shape_out,
          event_shape_in=shape_in,
          validate_args=True)
      self.evaluate(bijector.forward_event_shape_tensor(shape_in))

  def testUnknownShapeRank(self):
    if tf.executing_eagerly(): return
    unknown_shape = tf1.placeholder_with_default([2, 2], shape=None)
    known_shape = [2, 2]

    with self.assertRaisesRegexp(NotImplementedError,
                                 'must be statically known.'):
      tfb.Reshape(event_shape_out=unknown_shape)

    with self.assertRaisesRegexp(NotImplementedError,
                                 'must be statically known.'):
      tfb.Reshape(event_shape_out=known_shape, event_shape_in=unknown_shape)

  def testScalarInVectorOut(self):
    bijector = tfb.Reshape(event_shape_in=[], event_shape_out=[-1])
    self.assertAllEqual(np.zeros([3, 4, 5, 1]),
                        self.evaluate(bijector.forward(np.zeros([3, 4, 5]))))
    self.assertAllEqual(np.zeros([3, 4, 5]),
                        self.evaluate(bijector.inverse(np.zeros([3, 4, 5, 1]))))


if __name__ == '__main__':
  tf.test.main()
