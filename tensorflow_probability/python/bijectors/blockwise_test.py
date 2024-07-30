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
"""Tests for the Blockwise bijector."""

# Dependency imports

from absl.testing import parameterized
import mock
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
from tensorflow_probability.python.bijectors import bijector_test_util
from tensorflow_probability.python.bijectors import blockwise
from tensorflow_probability.python.bijectors import exp
from tensorflow_probability.python.bijectors import inline
from tensorflow_probability.python.bijectors import scale
from tensorflow_probability.python.bijectors import scale_matvec_diag
from tensorflow_probability.python.bijectors import softmax_centered
from tensorflow_probability.python.bijectors import softplus
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.internal import test_util


@test_util.test_all_tf_execution_regimes
class BlockwiseBijectorTest(test_util.TestCase):

  @parameterized.named_parameters(
      ('static', False, []),
      ('dynamic', True, []),
      ('static_batched', False, [2]),
      ('dynamic_batched', True, [2]),
  )
  def testExplicitBlocks(self, dynamic_shape, batch_shape):
    block_sizes = tf.convert_to_tensor(value=[2, 1, 3])
    block_sizes = tf1.placeholder_with_default(
        block_sizes,
        shape=([None] * len(block_sizes.shape)
               if dynamic_shape else
               block_sizes.shape))
    e = exp.Exp()
    sp = softplus.Softplus()
    aff = scale_matvec_diag.ScaleMatvecDiag(scale_diag=[2., 3., 4.])
    bijector = blockwise.Blockwise(
        bijectors=[e, sp, aff],
        block_sizes=block_sizes,
        maybe_changes_size=False)

    x = tf.cast([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=tf.float32)
    for s in batch_shape:
      x = tf.expand_dims(x, 0)
      x = tf.tile(x, [s] + [1] * (tensorshape_util.rank(x.shape) - 1))
    x = tf1.placeholder_with_default(
        x, shape=None if dynamic_shape else x.shape)

    # Identity to break the caching.
    blockwise_y = tf.identity(bijector.forward(x))
    blockwise_fldj = bijector.forward_log_det_jacobian(x, event_ndims=1)
    blockwise_x = bijector.inverse(blockwise_y)
    blockwise_ildj = bijector.inverse_log_det_jacobian(
        blockwise_y, event_ndims=1)

    if not dynamic_shape:
      self.assertEqual(blockwise_y.shape, batch_shape + [6])
      self.assertEqual(blockwise_fldj.shape, batch_shape + [])
      self.assertEqual(blockwise_x.shape, batch_shape + [6])
      self.assertEqual(blockwise_ildj.shape, batch_shape + [])
    self.assertAllEqual(
        self.evaluate(tf.shape(blockwise_y)), batch_shape + [6])
    self.assertAllEqual(
        self.evaluate(tf.shape(blockwise_fldj)), batch_shape + [])
    self.assertAllEqual(
        self.evaluate(tf.shape(blockwise_x)), batch_shape + [6])
    self.assertAllEqual(
        self.evaluate(tf.shape(blockwise_ildj)), batch_shape + [])

    expl_y = tf.concat([
        e.forward(x[..., :2]),
        sp.forward(x[..., 2:3]),
        aff.forward(x[..., 3:]),
    ],
                       axis=-1)
    expl_fldj = sum([
        e.forward_log_det_jacobian(x[..., :2], event_ndims=1),
        sp.forward_log_det_jacobian(x[..., 2:3], event_ndims=1),
        aff.forward_log_det_jacobian(x[..., 3:], event_ndims=1)
    ])
    expl_x = tf.concat([
        e.inverse(expl_y[..., :2]),
        sp.inverse(expl_y[..., 2:3]),
        aff.inverse(expl_y[..., 3:])
    ],
                       axis=-1)
    expl_ildj = sum([
        e.inverse_log_det_jacobian(expl_y[..., :2], event_ndims=1),
        sp.inverse_log_det_jacobian(expl_y[..., 2:3], event_ndims=1),
        aff.inverse_log_det_jacobian(expl_y[..., 3:], event_ndims=1)
    ])

    self.assertAllClose(self.evaluate(expl_y), self.evaluate(blockwise_y))
    self.assertAllClose(self.evaluate(expl_fldj), self.evaluate(blockwise_fldj))
    self.assertAllClose(self.evaluate(expl_x), self.evaluate(blockwise_x))
    self.assertAllClose(self.evaluate(expl_ildj), self.evaluate(blockwise_ildj))

  @parameterized.named_parameters(
      ('static', False, []),
      ('dynamic', True, []),
      ('static_batched', False, [2]),
      ('dynamic_batched', True, [2]),
  )
  def testSizeChangingExplicitBlocks(self, dynamic_shape, batch_shape):
    block_sizes = tf.convert_to_tensor(value=[2, 1, 3])
    if dynamic_shape:
      block_sizes = tf1.placeholder_with_default(
          block_sizes, shape=block_sizes.shape)
    e = exp.Exp()
    sc = softmax_centered.SoftmaxCentered()
    aff = scale_matvec_diag.ScaleMatvecDiag(scale_diag=[2., 3., 4.])
    bijector = blockwise.Blockwise(
        bijectors=[e, sc, aff],
        block_sizes=block_sizes,
        maybe_changes_size=True)

    x = tf.cast([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=tf.float32)
    for s in batch_shape:
      x = tf.expand_dims(x, 0)
      x = tf.tile(x, [s] + [1] * (tensorshape_util.rank(x.shape) - 1))
    x = tf1.placeholder_with_default(
        x, shape=None if dynamic_shape else x.shape)

    # Identity to break the caching.
    blockwise_y = tf.identity(bijector.forward(x))
    blockwise_fldj = bijector.forward_log_det_jacobian(x, event_ndims=1)
    blockwise_y_shape_tensor = bijector.forward_event_shape_tensor(tf.shape(x))
    blockwise_y_shape = bijector.forward_event_shape(x.shape)

    blockwise_x = bijector.inverse(blockwise_y)
    blockwise_x_shape_tensor = bijector.inverse_event_shape_tensor(
        tf.shape(blockwise_y))
    blockwise_x_shape = bijector.inverse_event_shape(blockwise_y.shape)
    blockwise_ildj = bijector.inverse_log_det_jacobian(
        blockwise_y, event_ndims=1)

    if not dynamic_shape:
      self.assertEqual(blockwise_y.shape, batch_shape + [7])
      self.assertEqual(blockwise_y_shape, batch_shape + [7])
      self.assertEqual(blockwise_fldj.shape, batch_shape + [])
      self.assertEqual(blockwise_x.shape, batch_shape + [6])
      self.assertEqual(blockwise_x_shape, batch_shape + [6])
      self.assertEqual(blockwise_ildj.shape, batch_shape + [])
    self.assertAllEqual(
        self.evaluate(tf.shape(blockwise_y)), batch_shape + [7])
    self.assertAllEqual(
        self.evaluate(blockwise_y_shape_tensor), batch_shape + [7])
    self.assertAllEqual(
        self.evaluate(tf.shape(blockwise_fldj)), batch_shape + [])
    self.assertAllEqual(
        self.evaluate(tf.shape(blockwise_x)), batch_shape + [6])
    self.assertAllEqual(
        self.evaluate(blockwise_x_shape_tensor), batch_shape + [6])
    self.assertAllEqual(
        self.evaluate(tf.shape(blockwise_ildj)), batch_shape + [])

    expl_y = tf.concat([
        e.forward(x[..., :2]),
        sc.forward(x[..., 2:3]),
        aff.forward(x[..., 3:]),
    ],
                       axis=-1)
    expl_fldj = sum([
        e.forward_log_det_jacobian(x[..., :2], event_ndims=1),
        sc.forward_log_det_jacobian(x[..., 2:3], event_ndims=1),
        aff.forward_log_det_jacobian(x[..., 3:], event_ndims=1)
    ])
    expl_x = tf.concat([
        e.inverse(expl_y[..., :2]),
        sc.inverse(expl_y[..., 2:4]),
        aff.inverse(expl_y[..., 4:])
    ],
                       axis=-1)
    expl_ildj = sum([
        e.inverse_log_det_jacobian(expl_y[..., :2], event_ndims=1),
        sc.inverse_log_det_jacobian(expl_y[..., 2:4], event_ndims=1),
        aff.inverse_log_det_jacobian(expl_y[..., 4:], event_ndims=1)
    ])

    self.assertAllClose(self.evaluate(expl_y), self.evaluate(blockwise_y))
    self.assertAllClose(self.evaluate(expl_fldj), self.evaluate(blockwise_fldj))
    self.assertAllClose(self.evaluate(expl_x), self.evaluate(blockwise_x))
    self.assertAllClose(self.evaluate(expl_ildj), self.evaluate(blockwise_ildj))

  def testBijectiveAndFinite(self):
    e = exp.Exp()
    sp = softplus.Softplus()
    aff = scale_matvec_diag.ScaleMatvecDiag(scale_diag=[2., 3., 4.])
    bijector = blockwise.Blockwise(
        bijectors=[e, sp, aff], block_sizes=[2, 1, 3])

    x = tf.cast([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=tf.float32)
    x = tf1.placeholder_with_default(x, shape=x.shape)
    # Identity to break the caching.
    blockwise_y = tf.identity(bijector.forward(x))

    bijector_test_util.assert_bijective_and_finite(
        bijector,
        x=self.evaluate(x),
        y=self.evaluate(blockwise_y),
        eval_func=self.evaluate,
        event_ndims=1)

  def testImplicitBlocks(self):
    e = exp.Exp()
    sp = softplus.Softplus()
    aff = scale_matvec_diag.ScaleMatvecDiag(scale_diag=[2.])
    bijector = blockwise.Blockwise(bijectors=[e, sp, aff])
    self.assertAllEqual(self.evaluate(bijector.block_sizes), [1, 1, 1])

  def testName(self):
    e = exp.Exp()
    sp = softplus.Softplus()
    aff = scale_matvec_diag.ScaleMatvecDiag(scale_diag=[2., 3., 4.])
    bijector = blockwise.Blockwise(
        bijectors=[e, sp, aff], block_sizes=[2, 1, 3])
    self.assertStartsWith(
        bijector.name, 'blockwise_of_exp_and_softplus_and_scale_matvec_diag')

  def testNameOneBijector(self):
    e = exp.Exp()
    bijector = blockwise.Blockwise(bijectors=[e], block_sizes=[3])
    self.assertStartsWith(bijector.name, 'blockwise_of_exp')

  def testRaisesEmptyBijectors(self):
    with self.assertRaisesRegex(ValueError, '`bijectors` must not be empty'):
      blockwise.Blockwise(bijectors=[])

  def testRaisesBadBlocks(self):
    with self.assertRaisesRegex(
        ValueError,
        r'`block_sizes` must be `None`, or a vector of the same length as '
        r'`bijectors`. Got a `Tensor` with shape \(2L?,\) and `bijectors` of '
        r'length 1'):
      blockwise.Blockwise(bijectors=[exp.Exp()], block_sizes=[1, 2])

  def testRaisesBadBlocksDynamic(self):
    if tf.executing_eagerly(): return
    with self.assertRaises(tf.errors.InvalidArgumentError):
      block_sizes = tf1.placeholder_with_default([1, 2], shape=None)
      bijector = blockwise.Blockwise(
          bijectors=[exp.Exp()], block_sizes=block_sizes, validate_args=True)
      self.evaluate(bijector.block_sizes)

    with self.assertRaises(tf.errors.InvalidArgumentError):
      block_sizes = tf1.placeholder_with_default([[1]], shape=None)
      bijector = blockwise.Blockwise(
          bijectors=[exp.Exp()], block_sizes=block_sizes, validate_args=True)
      self.evaluate(bijector.block_sizes)

  def testKwargs(self):
    zeros = tf.zeros(1)

    bijectors = [
        inline.Inline(  # pylint: disable=g-complex-comprehension
            forward_fn=mock.Mock(return_value=zeros),
            inverse_fn=mock.Mock(return_value=zeros),
            forward_log_det_jacobian_fn=mock.Mock(return_value=zeros),
            inverse_log_det_jacobian_fn=mock.Mock(return_value=zeros),
            forward_min_event_ndims=0,
            name='inner{}'.format(i)) for i in range(2)
    ]

    bijector = blockwise.Blockwise(bijectors)

    x = [1, 2]
    bijector.forward(x, inner0={'arg': 1}, inner1={'arg': 2})
    bijector.inverse(x, inner0={'arg': 3}, inner1={'arg': 4})
    bijector.forward_log_det_jacobian(
        x, event_ndims=1, inner0={'arg': 5}, inner1={'arg': 6})
    bijector.inverse_log_det_jacobian(
        x, event_ndims=1, inner0={'arg': 7}, inner1={'arg': 8})

    bijectors[0]._forward.assert_any_call(mock.ANY, arg=1)
    bijectors[1]._forward.assert_any_call(mock.ANY, arg=2)
    bijectors[0]._inverse.assert_any_call(mock.ANY, arg=3)
    bijectors[1]._inverse.assert_any_call(mock.ANY, arg=4)
    bijectors[0]._forward_log_det_jacobian.assert_called_with(mock.ANY, arg=5)
    bijectors[1]._forward_log_det_jacobian.assert_called_with(mock.ANY, arg=6)
    bijectors[0]._inverse_log_det_jacobian.assert_called_with(mock.ANY, arg=7)
    bijectors[1]._inverse_log_det_jacobian.assert_called_with(mock.ANY, arg=8)

  @test_util.disable_test_for_backend(
      disable_numpy=True, disable_jax=True,
      reason='Numpy and JAX have no notion of CompositeTensor.')
  def testCompositeTensor(self):
    e = exp.Exp()
    sp = softplus.Softplus()
    aff = scale.Scale(scale=2.)
    bijector = blockwise.Blockwise(bijectors=[e, sp, aff])
    self.assertIsInstance(bijector, tf.__internal__.CompositeTensor)

    # Bijector may be flattened into `Tensor` components and rebuilt.
    flat = tf.nest.flatten(bijector, expand_composites=True)
    unflat = tf.nest.pack_sequence_as(bijector, flat, expand_composites=True)
    self.assertIsInstance(unflat, blockwise.Blockwise)

    # Bijector may be input to a `tf.function`-decorated callable.
    @tf.function
    def call_forward(bij, x):
      return bij.forward(x)

    x = tf.ones([2, 3], dtype=tf.float32)
    self.assertAllClose(call_forward(unflat, x), bijector.forward(x))

    # Type spec can be encoded/decoded.
    enc = tf.__internal__.saved_model.encode_structure(bijector._type_spec)
    dec = tf.__internal__.saved_model.decode_proto(enc)
    self.assertEqual(bijector._type_spec, dec)

  def testNonCompositeTensor(self):
    e = exp.Exp()
    s = bijector_test_util.NonCompositeTensorScale(scale=tf.constant(3.))
    bijector = blockwise.Blockwise(bijectors=[e, s])
    self.assertNotIsInstance(bijector, tf.__internal__.CompositeTensor)
    self.assertAllClose(
        bijector.forward([1., 1.]),
        tf.convert_to_tensor([e.forward(1.), s.forward(1.)]))


if __name__ == '__main__':
  test_util.main()
