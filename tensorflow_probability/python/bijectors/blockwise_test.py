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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from absl.testing import parameterized
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python.bijectors import bijector_test_util
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.internal import test_util


@test_util.test_all_tf_execution_regimes
class BlockwiseBijectorTest(test_util.TestCase):

  @parameterized.parameters((False, []), (True, []), (False, [2]), (True, [2]))
  def testExplicitBlocks(self, dynamic_shape, batch_shape):
    block_sizes = tf.convert_to_tensor(value=[2, 1, 3])
    block_sizes = tf1.placeholder_with_default(
        block_sizes, shape=None if dynamic_shape else block_sizes.shape)
    exp = tfb.Exp()
    sp = tfb.Softplus()
    aff = tfb.Affine(scale_diag=[2., 3., 4.])
    blockwise = tfb.Blockwise(bijectors=[exp, sp, aff], block_sizes=block_sizes)

    x = tf.cast([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=tf.float32)
    for s in batch_shape:
      x = tf.expand_dims(x, 0)
      x = tf.tile(x, [s] + [1] * (tensorshape_util.rank(x.shape) - 1))
    x = tf1.placeholder_with_default(
        x, shape=None if dynamic_shape else x.shape)

    # Identity to break the caching.
    blockwise_y = tf.identity(blockwise.forward(x))
    blockwise_fldj = blockwise.forward_log_det_jacobian(x, event_ndims=1)
    blockwise_x = blockwise.inverse(blockwise_y)
    blockwise_ildj = blockwise.inverse_log_det_jacobian(
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
        exp.forward(x[..., :2]),
        sp.forward(x[..., 2:3]),
        aff.forward(x[..., 3:]),
    ],
                       axis=-1)
    expl_fldj = sum([
        exp.forward_log_det_jacobian(x[..., :2], event_ndims=1),
        sp.forward_log_det_jacobian(x[..., 2:3], event_ndims=1),
        aff.forward_log_det_jacobian(x[..., 3:], event_ndims=1)
    ])
    expl_x = tf.concat([
        exp.inverse(expl_y[..., :2]),
        sp.inverse(expl_y[..., 2:3]),
        aff.inverse(expl_y[..., 3:])
    ],
                       axis=-1)
    expl_ildj = sum([
        exp.inverse_log_det_jacobian(expl_y[..., :2], event_ndims=1),
        sp.inverse_log_det_jacobian(expl_y[..., 2:3], event_ndims=1),
        aff.inverse_log_det_jacobian(expl_y[..., 3:], event_ndims=1)
    ])

    self.assertAllClose(self.evaluate(expl_y), self.evaluate(blockwise_y))
    self.assertAllClose(self.evaluate(expl_fldj), self.evaluate(blockwise_fldj))
    self.assertAllClose(self.evaluate(expl_x), self.evaluate(blockwise_x))
    self.assertAllClose(self.evaluate(expl_ildj), self.evaluate(blockwise_ildj))

  def testBijectiveAndFinite(self):
    exp = tfb.Exp()
    sp = tfb.Softplus()
    aff = tfb.Affine(scale_diag=[2., 3., 4.])
    blockwise = tfb.Blockwise(bijectors=[exp, sp, aff], block_sizes=[2, 1, 3])

    x = tf.cast([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=tf.float32)
    x = tf1.placeholder_with_default(x, shape=x.shape)
    # Identity to break the caching.
    blockwise_y = tf.identity(blockwise.forward(x))

    bijector_test_util.assert_bijective_and_finite(
        blockwise,
        x=self.evaluate(x),
        y=self.evaluate(blockwise_y),
        eval_func=self.evaluate,
        event_ndims=1)

  def testImplicitBlocks(self):
    exp = tfb.Exp()
    sp = tfb.Softplus()
    aff = tfb.Affine(scale_diag=[2.])
    blockwise = tfb.Blockwise(bijectors=[exp, sp, aff])
    self.assertAllEqual(self.evaluate(blockwise.block_sizes), [1, 1, 1])

  def testName(self):
    exp = tfb.Exp()
    sp = tfb.Softplus()
    aff = tfb.Affine(scale_diag=[2., 3., 4.])
    blockwise = tfb.Blockwise(bijectors=[exp, sp, aff], block_sizes=[2, 1, 3])
    self.assertStartsWith(blockwise.name,
                          'blockwise_of_exp_and_softplus_and_affine')

  def testNameOneBijector(self):
    exp = tfb.Exp()
    blockwise = tfb.Blockwise(bijectors=[exp], block_sizes=[3])
    self.assertStartsWith(blockwise.name, 'blockwise_of_exp')

  def testRaisesEmptyBijectors(self):
    with self.assertRaisesRegexp(ValueError, '`bijectors` must not be empty'):
      tfb.Blockwise(bijectors=[])

  def testRaisesBadBijectors(self):
    with self.assertRaisesRegexp(NotImplementedError,
                                 'Only scalar and vector event-shape'):
      tfb.Blockwise(bijectors=[tfb.Reshape(event_shape_out=[1, 1])])

    with self.assertRaisesRegexp(NotImplementedError,
                                 'Only scalar and vector event-shape'):
      tfb.Blockwise(
          bijectors=[tfb.Reshape(event_shape_out=[1], event_shape_in=[])])

  def testRaisesBadBlocks(self):
    with self.assertRaisesRegexp(
        ValueError,
        r'`block_sizes` must be `None`, or a vector of the same length as '
        r'`bijectors`. Got a `Tensor` with shape \(2L?,\) and `bijectors` of '
        r'length 1'):
      tfb.Blockwise(bijectors=[tfb.Exp()], block_sizes=[1, 2])

  def testRaisesBadBlocksDynamic(self):
    if tf.executing_eagerly(): return
    with self.assertRaises(tf.errors.InvalidArgumentError):
      block_sizes = tf1.placeholder_with_default([1, 2], shape=None)
      blockwise = tfb.Blockwise(
          bijectors=[tfb.Exp()], block_sizes=block_sizes, validate_args=True)
      self.evaluate(blockwise.block_sizes)

    with self.assertRaises(tf.errors.InvalidArgumentError):
      block_sizes = tf1.placeholder_with_default([[1]], shape=None)
      blockwise = tfb.Blockwise(
          bijectors=[tfb.Exp()], block_sizes=block_sizes, validate_args=True)
      self.evaluate(blockwise.block_sizes)


if __name__ == '__main__':
  tf.test.main()
