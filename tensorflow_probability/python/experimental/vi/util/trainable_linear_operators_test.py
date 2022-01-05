# Copyright 2021 The TensorFlow Probability Authors.
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
"""Tests for trainable `LinearOperator` builders."""

import functools
from absl.testing import parameterized

import numpy as np
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import test_util

tfb = tfp.bijectors
tfd = tfp.distributions


def _strip_defer(op):
  if isinstance(op, tfp.experimental.util.DeferredModule):
    return op._build_module()
  return op


def diag_operator(shape, dtype):
  init_fn = functools.partial(
      samplers.uniform, shape, maxval=2., dtype=dtype)
  apply_fn = lambda scale_diag: tf.linalg.LinearOperatorDiag(  # pylint: disable=g-long-lambda
      scale_diag, is_non_singular=True)
  return init_fn, apply_fn


def scaled_identity_operator(shape, dtype):
  init_fn = lambda _: ()
  apply_fn = lambda _: tf.linalg.LinearOperatorScaledIdentity(  # pylint: disable=g-long-lambda
      num_rows=shape[-1], multiplier=tf.constant(1., dtype=dtype))
  return init_fn, apply_fn


@test_util.test_all_tf_execution_regimes
class TrainableLinearOperators(test_util.TestCase):

  def _test_linear_operator_shape_and_gradients(
      self, op, batch_shape, domain_dim, range_dim, dtype):
    x = tf.ones(domain_dim, dtype=dtype)
    with tf.GradientTape() as tape:
      y = op.matvec(x)
    grad = tape.gradient(y, op.trainable_variables)
    self.evaluate([v.initializer for v in op.trainable_variables])

    self.assertEqual(self.evaluate(op.domain_dimension_tensor()), domain_dim)
    self.assertEqual(self.evaluate(op.range_dimension_tensor()), range_dim)
    self.assertAllEqual(self.evaluate(op.batch_shape_tensor()),
                        np.array(batch_shape, dtype=np.int32))
    self.assertDTypeEqual(self.evaluate(op.to_dense()), dtype.as_numpy_dtype)
    self.assertNotEmpty(op.trainable_variables)
    self.assertAllNotNone(grad)

  @parameterized.parameters(
      ([5], tf.float32, False),
      ([3, 2, 4], tf.float64, True))
  def test_trainable_linear_operator_diag(self, shape, dtype, use_static_shape):
    scale_initializer = tf.constant(1e-2, dtype=dtype)
    op = tfp.experimental.vi.util.build_trainable_linear_operator_diag(
        _build_shape(shape, is_static=use_static_shape),
        scale_initializer=scale_initializer,
        dtype=dtype)
    dim = shape[-1]
    self._test_linear_operator_shape_and_gradients(
        op, shape[:-1], dim, dim, dtype)

  @parameterized.parameters(
      (np.array([4, 4], np.int32), tf.float32, True),
      ([6], tf.float64, False))
  def test_trainable_linear_operator_tril(self, shape, dtype, use_static_shape):
    op = tfp.experimental.vi.util.build_trainable_linear_operator_tril(
        _build_shape(shape, is_static=use_static_shape),
        dtype=dtype)
    dim = shape[-1]
    self._test_linear_operator_shape_and_gradients(
        op, shape[:-1], dim, dim, dtype)

  @parameterized.parameters(
      ([5, 3], tf.float32, False),
      ([2, 1, 2, 3], tf.float64, True))
  def test_trainable_linear_operator_full_matrix(
      self, shape, dtype, use_static_shape):
    scale_initializer = tf.constant(1e-2, dtype=dtype)
    op = tfp.experimental.vi.util.build_trainable_linear_operator_full_matrix(
        _build_shape(shape, is_static=use_static_shape),
        scale_initializer=scale_initializer)
    self._test_linear_operator_shape_and_gradients(
        op, shape[:-2], shape[-1], shape[-2], dtype)

  @parameterized.parameters(
      ((2, 3), (),
       [[tf.linalg.LinearOperatorDiag],
        [None, tf.linalg.LinearOperatorIdentity(3)]],
       tf.float32,
       True),
      ((2, 4), (3,),
       ((tf.linalg.LinearOperatorLowerTriangular,),
        (tf.linalg.LinearOperatorZeros, diag_operator)),
       tf.float64,
       False),
      )
  def test_trainable_linear_operator_block_tril(
      self, block_dims, batch_shape, operators, dtype, use_static_shape):
    op = (tfp.experimental.vi.util.
          build_trainable_linear_operator_block(
              operators,
              block_dims=_build_shape(block_dims, is_static=use_static_shape),
              batch_shape=_build_shape(batch_shape, is_static=use_static_shape),
              dtype=dtype))
    dim = sum(block_dims)
    self._test_linear_operator_shape_and_gradients(
        op, batch_shape, dim, dim, dtype)
    self.assertIsInstance(_strip_defer(op),
                          tf.linalg.LinearOperatorBlockLowerTriangular)

    # Test that bijector builds.
    tfb.ScaleMatvecLinearOperatorBlock(op, validate_args=True)

  @parameterized.parameters(
      ((2, 3), (),
       (tf.linalg.LinearOperatorDiag, tf.linalg.LinearOperatorIdentity(3)),
       tf.float32,
       False),
      ((2,), (3,),
       (tf.linalg.LinearOperatorLowerTriangular,),
       tf.float64,
       True),
      ((3, 2), (2, 2),
       (tf.linalg.LinearOperatorDiag, scaled_identity_operator),
       tf.float32,
       False)
      )
  def test_trainable_linear_operator_block_diag(
      self, block_dims, batch_shape, operators, dtype, use_static_shape):
    op = (tfp.experimental.vi.util.
          build_trainable_linear_operator_block(
              operators,
              block_dims=_build_shape(block_dims, is_static=use_static_shape),
              batch_shape=_build_shape(batch_shape, is_static=use_static_shape),
              dtype=dtype))
    dim = sum(block_dims)
    self._test_linear_operator_shape_and_gradients(
        op, batch_shape, dim, dim, dtype)
    self.assertIsInstance(_strip_defer(op), tf.linalg.LinearOperatorBlockDiag)

    # Test that bijector builds.
    tfb.ScaleMatvecLinearOperatorBlock(op, validate_args=True)

  def test_deterministic_initialization_from_seed(self):
    seed = test_util.test_seed(sampler_type='stateless')
    op = tfp.experimental.vi.util.build_trainable_linear_operator_block(
        operators=(tf.linalg.LinearOperatorDiag,
                   tf.linalg.LinearOperatorLowerTriangular),
        block_dims=(2, 3), batch_shape=[3], dtype=tf.float32, seed=seed)
    self.evaluate([v.initializer for v in op.trainable_variables])
    op2 = tfp.experimental.vi.util.build_trainable_linear_operator_block(
        operators=(tf.linalg.LinearOperatorDiag,
                   tf.linalg.LinearOperatorLowerTriangular),
        block_dims=(2, 3), batch_shape=[3], dtype=tf.float32, seed=seed)
    self.evaluate([v.initializer for v in op2.trainable_variables])
    self.assertAllEqual(op.to_dense(), op2.to_dense())

  def test_undefined_block_dims_raises(self):
    op = (
        tfp.experimental.vi.util.build_trainable_linear_operator_block(
            operators=(
                (tf.linalg.LinearOperatorIdentity(2),),
                (tf.linalg.LinearOperatorZeros(
                    3, 2, is_square=False, is_self_adjoint=False,
                    dtype=tf.float32),
                 tf.linalg.LinearOperatorDiag(
                     tf.Variable(tf.ones((2, 3), dtype=tf.float32)),
                     is_non_singular=True)))))
    self.evaluate([v.initializer for v in op.trainable_variables])
    self.assertAllEqual(self.evaluate(op.shape_tensor()), [2, 5, 5])
    self.assertLen(op.trainable_variables, 1)

    operators = (tf.linalg.LinearOperatorIdentity(2),
                 tf.linalg.LinearOperatorDiag)
    with self.assertRaisesRegexp(ValueError, '`block_dims` must be defined'):
      tfp.experimental.vi.util.build_trainable_linear_operator_block(
          operators)

  @parameterized.parameters(
      ((3, 4), tf.float32, True),
      ((2, 4, 2), tf.float64, False))
  def test_linear_operator_zeros(self, shape, dtype, use_static_shape):
    op = tfp.experimental.vi.util.build_linear_operator_zeros(
        _build_shape(shape, is_static=use_static_shape), dtype=dtype)
    self.assertAllEqual(self.evaluate(op.shape_tensor()), shape)
    self.assertIsInstance(_strip_defer(op), tf.linalg.LinearOperatorZeros)
    self.assertDTypeEqual(self.evaluate(op.to_dense()), dtype.as_numpy_dtype)


def _build_shape(shape, is_static):
  if is_static:
    return shape
  input_shape = np.array(shape, dtype=np.int32)
  return tf1.placeholder_with_default(
      input_shape, shape=input_shape.shape)


if __name__ == '__main__':
  test_util.main()
