# Copyright 2020 The TensorFlow Probability Authors.
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
"""Tests for batched convolutions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from absl.testing import parameterized

import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.experimental.nn.util import convolution_util
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import test_util

tfn = tfp.experimental.nn


# pylint: disable=bad-whitespace
_CONV_TEST_CASES = (
    # input dim         filter   c_out  strides    padding     dilations
    ((1, 32, 32, 3),    (3, 4),  2,     (1, 1),    'VALID',    (1, 1)),
    ((5, 2, 32, 32, 3), (2, 2),  4,     (1, 2),    'SAME',     (1, 1)),
    ((5, 2, 7, 7, 3),   (2, 2),  4,     (1, 2),    'SAME',     (2, 1)),
    ((5, 2, 13, 13, 3), (2, 2),  4,     (1, 2),    'SAME',     (1, 1)),
    ((4, 28, 28, 2),    (2, 3),  2,     (2, 2),    'VALID',    (1, 2)),
    ((4, 8, 8, 3),      (1, 1),  2,     (2, 3),    'VALID',    (1, 1)),
    )

_CONV_TRANSPOSE_TEST_CASES = (
    # input dim         filter   c_out  strides    padding     dilations
    ((2, 16, 16, 3),    (3, 3),  4,     (2, 2),    'SAME',     (1, 1)),
    ((2, 16, 16, 3),    (4, 4),  3,     (2, 2),    'SAME',     (1, 1)),
    ((2, 8, 8, 2),      (3, 3),  3,     (1, 2),    'SAME',     (1, 1)),
    ((4, 9, 9, 3),      (3, 3),  2,     (1, 1),    'SAME',     (2, 2)),
    ((4, 12, 9, 3),     (3, 3),  1,     (3, 3),    'VALID',    (1, 1)),
    ((2, 12, 12, 2),    (2, 3),  1,     (2, 2),    'VALID',    (1, 1)),
    ((4, 9, 9, 3),      (1, 1),  2,     (2, 2),    'VALID',    (1, 1)),
    ((4, 9, 9, 3),      (1, 2),  2,     (3, 3),    'VALID',    (1, 1)),
    ((4, 8, 7, 1),      (3, 4),  1,     (2, 2),    'SAME',     (1, 1)),
    ((4, 8, 7, 1),      (4, 2),  1,     (3, 3),    'SAME',     (1, 1)),
    )
# pylint: enable=bad-whitespace


def _make_input_and_kernel(
    make_input, input_batch_shape, input_shape, kernel_batch_shape,
    filter_shape, channels_out, dtype):
  total_input_shape = ps.concat([input_batch_shape, input_shape], axis=0)
  total_kernel_shape = ps.concat(
      [kernel_batch_shape, [filter_shape[0] * filter_shape[1] * input_shape[-1],
                            channels_out]], axis=0)
  # Use integers for numerical stability.
  sample_fn = lambda s: make_input(tf.cast(  # pylint: disable=g-long-lambda
      tf.random.uniform(
          ps.cast(s, tf.int32), minval=-10, maxval=10, dtype=tf.int32),
      dtype=dtype))
  return sample_fn(total_input_shape), sample_fn(total_kernel_shape)


def _get_conv_transpose_fn(method):
  if method == 'subkernels':
    return tfn.util.make_convolution_transpose_fn_with_subkernels
  elif method == 'subkernels_matrix':
    return tfn.util.make_convolution_transpose_fn_with_subkernels_matrix
  elif method == 'dilation':
    return tfn.util.make_convolution_transpose_fn_with_dilation
  else:
    raise ValueError('Unsupported method for `_get_conv_transpose_fn`: {}.'
                     ''.format(method))


class _Common(object):
  """Common methods for Conv/ConvTranspose tests."""

  def assertRaisesMaybeStaticError(self, msg):
    if tf.executing_eagerly() or self.use_static_shape:
      return self.assertRaisesRegex(ValueError, msg)
    return self.assertRaisesOpError(msg)

  def make_integer_input(self, number):
    if self.use_static_shape:
      return number
    output = tf.Variable(number, dtype=tf.int32)
    self.evaluate(output.initializer)
    return output


@test_util.test_all_tf_execution_regimes
class Im2RowTest(test_util.TestCase):

  def test_works_like_conv2d(self):
    x = tf.constant([[
        [[2], [1], [2], [0], [1]],
        [[1], [3], [2], [2], [3]],
        [[1], [1], [3], [3], [0]],
        [[2], [2], [0], [1], [1]],
        [[0], [0], [3], [1], [2]],
    ]], tf.float32)  # shape=[1, 5, 5, 1]
    x = tf.concat([x, x], axis=-1)
    k = tf.constant([
        [[[2, 0.1]], [[3, 0.2]]],
        [[[0, 0.3]], [[1, 0.4]]],
    ], tf.float32)  # shape=[2, 2, 1, 2]
    k = tf.concat([k, k], axis=-2)
    strides = [1, 2]
    im2row_x = tfn.util.im2row(
        x,
        block_shape=ps.shape(k)[:2],
        slice_step=strides,
        padding='VALID')
    y_expected = tf.nn.conv2d(x, k, strides=strides, padding='VALID')
    y_actual = tf.matmul(im2row_x, tf.reshape(k, shape=[-1, k.shape[-1]]))
    [y_expected_, y_actual_] = self.evaluate([y_expected, y_actual])
    self.assertAllClose(y_expected_, y_actual_, rtol=1e-5, atol=0)

  @parameterized.parameters((tf.int32, np.int32), (tf.int64, np.int64))
  def test_dtype(self, tf_dtype, np_dtype):
    ind, _ = tfn.util.im2row_index(
        input_shape=(1, 12, 16, 3),
        block_shape=(2, 3),
        dtype=tf_dtype)
    self.assertDTypeEqual(ind, np_dtype)


@test_util.test_all_tf_execution_regimes
class ConvolutionUtilsTest(test_util.TestCase, _Common):

  use_static_shape = False

  def test_prepare_tuple_argument(self):

    rank = 3

    # Test that scalars are processed to tuples.
    arg = convolution_util.prepare_tuple_argument(
        self.make_integer_input(2), n=rank, arg_name='arg', validate_args=True)
    self.assertIsInstance(arg, list)
    self.assertLen(arg, rank)

    # Test that `Tensor` args are processed correctly.
    arg = convolution_util.prepare_tuple_argument(
        self.make_integer_input(
            [2, 3, 4]), n=rank, arg_name='arg_2', validate_args=True)
    self.assertIsInstance(arg, list)
    self.assertLen(arg, rank)

    with self.assertRaisesRegex(
        ValueError, 'must be equal to `1` or to the rank'):
      convolution_util.prepare_tuple_argument(
          self.make_integer_input([1, 2]), n=rank, arg_name='invalid_arg',
          validate_args=True)

  def test_prepare_conv_args(self):
    [filter_shape,
     rank,
     strides,
     padding,
     dilations] = convolution_util.prepare_conv_args(
         (3, 3),
         rank=2,
         strides=2,
         padding='same',
         dilations=(1, 1))

    for arg in [filter_shape, strides, dilations]:
      self.assertLen(arg, rank)

    self.assertEqual(padding, 'SAME')


@test_util.test_all_tf_execution_regimes
class _BatchedConvTest(test_util.TestCase, _Common):

  @parameterized.parameters(*_CONV_TEST_CASES)
  def test_works_like_conv2d(
      self, input_shape, filter_shape, channels_out,
      strides, padding, dilations):
    x, k = _make_input_and_kernel(
        self.make_input,
        input_batch_shape=[],
        input_shape=input_shape,
        # Use singleton kernel_batch_shape to bypass the short circuit to tf.nn.
        kernel_batch_shape=[1],
        filter_shape=filter_shape,
        channels_out=channels_out,
        dtype=self.dtype)

    tf_kernel = tf.reshape(
        k, shape=(filter_shape) + (input_shape[-1], channels_out))
    y_expected = tf.nn.conv2d(
        x, tf_kernel, strides=strides, padding=padding, dilations=dilations)

    conv_fn = tfn.util.make_convolution_fn(
        self.make_integer_input(filter_shape),
        rank=2,
        strides=self.make_integer_input(strides),
        padding=padding,
        dilations=self.make_integer_input(dilations),
        validate_args=True)

    with tf.GradientTape() as tape:
      tape.watch([x, k])
      y_actual = conv_fn(x, k)
    grad = tape.gradient(y_actual, [x, k])
    self.assertAllNotNone(grad)

    [y_expected_, y_actual_] = self.evaluate([y_expected, y_actual])
    self.assertAllClose(y_expected_, y_actual_, rtol=1e-5, atol=0)

  @parameterized.parameters(
      ((1,), ()),           # scalar input batch, scalar kernel batch
      ((1,), (2, 3)),       # non-scalar kernel batch
      ((3, 4), ()),         # non-scalar input batch
      ((3, 1), (2,)),       # broadcasting kernel and input batch shapes
      ((2, 3), (2, 3),))    # same kernel and input batch shapes
  def test_batching(self, input_batch_shape, kernel_batch_shape):
    input_shape = (12, 12, 2)
    filter_shape = (2, 2)
    channels_out = 3
    strides = (1, 1)
    dilations = (1, 1)
    padding = 'SAME'

    x, k = _make_input_and_kernel(
        self.make_input,
        input_batch_shape=input_batch_shape,
        input_shape=input_shape,
        kernel_batch_shape=kernel_batch_shape,
        filter_shape=filter_shape,
        channels_out=channels_out,
        dtype=self.dtype)

    conv_fn = tfn.util.make_convolution_fn(
        filter_shape, rank=2, strides=strides, padding=padding,
        dilations=dilations, validate_args=True)
    y_batched = conv_fn(x, k)

    broadcast_batch_shape = ps.broadcast_shape(
        input_batch_shape, kernel_batch_shape)
    broadcasted_input = tf.broadcast_to(
        x, shape=ps.concat([broadcast_batch_shape, input_shape], axis=0))
    broadcasted_kernel = tf.broadcast_to(
        k, shape=ps.concat([broadcast_batch_shape, ps.shape(k)[-2:]], axis=0))

    flat_y = tf.reshape(
        y_batched,
        shape=ps.pad(
            ps.shape(y_batched)[-3:], paddings=[[1, 0]], constant_values=-1))
    flat_x = tf.reshape(
        broadcasted_input,
        shape=ps.pad(input_shape, paddings=[[1, 0]], constant_values=-1))
    flat_tf_kernel = tf.reshape(
        broadcasted_kernel,
        shape=ps.concat([(-1,), filter_shape, (input_shape[-1], channels_out)],
                        axis=0))

    y_expected = tf.vectorized_map(
        lambda args: tf.nn.conv2d(  # pylint: disable=g-long-lambda
            args[0][tf.newaxis],
            args[1],
            strides=strides,
            padding=padding),
        elems=(flat_x, flat_tf_kernel))

    [y_actual_, y_expected_] = self.evaluate(
        [flat_y, tf.squeeze(y_expected, axis=1)])
    self.assertAllClose(y_expected_, y_actual_, rtol=1e-5, atol=0)

  def test_incompatible_shapes_raises(self):
    filter_shape = (3, 3)

    # Inconsistent channels in for kernel and image.
    c_in_kernel = 6
    c_in_image = 8
    c_out = 12

    k_dim = np.prod(filter_shape) * c_in_kernel
    kernel = self.make_input(tf.ones((2, k_dim, c_out), dtype=tf.float32))
    x = self.make_input(tf.ones((3, 2, 16, 16, c_in_image), dtype=tf.float32))
    conv_fn = tfn.util.make_convolution_fn(
        self.make_integer_input(filter_shape),
        rank=2,
        strides=self.make_integer_input((1, 1)),
        padding='SAME',
        dilations=self.make_integer_input((1, 1)),
        validate_args=True)
    with self.assertRaisesMaybeStaticError('size of the rightmost dimension'):
      self.evaluate(conv_fn(x, kernel))

  def test_dtype(self):
    # Test int64 indices.
    conv_fn = tfn.util.make_convolution_fn(
        (2, 2), rank=2, strides=(1, 1), padding='SAME', dilations=(1, 1),
        dtype=tf.int64, validate_args=True)
    x = tf.ones((2, 8, 8, 2), dtype=tf.float32)
    kernel = tf.ones((2, 8, 2), dtype=tf.float32)
    _ = self.evaluate(conv_fn(x, kernel))

    # Test f64 input.
    conv_fn = tfn.util.make_convolution_fn(
        self.make_integer_input((2, 2)),
        rank=2,
        strides=self.make_integer_input((1, 1)),
        padding='SAME',
        dilations=self.make_integer_input((1, 1)),
        validate_args=True)
    x = tf.ones((2, 8, 8, 2), dtype=tf.float64)
    kernel = tf.ones((2, 8, 2), dtype=tf.float64)
    y = self.evaluate(conv_fn(x, kernel))
    self.assertDTypeEqual(y, np.float64)


@test_util.test_all_tf_execution_regimes
class _BatchedConvTransposeTest(test_util.TestCase, _Common):

  dynamic_strides_ok = True
  unequal_strides_ok = True

  def make_conv_fn(self, filter_shape, strides, padding, dilations):
    return _get_conv_transpose_fn(self.method)(
        self.make_integer_input(filter_shape),
        strides=(self.make_integer_input(strides)
                 if self.dynamic_strides_ok else strides),
        padding=padding,
        dilations=self.make_integer_input(dilations),
        validate_args=True)

  @parameterized.parameters(*_CONV_TRANSPOSE_TEST_CASES)
  def test_works_like_conv2d_transpose(
      self, input_shape, filter_shape, channels_out, strides, padding,
      dilations):

    strides_tuple = strides
    if not self.unequal_strides_ok:
      if strides[0] != strides[1]:
        # Skip this test case if the method does not support unequal strides.
        return
      else:
        strides = strides[0]

    x, k = _make_input_and_kernel(
        self.make_input,
        input_batch_shape=[],
        input_shape=input_shape,
        # Use singleton kernel_batch_shape to avoid the short circuit to
        # `conv2d_transpose`.
        kernel_batch_shape=[1],
        filter_shape=filter_shape,
        channels_out=channels_out,
        dtype=self.dtype)

    output_shape, strides_ = convolution_util._get_output_shape(
        rank=2, strides=strides_tuple, padding=padding, dilations=dilations,
        input_shape=input_shape, output_size=channels_out,
        filter_shape=filter_shape)

    tf_kernel = tf.transpose(
        tf.reshape(k, ps.concat(
            [filter_shape, [input_shape[-1], channels_out]], axis=0)),
        perm=[0, 1, 3, 2])
    # conv2d_transpose does not support dilations > 1; use Keras instead.
    if any(d > 1 for d in dilations):
      keras_convt = tf.keras.layers.Conv2DTranspose(
          filters=channels_out,
          kernel_size=filter_shape,
          strides=strides,
          padding=padding,
          dilation_rate=dilations,
          use_bias=False)
      _ = keras_convt(x)  # build kernel
      keras_convt.kernel = tf_kernel
      y_expected = keras_convt(x)
    else:
      y_expected = tf.nn.conv2d_transpose(
          x, tf_kernel, output_shape=output_shape,
          strides=strides_, padding=padding, dilations=dilations)

    conv_fn = self.make_conv_fn(filter_shape, strides, padding, dilations)
    with tf.GradientTape() as tape:
      tape.watch([x, k])
      y_actual = conv_fn(x, k)
    grad = tape.gradient(y_actual, [x, k])
    self.assertAllNotNone(grad)

    [y_expected_, y_actual_] = self.evaluate([y_expected, y_actual])
    self.assertAllClose(y_expected_, y_actual_, rtol=1e-5, atol=0)

  @parameterized.parameters(
      ((1,), ()),           # scalar input batch, scalar kernel batch
      ((1,), (2, 3)),       # non-scalar kernel batch
      ((3, 4), ()),         # non-scalar input batch
      ((3, 1), (2,)),       # broadcasting kernel and input batch shapes
      ((2, 3), (2, 3),))    # same kernel and input batch shapes
  def test_batching(self, input_batch_shape, kernel_batch_shape):
    input_shape = (12, 12, 2)
    filter_shape = (3, 3)
    channels_out = 4
    strides = 2
    dilations = (1, 1)
    padding = 'SAME'

    x, k = _make_input_and_kernel(
        self.make_input,
        input_batch_shape=input_batch_shape,
        input_shape=input_shape,
        kernel_batch_shape=kernel_batch_shape,
        filter_shape=filter_shape,
        channels_out=channels_out,
        dtype=self.dtype)

    conv_fn = self.make_conv_fn(filter_shape, strides, padding, dilations)
    y_batched = conv_fn(x, k)

    broadcast_batch_shape = ps.broadcast_shape(
        input_batch_shape, kernel_batch_shape)
    broadcasted_input = tf.broadcast_to(
        x, shape=ps.concat([broadcast_batch_shape, input_shape], axis=0))
    broadcasted_kernel = tf.broadcast_to(
        k, shape=ps.concat([broadcast_batch_shape, ps.shape(k)[-2:]], axis=0))

    flat_y = tf.reshape(
        y_batched,
        shape=ps.pad(
            ps.shape(y_batched)[-3:], paddings=[[1, 0]], constant_values=-1))
    flat_x = tf.reshape(
        broadcasted_input,
        shape=ps.pad(input_shape, paddings=[[1, 0]], constant_values=-1))
    flat_tf_kernel = tf.einsum(
        '...ij->...ji',
        tf.reshape(
            broadcasted_kernel,
            shape=ps.concat(
                [(-1,), filter_shape, (input_shape[-1], channels_out)],
                axis=0)))

    rank = 2
    output_shape, strides_ = convolution_util._get_output_shape(
        rank=rank, strides=(strides,) * rank, padding=padding,
        dilations=dilations, input_shape=input_shape, output_size=channels_out,
        filter_shape=filter_shape)

    y_expected = tf.vectorized_map(
        lambda args: tf.nn.conv2d_transpose(  # pylint: disable=g-long-lambda
            args[0][tf.newaxis],
            args[1],
            output_shape=ps.concat([[1], output_shape], axis=0),
            strides=strides_,
            padding=padding),
        elems=(flat_x, flat_tf_kernel))

    [y_actual_, y_expected_] = self.evaluate(
        [flat_y, tf.squeeze(y_expected, axis=1)])
    self.assertAllClose(y_expected_, y_actual_, rtol=1e-5, atol=0)

  def test_incompatible_shapes_raises(self):
    filter_shape = (3, 3)

    # Inconsistent channels in for kernel and image.
    c_in_kernel = 6
    c_in_image = 8
    c_out = 12

    k_dim = np.prod(filter_shape) * c_in_kernel
    kernel = self.make_input(tf.ones((2, k_dim, c_out), dtype=self.dtype))
    x = self.make_input(tf.ones((3, 2, 16, 16, c_in_image), dtype=self.dtype))
    conv_fn = self.make_conv_fn(
        filter_shape, strides=1, padding='SAME', dilations=1)

    with self.assertRaisesMaybeStaticError('size of the rightmost dimension'):
      self.evaluate(conv_fn(x, kernel))

  def test_dtype(self):
    # Test int64 indices.
    conv_fn = self.make_conv_fn((2, 2), strides=1, padding='SAME', dilations=1)
    x = tf.ones((2, 8, 8, 2), dtype=tf.float32)
    kernel = tf.ones((2, 8, 2), dtype=tf.float32)
    _ = self.evaluate(conv_fn(x, kernel))

    # Test f64 input.
    conv_fn = self.make_conv_fn((2, 2), strides=1, padding='SAME', dilations=1)
    x = tf.ones((2, 8, 8, 2), dtype=tf.float64)
    kernel = tf.ones((2, 8, 2), dtype=tf.float64)
    y = self.evaluate(conv_fn(x, kernel))
    self.assertDTypeEqual(y, np.float64)


@test_util.test_all_tf_execution_regimes
class BatchedConvStaticTest(_BatchedConvTest):

  dtype = tf.float32
  use_static_shape = True


@test_util.test_all_tf_execution_regimes
class BatchedConvDynamicTest(_BatchedConvTest):

  dtype = tf.float32
  use_static_shape = False


@test_util.test_all_tf_execution_regimes
class BatchedConvTransposeWithDilationsStaticTest(_BatchedConvTransposeTest):

  method = 'dilation'
  dtype = tf.float32
  use_static_shape = True


@test_util.test_all_tf_execution_regimes
class BatchedConvTransposeWithSubkernelsMatrixStaticTest(
    _BatchedConvTransposeTest):

  method = 'subkernels_matrix'
  dtype = tf.float32
  use_static_shape = True
  unequal_strides_ok = False


@test_util.test_all_tf_execution_regimes
class BatchedConvTransposeWithSubkernelsStaticTest(_BatchedConvTransposeTest):

  method = 'subkernels'
  dtype = tf.float32
  use_static_shape = True


@test_util.test_all_tf_execution_regimes
class BatchedConvTransposeWithDilationsDynamicTest(_BatchedConvTransposeTest):

  method = 'dilation'
  dtype = tf.float32
  use_static_shape = False


@test_util.test_all_tf_execution_regimes
class BatchedConvTransposeWithSubkernelsMatrixDynamicTest(
    _BatchedConvTransposeTest):

  method = 'subkernels_matrix'
  dtype = tf.float32
  use_static_shape = False
  dynamic_strides_ok = False
  unequal_strides_ok = False


@test_util.test_all_tf_execution_regimes
class BatchedConvTransposeWithSubkernelsDynamicTest(_BatchedConvTransposeTest):

  method = 'subkernels'
  dtype = tf.float32
  use_static_shape = False


del _BatchedConvTest
del _BatchedConvTransposeTest


if __name__ == '__main__':
  tf.test.main()
