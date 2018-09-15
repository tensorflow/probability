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
"""Tests for linear algebra."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.python.framework import test_util


class _MatvecmulTest(object):

  def testMultiplicationWorks(self):
    a = self.dtype([[1., .4, .5],
                    [.4, .2, .25]])
    b = self.dtype([0.3, 0.7, 0.5])
    expected_result = np.matmul(a, b)

    a = make_tensor_hiding_attributes(a, hide_shape=self.use_dynamic_shape)
    b = make_tensor_hiding_attributes(b, hide_shape=self.use_dynamic_shape)
    result = tfp.math.matvecmul(a, b)
    result = self.evaluate(result)

    self.assertAllClose(expected_result, result)

  def testTransposedMultiplicationWorks(self):
    a = self.dtype([[1., .4, .5],
                    [.4, .2, .25]])
    b = self.dtype([0.3, 0.7])
    expected_result = np.matmul(np.transpose(a), b)

    a = make_tensor_hiding_attributes(a, hide_shape=self.use_dynamic_shape)
    b = make_tensor_hiding_attributes(b, hide_shape=self.use_dynamic_shape)
    result = tfp.math.matvecmul(a, b, transpose_a=True)
    result = self.evaluate(result)

    self.assertAllClose(expected_result, result)

  def testBatchedMultiplicationWorks(self):
    a = self.dtype([[[1., .4, .5],
                     [.4, .2, .25]],
                    [[2, .3, .4],
                     [.5, .6, .7]]])
    b = self.dtype([[0.3, 0.7, 0.5],
                    [1.0, 2.1, 3.2]])
    expected_result = np.stack([np.matmul(a[0, ...], b[0, ...]),
                                np.matmul(a[1, ...], b[1, ...])])

    a = make_tensor_hiding_attributes(a, hide_shape=self.use_dynamic_shape)
    b = make_tensor_hiding_attributes(b, hide_shape=self.use_dynamic_shape)
    result = tfp.math.matvecmul(a, b)
    result = self.evaluate(result)

    self.assertAllClose(expected_result, result)

  def testLiteralsWithMismatchedDtypes(self):
    a = np.array([[1, 2], [3, 4]], np.float64)
    b = [1., 1.]
    expected_result = np.matmul(a, b)

    result = tfp.math.matvecmul(a, b)
    result = self.evaluate(result)

    self.assertAllClose(expected_result, result)

  def testMismatchedRanksFails(self):
    data_1d = self.dtype([0.3, 0.7])
    data_2d = self.dtype([[1., .4],
                          [.4, .2]])
    data_3d = self.dtype([[[1., .4],
                           [.2, .5]]])

    data_1d = make_tensor_hiding_attributes(data_1d,
                                            hide_shape=self.use_dynamic_shape)
    data_2d = make_tensor_hiding_attributes(data_2d,
                                            hide_shape=self.use_dynamic_shape)
    data_3d = make_tensor_hiding_attributes(data_3d,
                                            hide_shape=self.use_dynamic_shape)

    with self.assertRaisesRegexp(Exception, 'similarly batched'):
      self.evaluate(tfp.math.matvecmul(data_2d, data_3d,
                                       validate_args=self.use_dynamic_shape))
    with self.assertRaisesRegexp(Exception, 'similarly batched'):
      self.evaluate(tfp.math.matvecmul(data_2d, data_2d,
                                       validate_args=self.use_dynamic_shape))
    with self.assertRaisesRegexp(Exception, 'similarly batched'):
      self.evaluate(tfp.math.matvecmul(data_3d, data_3d,
                                       validate_args=self.use_dynamic_shape))
    with self.assertRaisesRegexp(Exception, 'similarly batched'):
      self.evaluate(tfp.math.matvecmul(data_3d, data_1d,
                                       validate_args=self.use_dynamic_shape))


@test_util.run_all_in_graph_and_eager_modes
class MatvecmulTestStatic32(tf.test.TestCase, _MatvecmulTest):
  dtype = np.float32
  use_dynamic_shape = False


@test_util.run_all_in_graph_and_eager_modes
class MatvecmulTestDynamic32(tf.test.TestCase, _MatvecmulTest):
  dtype = np.float32
  use_dynamic_shape = True


@test_util.run_all_in_graph_and_eager_modes
class MatvecmulTestStatic64(tf.test.TestCase, _MatvecmulTest):
  dtype = np.float64
  use_dynamic_shape = False


@test_util.run_all_in_graph_and_eager_modes
class MatvecmulTestDynamic64(tf.test.TestCase, _MatvecmulTest):
  dtype = np.float64
  use_dynamic_shape = True


class _PinvTest(object):

  def expected_pinv(self, a, rcond):
    """Calls `np.linalg.pinv` but corrects its broken batch semantics."""
    if a.ndim < 3:
      return np.linalg.pinv(a, rcond)
    if rcond is None:
      rcond = 10. * max(a.shape[-2], a.shape[-1]) * np.finfo(a.dtype).eps
    s = np.concatenate([a.shape[:-2], [a.shape[-1], a.shape[-2]]])
    a_pinv = np.zeros(s, dtype=a.dtype)
    for i in np.ndindex(a.shape[:(a.ndim - 2)]):
      a_pinv[i] = np.linalg.pinv(
          a[i],
          rcond=rcond if isinstance(rcond, float) else rcond[i])
    return a_pinv

  def test_symmetric(self):
    a_ = self.dtype([[1., .4, .5],
                     [.4, .2, .25],
                     [.5, .25, .35]])
    a_ = np.stack([a_ + 1., a_], axis=0)  # Batch of matrices.
    a = tf.placeholder_with_default(
        input=a_,
        shape=a_.shape if self.use_static_shape else None)
    if self.use_default_rcond:
      rcond = None
    else:
      rcond = self.dtype([0., 0.01])  # Smallest 1 component is forced to zero.
    expected_a_pinv_ = self.expected_pinv(a_, rcond)
    a_pinv = tfp.math.pinv(a, rcond, validate_args=True)
    a_pinv_ = self.evaluate(a_pinv)
    self.assertAllClose(expected_a_pinv_, a_pinv_,
                        atol=1e-5, rtol=1e-5)
    if not self.use_static_shape:
      return
    self.assertAllEqual(expected_a_pinv_.shape, a_pinv.shape)

  def test_nonsquare(self):
    a_ = self.dtype([[1., .4, .5, 1.],
                     [.4, .2, .25, 2.],
                     [.5, .25, .35, 3.]])
    a_ = np.stack([a_ + 0.5, a_], axis=0)  # Batch of matrices.
    a = tf.placeholder_with_default(
        input=a_,
        shape=a_.shape if self.use_static_shape else None)
    if self.use_default_rcond:
      rcond = None
    else:
      # Smallest 2 components are forced to zero.
      rcond = self.dtype([0., 0.25])
    expected_a_pinv_ = self.expected_pinv(a_, rcond)
    a_pinv = tfp.math.pinv(a, rcond, validate_args=True)
    a_pinv_ = self.evaluate(a_pinv)
    self.assertAllClose(expected_a_pinv_, a_pinv_,
                        atol=1e-5, rtol=1e-4)
    if not self.use_static_shape:
      return
    self.assertAllEqual(expected_a_pinv_.shape, a_pinv.shape)


@test_util.run_all_in_graph_and_eager_modes
class PinvTestDynamic32DefaultRcond(tf.test.TestCase, _PinvTest):
  dtype = np.float32
  use_static_shape = False
  use_default_rcond = True


@test_util.run_all_in_graph_and_eager_modes
class PinvTestStatic64DefaultRcond(tf.test.TestCase, _PinvTest):
  dtype = np.float64
  use_static_shape = True
  use_default_rcond = True


@test_util.run_all_in_graph_and_eager_modes
class PinvTestDynamic32CustomtRcond(tf.test.TestCase, _PinvTest):
  dtype = np.float32
  use_static_shape = False
  use_default_rcond = False


@test_util.run_all_in_graph_and_eager_modes
class PinvTestStatic64CustomRcond(tf.test.TestCase, _PinvTest):
  dtype = np.float64
  use_static_shape = True
  use_default_rcond = False


def make_tensor_hiding_attributes(value, hide_shape, hide_value=True):
  if not hide_value:
    return tf.convert_to_tensor(value)

  shape = None if hide_shape else getattr(value, 'shape', None)
  return tf.placeholder_with_default(input=value, shape=shape)


if __name__ == '__main__':
  tf.test.main()
