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
"""Tests for generating random samples."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np

import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import test_util


class _RandomRademacher(object):

  def test_expected_value(self):
    shape_ = np.array([2, 3, int(1e3)], np.int32)
    shape = (
        tf.constant(shape_) if self.use_static_shape else
        tf1.placeholder_with_default(shape_, shape=None))
    x = tfp.math.random_rademacher(shape, self.dtype,
                                   seed=test_util.test_seed())
    if self.use_static_shape:
      self.assertAllEqual(shape_, x.shape)
    x_ = self.evaluate(x)
    self.assertEqual(self.dtype, dtype_util.as_numpy_dtype(x.dtype))
    self.assertAllEqual(shape_, x_.shape)
    self.assertAllEqual([-1., 1], np.unique(np.reshape(x_, [-1])))
    self.assertAllClose(
        np.zeros(shape_[:-1]),
        np.mean(x_, axis=-1),
        atol=0.07, rtol=0.)


@test_util.test_all_tf_execution_regimes
class RandomRademacherDynamic32(test_util.TestCase, _RandomRademacher):
  dtype = np.float32
  use_static_shape = False


@test_util.test_all_tf_execution_regimes
class RandomRademacherDynamic64(test_util.TestCase, _RandomRademacher):
  dtype = np.float64
  use_static_shape = True


class _RandomRayleigh(object):

  def test_expected_value(self):
    shape_ = np.array([2, int(1e3)], np.int32)
    shape = (
        tf.constant(shape_) if self.use_static_shape else
        tf1.placeholder_with_default(shape_, shape=None))
    # This shape will require broadcasting before sampling.
    scale_ = np.linspace(0.1, 0.5, 3 * 2).astype(self.dtype).reshape(3, 2)
    scale = (
        tf.constant(scale_) if self.use_static_shape else
        tf1.placeholder_with_default(scale_, shape=None))
    x = tfp.math.random_rayleigh(shape,
                                 scale=scale[..., tf.newaxis],
                                 dtype=self.dtype,
                                 seed=test_util.test_seed())
    self.assertEqual(self.dtype, dtype_util.as_numpy_dtype(x.dtype))
    final_shape_ = [3, 2, int(1e3)]
    if self.use_static_shape:
      self.assertAllEqual(final_shape_, x.shape)
    sample_mean = tf.reduce_mean(x, axis=-1, keepdims=True)
    sample_var = tf.reduce_mean(
        tf.math.squared_difference(x, sample_mean), axis=-1)
    [x_, sample_mean_, sample_var_] = self.evaluate([
        x, sample_mean[..., 0], sample_var])
    self.assertAllEqual(final_shape_, x_.shape)
    self.assertAllEqual(np.ones_like(x_, dtype=np.bool), x_ > 0.)
    self.assertAllClose(np.sqrt(np.pi / 2.) * scale_, sample_mean_,
                        atol=0.05, rtol=0.)
    self.assertAllClose(0.5 * (4. - np.pi) * scale_**2., sample_var_,
                        atol=0.05, rtol=0.)


@test_util.test_all_tf_execution_regimes
class RandomRayleighDynamic32(test_util.TestCase, _RandomRayleigh):
  dtype = np.float32
  use_static_shape = False


@test_util.test_all_tf_execution_regimes
class RandomRayleighDynamic64(test_util.TestCase, _RandomRayleigh):
  dtype = np.float64
  use_static_shape = True


if __name__ == '__main__':
  tf.test.main()
