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

import tensorflow as tf

from tensorflow_probability.python.math import random_rademacher
from tensorflow.python.framework import test_util


class _RandomRademacher(object):

  @test_util.run_in_graph_and_eager_modes()
  def test_expected_value(self):
    shape_ = np.array([2, 3, int(1e3)], np.int32)
    shape = (tf.constant(shape_) if self.use_static_shape
             else tf.placeholder_with_default(shape_, shape=None))
    x = random_rademacher(shape, self.dtype, seed=42)
    if self.use_static_shape:
      self.assertAllEqual(shape_, x.shape)
    x_ = self.evaluate(x)
    self.assertEqual(self.dtype, x.dtype.as_numpy_dtype)
    self.assertAllEqual(shape_, x_.shape)
    self.assertAllEqual([-1., 1], np.unique(np.reshape(x_, [-1])))
    self.assertAllClose(
        np.zeros(shape_[:-1]),
        np.mean(x_, axis=-1),
        atol=0.05, rtol=0.)


class RandomRademacherDynamic32(tf.test.TestCase, _RandomRademacher):
  dtype = np.float32
  use_static_shape = False


class RandomRademacherDynamic64(tf.test.TestCase, _RandomRademacher):
  dtype = np.float64
  use_static_shape = True


if __name__ == '__main__':
  tf.test.main()
