# Copyright 2023 The TensorFlow Probability Authors.
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
"""Tests for tensorflow_probability internal prefer_static with int64 shape."""

import numpy as np

import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import test_util


class ShapeTestInt64(test_util.TestCase):

  def test_shape(self):
    vector_value = [0., 1.]

    # case: numpy input
    self.assertAllEqual(ps.shape(np.array(vector_value)), [2])

    # case: tensor input with static shape
    self.assertAllEqual(ps.shape(tf.constant(vector_value)), [2])

    # case: tensor input with dynamic shape
    if not tf.executing_eagerly():
      shape = ps.shape(tf1.placeholder_with_default(vector_value, shape=None))
      self.assertAllEqual(self.evaluate(shape), [2])

    # case: tensor input with static shape (with
    # TF_FLAG_TF_SHAPE_DEFAULT_INT64=true which is set by an environment
    # variable when this test is run.)
    self.assertEqual(ps.shape(tf.constant(vector_value)).dtype, tf.int64)

if __name__ == '__main__':
  test_util.main()
