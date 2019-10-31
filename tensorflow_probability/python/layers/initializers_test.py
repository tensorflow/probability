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
"""Tests for tensorflow_probability.layers Keras initializers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.internal import test_util


@test_util.test_all_tf_execution_regimes
class BlockwiseInitializerTest(test_util.TestCase):

  def test_works_correctly(self):
    init = tfp.layers.BlockwiseInitializer(['glorot_uniform', 'zeros'], [3, 4])
    x = init([2, 1, 7])
    self.assertEqual((2, 1, 7), x.shape)
    x_ = self.evaluate(x)
    self.assertAllEqual(np.zeros([2, 1, 4]), x_[..., 3:])

  def test_de_serialization(self):
    s = tf.initializers.serialize(
        tfp.layers.BlockwiseInitializer(['glorot_uniform', 'zeros'], [3, 4]))
    init_clone = tf.initializers.deserialize(s)
    x = init_clone([2, 1, 7])
    self.assertEqual((2, 1, 7), x.shape)
    x_ = self.evaluate(x)
    self.assertAllEqual(np.zeros([2, 1, 4]), x_[..., 3:])


if __name__ == '__main__':
  tf.test.main()
