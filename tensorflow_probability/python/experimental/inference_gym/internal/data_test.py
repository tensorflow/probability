# Lint as: python2, python3
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
# See the License for the modelific language governing permissions and
# limitations under the License.
# ============================================================================
"""Tests for inference_gym.targets.logistic_regression."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.experimental.inference_gym.internal import data
from tensorflow_probability.python.experimental.inference_gym.internal import test_util


class DataTest(test_util.InferenceGymTestCase):

  @test_util.uses_tfds
  def testGermanCreditNumeric(self):
    dataset = data.german_credit_numeric(train_fraction=0.75)
    self.assertEqual((750, 24), dataset['train_features'].shape)
    self.assertEqual((750,), dataset['train_labels'].shape)
    self.assertEqual((250, 24), dataset['test_features'].shape)
    self.assertEqual((250,), dataset['test_labels'].shape)
    self.assertAllClose(
        np.zeros([24]), dataset['train_features'].mean(0), atol=1e-5)
    self.assertAllClose(
        np.ones([24]), dataset['train_features'].std(0), atol=1e-5)
    self.assertAllClose(
        np.zeros([24]), dataset['test_features'].mean(0), atol=0.3)
    self.assertAllClose(
        np.ones([24]), dataset['test_features'].std(0), atol=0.3)


if __name__ == '__main__':
  tf.test.main()
