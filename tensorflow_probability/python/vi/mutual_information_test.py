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
"""Tests for mutual information estimators and helper functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow_probability.python.internal import test_case
from tensorflow_probability.python.internal import test_util as tfp_test_util
from tensorflow_probability.python.vi import mutual_information as mi


class MutualInformationTest(test_case.TestCase):

  def setUp(self):
    super(MutualInformationTest, self).setUp()
    self.seed = tfp_test_util.test_seed()
    np.random.seed(self.seed)

  def test_lower_bound_barber_agakov(self):
    test_logits = np.float32(np.random.normal(scale=5., size=[100,]))
    test_entropy = np.float32(np.random.normal(scale=10.))
    impl_estimation = mi.lower_bound_barber_agakov(
        logits=test_logits,
        entropy=test_entropy)
    numpy_estimation = np.mean(test_logits) + test_entropy
    self.assertAllClose(
        impl_estimation,
        numpy_estimation)


if __name__ == '__main__':
  tf.test.main()
