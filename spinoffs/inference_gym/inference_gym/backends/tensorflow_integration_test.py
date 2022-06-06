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
"""Integration test for the TensorFlow backend."""

from absl.testing import absltest
import tensorflow.compat.v2 as tf
from inference_gym import using_tensorflow as gym

tf.enable_v2_behavior()


class TensorFlowIntegrationTest(absltest.TestCase):

  def testBasic(self):
    """It should be possible for this test to pass with only TF installed."""
    model = gym.targets.Banana()
    self.assertAlmostEqual(-8.640462875, model.unnormalized_log_prob([0., 0.]))


if __name__ == '__main__':
  absltest.main()
