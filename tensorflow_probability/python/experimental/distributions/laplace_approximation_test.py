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
"""Tests for Laplace approximation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import test_util

tfd = tfp.distributions
tfde = tfp.experimental.distributions


@test_util.test_graph_mode_only
class LaplaceApproximationTest(test_util.TestCase):

  def testJointDistributionCoroutineAutoBatched(self):
    @tfd.JointDistributionCoroutineAutoBatched
    def joint_dist():
      yield tfd.Normal(loc=1.0, scale=1.0, name="a")
      yield tfd.Normal(loc=[2.0, 2.0], scale=2.0, name="b")
      yield tfd.Gamma(concentration=[3.0, 3.0, 3.0], rate=10.0, name="c")

    # without conditioning
    approximation = tfde.laplace_approximation(joint_dist)
    mode = approximation.bijector(approximation.distribution.mode())

    atol = 1e-4
    self.assertAllClose(mode[0], 1.0, atol=atol)
    self.assertAllClose(mode[1], [2.0, 2.0], atol=atol)
    self.assertAllClose(mode[2], [0.2, 0.2, 0.2], atol=atol)


if __name__ == '__main__':
  tf.test.main()
