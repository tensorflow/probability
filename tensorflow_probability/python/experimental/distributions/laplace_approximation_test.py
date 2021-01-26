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

import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import test_util

tfd = tfp.distributions
tfde = tfp.experimental.distributions

ATOL = 1e-4

@test_util.test_graph_mode_only
class LaplaceApproximationTest(test_util.TestCase):

  def testJointDistributionCoroutineAutoBatchedWithDefaultArgs(self):
    @tfd.JointDistributionCoroutineAutoBatched
    def joint_dist():
      yield tfd.Normal(loc=1., scale=1.)
      yield tfd.Normal(loc=[2., 2.], scale=2.)
      yield tfd.Gamma(concentration=[3., 3., 3.], rate=10.)

    approximation = tfde.laplace_approximation(joint_dist)
    mean = approximation.bijector(approximation.distribution.mean())
    mean = self.evaluate(mean)

    self.assertEqual(len(mean), 3)

    self.assertAllClose(mean[0], 1., atol=ATOL)
    self.assertAllClose(mean[1], [2., 2.], atol=ATOL)
    self.assertAllClose(mean[2], [0.2, 0.2, 0.2], atol=ATOL)


  def testJointDistributionNamedWithDefaultArgs(self):
    joint_dist = tfd.JointDistributionNamed(dict(
        a=tfd.Normal(loc=1.5, scale=2.),
        b=lambda a: tfd.Normal(loc=a + 2.5, scale=3.),
    ))

    approximation = tfde.laplace_approximation(joint_dist)
    mvn = approximation.distribution
    mean, variance = self.evaluate([mvn.mean(), mvn.variance()])

    self.assertAllClose(mean[0], 1.5, atol=ATOL)
    self.assertAllClose(mean[1], 4., atol=ATOL)

    self.assertAllClose(variance[0], 2.**2, atol=ATOL)
    self.assertAllClose(variance[1], 2.**2 + 3.**2, atol=ATOL)


  def testJointDistributionSequentialWithDefaultArgs(self):
    joint_dist = tfd.JointDistributionSequential([
        tfd.MultivariateNormalDiag(loc=0.123, scale_diag=[1., 2., 3.]),
        tfd.StudentT(loc=2.5, scale=3., df=3.),
    ])

    approximation = tfde.laplace_approximation(joint_dist)
    mean = approximation.bijector(approximation.distribution.mean())
    mean = self.evaluate(mean)

    self.assertAllClose(mean[0], np.full([3], 0.123), atol=ATOL)
    self.assertAllClose(mean[1], 2.5, atol=ATOL)


if __name__ == '__main__':
  tf.test.main()
