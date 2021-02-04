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

import collections
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import test_util

tfd = tfp.distributions
tfb = tfp.bijectors
tfde = tfp.experimental.distributions

ATOL = 1e-4

@test_util.test_graph_mode_only
class LaplaceApproximationTest(test_util.TestCase):

  def testJointDistributionCoroutineAutoBatchedWithDefaults(self):
    @tfd.JointDistributionCoroutineAutoBatched
    def joint_dist():
      yield tfd.Normal(loc=1., scale=1.)
      yield tfd.Normal(loc=[2., 2.], scale=2.)
      yield tfd.Gamma(concentration=[3., 3., 3.], rate=10.)

    approx = tfde.laplace_approximation(joint_dist)
    mean = approx.bijector(approx.distribution.mean())
    a, b, c = self.evaluate(mean)

    self.assertEqual(len(mean), 3)

    self.assertAllClose(a, 1., atol=ATOL)
    self.assertAllClose(b, [2., 2.], atol=ATOL)
    # this is the mode of the gamma distribution, not the mean
    self.assertAllClose(c, [0.2, 0.2, 0.2], atol=ATOL)

  def testJointDistributionCoroutineWithData(self):
    Root = tfd.JointDistributionCoroutine.Root

    @tfd.JointDistributionCoroutine
    def joint_dist():
      a = yield Root(tfd.Normal(loc=1., scale=1., name="a"))
      yield tfd.MultivariateNormalDiag(loc=a + [2., 2.], scale_diag=[2., 2.])

    approx = tfde.laplace_approximation(joint_dist, data={"a": 5.})
    mean = self.evaluate(approx.distribution.mean())

    self.assertAllClose(mean, [7., 7.], atol=ATOL)


  def testJointDistributionNamedWithDefaults(self):
    joint_dist = tfd.JointDistributionNamed(collections.OrderedDict(
        a=tfd.Normal(loc=1.5, scale=2.),
        b=lambda a: tfd.Normal(loc=a + 2.5, scale=3.),
    ))

    approx = tfde.laplace_approximation(joint_dist)
    mvn = approx.distribution
    mean, variance = self.evaluate([mvn.mean(), mvn.variance()])

    self.assertAllClose(mean[0], 1.5, atol=ATOL)
    self.assertAllClose(mean[1], 4., atol=ATOL)

    self.assertAllClose(variance[0], 2.**2, atol=ATOL)
    self.assertAllClose(variance[1], 2.**2 + 3.**2, atol=ATOL)


  def testJointDistributionNamedWithBijector(self):
    joint_dist = tfd.JointDistributionNamed(collections.OrderedDict(
        a=tfd.Normal(loc=1.5, scale=2.),
        b=tfd.Gamma(concentration=2.5, rate=3.),
    ))

    approx = tfde.laplace_approximation(
        joint_dist, bijectors=[tfb.Identity(), tfb.Exp()])
    mean = approx.distribution.mean()
    a_mean, b_mean = self.evaluate(mean)

    # the mean of the underlying unconstrained mvn
    self.assertAllClose(a_mean, 1.5, atol=ATOL)
    self.assertAllClose(b_mean, np.log((2.5 - 1.) / 3.), atol=ATOL)


  def testJointDistributionSequentialAutoBatchedWithDefaults(self):
    joint_dist = tfd.JointDistributionSequentialAutoBatched([
        tfd.MultivariateNormalDiag(loc=0.123, scale_diag=[1., 2., 3.]),
        tfd.StudentT(loc=2.5, scale=3., df=3.),
    ])

    approx = tfde.laplace_approximation(joint_dist)
    mean = approx.bijector(approx.distribution.mean())
    mvn_mean, student_t_mean = self.evaluate(mean)

    self.assertAllClose(mvn_mean, np.full([3], 0.123), atol=ATOL)
    self.assertAllClose(student_t_mean, 2.5, atol=ATOL)

  def testJointDistributionSequentialWithInitialValues(self):
    joint_dist = tfd.JointDistributionSequential([
        tfd.Normal(loc=0., scale=1.),
        tfd.Normal(loc=2., scale=3.),
    ])

    # check single sample case
    approx = tfde.laplace_approximation(joint_dist, initial_values=[0.123, 2.42])
    mean = approx.bijector(approx.distribution.mean())
    a, b = self.evaluate(mean)

    self.assertAllClose(a, 0., atol=ATOL)
    self.assertAllClose(b, 2., atol=ATOL)

  def testShapes(self):
    @tfd.JointDistributionCoroutineAutoBatched
    def joint_dist():
      yield tfd.Normal(loc=1.0, scale=1.0, name="a")
      yield tfd.Normal(loc=[2.0, 2.0], scale=2.0, name="b")
      yield tfd.Gamma(concentration=[2.0, 2.0, 2.0], rate=10.0, name="c")
      yield tfd.CholeskyLKJ(dimension=4, concentration=1.0, name="d")

    approx = tfde.laplace_approximation(joint_dist)
    a, b, c, d = self.evaluate(approx.bijector(approx.distribution.mode()))

    self.assertEqual(a.shape, ())
    self.assertEqual(b.shape, (2,))
    self.assertEqual(c.shape, (3,))
    self.assertEqual(d.shape, (4, 4))


if __name__ == '__main__':
  tf.test.main()
