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
"""Tests for VectorDiffeomixture."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.internal import test_util

rng = np.random.RandomState(0)


@test_util.test_all_tf_execution_regimes
class VectorDiffeomixtureTest(test_util.VectorDistributionTestHelpers,
                              test_util.TestCase):
  """Tests the VectorDiffeomixture distribution."""

  def testSampleProbConsistentBroadcastMixNoBatch(self):
    dims = 4
    vdm = tfd.VectorDiffeomixture(
        mix_loc=[[0.], [1.]],
        temperature=[1.],
        distribution=tfd.Normal(0., 1.),
        loc=[
            None,
            np.float32([2.] * dims),
        ],
        scale=[
            tf.linalg.LinearOperatorScaledIdentity(
                num_rows=dims,
                multiplier=np.float32(1.1),
                is_positive_definite=True),
            tf.linalg.LinearOperatorDiag(
                diag=np.linspace(2.5, 3.5, dims, dtype=np.float32),
                is_positive_definite=True),
        ],
        quadrature_size=8,
        validate_args=True)
    # Ball centered at component0's mean.
    self.run_test_sample_consistent_log_prob(
        self.evaluate,
        vdm,
        num_samples=int(3e5),
        radius=2.,
        center=0.,
        rtol=0.025)
    # Larger ball centered at component1's mean.
    self.run_test_sample_consistent_log_prob(
        self.evaluate,
        vdm,
        num_samples=int(3e5),
        radius=4.,
        center=2.,
        rtol=0.025)

  def testSampleProbConsistentBroadcastMixNonStandardBase(self):
    dims = 4
    vdm = tfd.VectorDiffeomixture(
        mix_loc=[[0.], [1.]],
        temperature=[1.],
        distribution=tfd.Normal(1., 1.5),
        loc=[
            None,
            np.float32([2.] * dims),
        ],
        scale=[
            tf.linalg.LinearOperatorScaledIdentity(
                num_rows=dims,
                multiplier=np.float32(1.1),
                is_positive_definite=True),
            tf.linalg.LinearOperatorDiag(
                diag=np.linspace(2.5, 3.5, dims, dtype=np.float32),
                is_positive_definite=True),
        ],
        quadrature_size=8,
        validate_args=True)
    # Ball centered at component0's mean.
    self.run_test_sample_consistent_log_prob(
        self.evaluate,
        vdm,
        num_samples=int(5e5),
        radius=2.,
        center=1.,
        rtol=0.05)
    # Larger ball centered at component1's mean.
    self.run_test_sample_consistent_log_prob(
        self.evaluate, vdm, radius=4., center=3., rtol=0.025)

  def testSampleProbConsistentBroadcastMixBatch(self):
    dims = 4
    vdm = tfd.VectorDiffeomixture(
        mix_loc=[[0.], [1.]],
        temperature=[1.],
        distribution=tfd.Normal(0., 1.),
        loc=[
            None,
            np.float32([2.] * dims),
        ],
        scale=[
            tf.linalg.LinearOperatorScaledIdentity(
                num_rows=dims,
                multiplier=[np.float32(1.1)],
                is_positive_definite=True),
            tf.linalg.LinearOperatorDiag(
                diag=np.stack([
                    np.linspace(2.5, 3.5, dims, dtype=np.float32),
                    np.linspace(2.75, 3.25, dims, dtype=np.float32),
                ]),
                is_positive_definite=True),
        ],
        quadrature_size=8,
        validate_args=True)
    # Ball centered at component0's mean.
    self.run_test_sample_consistent_log_prob(
        self.evaluate,
        vdm,
        num_samples=int(3e5),
        radius=2.,
        center=0.,
        rtol=0.025)
    # Larger ball centered at component1's mean.
    self.run_test_sample_consistent_log_prob(
        self.evaluate,
        vdm,
        num_samples=int(3e5),
        radius=4.,
        center=2.,
        rtol=0.025)

  def testSampleProbConsistentBroadcastMixTwoBatchDims(self):
    dims = 4
    loc_1 = rng.randn(2, 3, dims).astype(np.float32)

    vdm = tfd.VectorDiffeomixture(
        mix_loc=(rng.rand(2, 3, 1) - 0.5).astype(np.float32),
        temperature=[1.],
        distribution=tfd.Normal(0., 1.),
        loc=[
            None,
            loc_1,
        ],
        scale=[
            tf.linalg.LinearOperatorScaledIdentity(
                num_rows=dims,
                multiplier=[np.float32(1.1)],
                is_positive_definite=True),
        ] * 2,
        validate_args=True)
    # Ball centered at component0's mean.
    self.run_test_sample_consistent_log_prob(
        self.evaluate,
        vdm,
        num_samples=int(3e5),
        radius=2.,
        center=0.,
        rtol=0.02)
    # Larger ball centered at component1's mean.
    self.run_test_sample_consistent_log_prob(
        self.evaluate,
        vdm,
        num_samples=int(3e5),
        radius=3.,
        center=loc_1,
        rtol=0.025)

  def testMeanCovarianceNoBatch(self):
    dims = 3
    vdm = tfd.VectorDiffeomixture(
        mix_loc=[[0.], [4.]],
        temperature=[1 / 10.],
        distribution=tfd.Normal(0., 1.),
        loc=[
            np.float32([-2.]),
            None,
        ],
        scale=[
            tf.linalg.LinearOperatorScaledIdentity(
                num_rows=dims,
                multiplier=np.float32(1.5),
                is_positive_definite=True),
            tf.linalg.LinearOperatorDiag(
                diag=np.linspace(2.5, 3.5, dims, dtype=np.float32),
                is_positive_definite=True),
        ],
        quadrature_size=8,
        validate_args=True)
    self.run_test_sample_consistent_mean_covariance(
        self.evaluate, vdm, rtol=0.02, cov_rtol=0.08)

  def testTemperatureControlsHowMuchThisLooksLikeDiscreteMixture(self):
    # As temperature decreases, this should approach a mixture of normals, with
    # components at -2, 2.
    dims = 1
    vdm = tfd.VectorDiffeomixture(
        mix_loc=[0.],
        temperature=[[2.], [1.], [0.2]],
        distribution=tfd.Normal(0., 1.),
        loc=[
            np.float32([-2.]),
            np.float32([2.]),
        ],
        scale=[
            tf.linalg.LinearOperatorScaledIdentity(
                num_rows=dims,
                multiplier=np.float32(0.5),
                is_positive_definite=True),
        ] * 2,  # Use the same scale for each component.
        quadrature_size=8,
        validate_args=True)

    samps = vdm.sample(10000, seed=test_util.test_seed())
    self.assertAllEqual((10000, 3, 1), samps.shape)
    samps_ = self.evaluate(samps).reshape(10000, 3)  # Make scalar event shape.

    # One characteristic of a discrete mixture (as opposed to a "smear") is
    # that more weight is put near the component centers at -2, 2, and thus
    # less weight is put near the origin.
    prob_of_being_near_origin = (np.abs(samps_) < 1).mean(axis=0)
    self.assertGreater(prob_of_being_near_origin[0],
                       prob_of_being_near_origin[1])
    self.assertGreater(prob_of_being_near_origin[1],
                       prob_of_being_near_origin[2])

    # Run this test as well, just because we can.
    self.run_test_sample_consistent_mean_covariance(
        self.evaluate, vdm, rtol=0.02, cov_rtol=0.08)

  def testConcentrationLocControlsHowMuchWeightIsOnEachComponent(self):
    dims = 1
    vdm = tfd.VectorDiffeomixture(
        mix_loc=[[-1.], [0.], [1.]],
        temperature=[0.5],
        distribution=tfd.Normal(0., 1.),
        loc=[
            np.float32([-2.]),
            np.float32([2.]),
        ],
        scale=[
            tf.linalg.LinearOperatorScaledIdentity(
                num_rows=dims,
                multiplier=np.float32(0.5),
                is_positive_definite=True),
        ] * 2,  # Use the same scale for each component.
        quadrature_size=8,
        validate_args=True)

    samps = vdm.sample(10000, seed=test_util.test_seed())
    self.assertAllEqual((10000, 3, 1), samps.shape)
    samps_ = self.evaluate(samps).reshape(10000, 3)  # Make scalar event shape.

    # One characteristic of putting more weight on a component is that the
    # mean is closer to that component's mean.
    # Get the mean for each batch member, the names signify the value of
    # concentration for that batch member.
    mean_neg1, mean_0, mean_1 = samps_.mean(axis=0)

    # Since concentration is the concentration for component 0,
    # concentration = -1 ==> more weight on component 1, which has mean = 2
    # concentration = 0 ==> equal weight
    # concentration = 1 ==> more weight on component 0, which has mean = -2
    self.assertLess(-2, mean_1)
    self.assertLess(mean_1, mean_0)
    self.assertLess(mean_0, mean_neg1)
    self.assertLess(mean_neg1, 2)

    # Run this test as well, just because we can.
    self.run_test_sample_consistent_mean_covariance(
        self.evaluate, vdm, rtol=0.02, cov_rtol=0.08)

  def testMeanCovarianceNoBatchUncenteredNonStandardBase(self):
    dims = 3
    vdm = tfd.VectorDiffeomixture(
        mix_loc=[[0.], [4.]],
        temperature=[0.1],
        distribution=tfd.Normal(-1., 1.5),
        loc=[
            np.float32([-2.]),
            np.float32([0.]),
        ],
        scale=[
            tf.linalg.LinearOperatorScaledIdentity(
                num_rows=dims,
                multiplier=np.float32(1.5),
                is_positive_definite=True),
            tf.linalg.LinearOperatorDiag(
                diag=np.linspace(2.5, 3.5, dims, dtype=np.float32),
                is_positive_definite=True),
        ],
        quadrature_size=8,
        validate_args=True)
    self.run_test_sample_consistent_mean_covariance(
        self.evaluate, vdm, num_samples=int(1e6), rtol=0.01,
        cov_atol=0.05)

  def testMeanCovarianceBatch(self):
    dims = 3
    vdm = tfd.VectorDiffeomixture(
        mix_loc=[[0.], [4.]],
        temperature=[0.1],
        distribution=tfd.Normal(0., 1.),
        loc=[
            np.float32([[-2.]]),
            None,
        ],
        scale=[
            tf.linalg.LinearOperatorScaledIdentity(
                num_rows=dims,
                multiplier=[np.float32(1.5)],
                is_positive_definite=True),
            tf.linalg.LinearOperatorDiag(
                diag=np.stack([
                    np.linspace(2.5, 3.5, dims, dtype=np.float32),
                    np.linspace(0.5, 1.5, dims, dtype=np.float32),
                ]),
                is_positive_definite=True),
        ],
        quadrature_size=8,
        validate_args=True)
    self.run_test_sample_consistent_mean_covariance(
        self.evaluate, vdm, rtol=0.02, cov_rtol=0.07)

  def testSampleProbConsistentQuadrature(self):
    dims = 4
    vdm = tfd.VectorDiffeomixture(
        mix_loc=[0.],
        temperature=[0.1],
        distribution=tfd.Normal(0., 1.),
        loc=[
            None,
            np.float32([2.] * dims),
        ],
        scale=[
            tf.linalg.LinearOperatorScaledIdentity(
                num_rows=dims,
                multiplier=np.float32(1.1),
                is_positive_definite=True),
            tf.linalg.LinearOperatorDiag(
                diag=np.linspace(2.5, 3.5, dims, dtype=np.float32),
                is_positive_definite=True),
        ],
        quadrature_size=3,
        validate_args=True)
    # Ball centered at component0's mean.
    self.run_test_sample_consistent_log_prob(
        self.evaluate,
        vdm,
        num_samples=int(3e5),
        radius=2.,
        center=0.,
        rtol=0.015)
    # Larger ball centered at component1's mean.
    self.run_test_sample_consistent_log_prob(
        self.evaluate,
        vdm,
        num_samples=int(3e5),
        radius=4.,
        center=2.,
        rtol=0.02)


if __name__ == "__main__":
  tf.test.main()
