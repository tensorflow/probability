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
"""Tests for MultivariateNormal."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import test_util as tfp_test_util
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top
tfd = tfp.distributions


@test_util.run_all_in_graph_and_eager_modes
class MultivariateNormalDiagPlusLowRankTest(tf.test.TestCase):
  """Well tested because this is a simple override of the base class."""

  def setUp(self):
    self._rng = np.random.RandomState(42)

  def testDiagBroadcastBothBatchAndEvent(self):
    # batch_shape: [3], event_shape: [2]
    diag = np.array([[1., 2], [3, 4], [5, 6]])
    # batch_shape: [1], event_shape: []
    identity_multiplier = np.array([5.])
    dist = tfd.MultivariateNormalDiagPlusLowRank(
        scale_diag=diag,
        scale_identity_multiplier=identity_multiplier,
        validate_args=True)
    self.assertAllClose(
        np.array([[[1. + 5, 0], [0, 2 + 5]], [[3 + 5, 0], [0, 4 + 5]],
                  [[5 + 5, 0], [0, 6 + 5]]]),
        self.evaluate(dist.scale.to_dense()))

  def testDiagBroadcastBothBatchAndEvent2(self):
    # This test differs from `testDiagBroadcastBothBatchAndEvent` in that it
    # broadcasts batch_shape's from both the `scale_diag` and
    # `scale_identity_multiplier` args.
    # batch_shape: [3], event_shape: [2]
    diag = np.array([[1., 2], [3, 4], [5, 6]])
    # batch_shape: [3, 1], event_shape: []
    identity_multiplier = np.array([[5.], [4], [3]])
    dist = tfd.MultivariateNormalDiagPlusLowRank(
        scale_diag=diag,
        scale_identity_multiplier=identity_multiplier,
        validate_args=True)
    self.assertAllEqual([3, 3, 2, 2], dist.scale.to_dense().shape)

  def testDiagBroadcastOnlyEvent(self):
    # batch_shape: [3], event_shape: [2]
    diag = np.array([[1., 2], [3, 4], [5, 6]])
    # batch_shape: [3], event_shape: []
    identity_multiplier = np.array([5., 4, 3])
    dist = tfd.MultivariateNormalDiagPlusLowRank(
        scale_diag=diag,
        scale_identity_multiplier=identity_multiplier,
        validate_args=True)
    self.assertAllClose(
        np.array([[[1. + 5, 0], [0, 2 + 5]], [[3 + 4, 0], [0, 4 + 4]],
                  [[5 + 3, 0], [0, 6 + 3]]]),  # shape: [3, 2, 2]
        self.evaluate(dist.scale.to_dense()))

  def testDiagBroadcastMultiplierAndLoc(self):
    # batch_shape: [], event_shape: [3]
    loc = np.array([1., 0, -1])
    # batch_shape: [3], event_shape: []
    identity_multiplier = np.array([5., 4, 3])
    dist = tfd.MultivariateNormalDiagPlusLowRank(
        loc=loc,
        scale_identity_multiplier=identity_multiplier,
        validate_args=True)
    self.assertAllClose(
        np.array([[[5, 0, 0], [0, 5, 0], [0, 0, 5]],
                  [[4, 0, 0], [0, 4, 0], [0, 0, 4]], [[3, 0, 0], [0, 3, 0],
                                                      [0, 0, 3]]]),
        self.evaluate(dist.scale.to_dense()))

  def testMean(self):
    mu = [-1.0, 1.0]
    diag_large = [1.0, 5.0]
    v = [[2.0], [3.0]]
    diag_small = [3.0]
    dist = tfd.MultivariateNormalDiagPlusLowRank(
        loc=mu,
        scale_diag=diag_large,
        scale_perturb_factor=v,
        scale_perturb_diag=diag_small,
        validate_args=True)
    self.assertAllEqual(mu, self.evaluate(dist.mean()))

  def testSample(self):
    # TODO(jvdillon): This test should be the basis of a new test fixture which
    # is applied to every distribution. When we make this fixture, we'll also
    # separate the analytical- and sample-based tests as well as for each
    # function tested. For now, we group things so we can recycle one batch of
    # samples (thus saving resources).

    mu = np.array([-1., 1, 0.5], dtype=np.float32)
    diag_large = np.array([1., 0.5, 0.75], dtype=np.float32)
    diag_small = np.array([-1.1, 1.2], dtype=np.float32)
    v = np.array([[0.7, 0.8],
                  [0.9, 1],
                  [0.5, 0.6]], dtype=np.float32)  # shape: [k, r] = [3, 2]

    true_mean = mu
    true_scale = np.diag(diag_large) + np.matmul(np.matmul(
        v, np.diag(diag_small)), v.T)
    true_covariance = np.matmul(true_scale, true_scale.T)
    true_variance = np.diag(true_covariance)
    true_stddev = np.sqrt(true_variance)

    dist = tfd.MultivariateNormalDiagPlusLowRank(
        loc=mu,
        scale_diag=diag_large,
        scale_perturb_factor=v,
        scale_perturb_diag=diag_small,
        validate_args=True)

    # The following distributions will test the KL divergence calculation.
    mvn_identity = tfd.MultivariateNormalDiag(
        loc=np.array([1., 2, 0.25], dtype=np.float32), validate_args=True)
    mvn_scaled = tfd.MultivariateNormalDiag(
        loc=mvn_identity.loc, scale_identity_multiplier=2.2, validate_args=True)
    mvn_diag = tfd.MultivariateNormalDiag(
        loc=mvn_identity.loc,
        scale_diag=np.array([0.5, 1.5, 1.], dtype=np.float32),
        validate_args=True)
    mvn_chol = tfd.MultivariateNormalTriL(
        loc=np.array([1., 2, -1], dtype=np.float32),
        scale_tril=np.array(
            [[6., 0, 0], [2, 5, 0], [1, 3, 4]], dtype=np.float32) / 10.,
        validate_args=True)

    scale = dist.scale.to_dense()

    n = int(30e3)
    samps = dist.sample(
        n, seed=tfp_test_util.test_seed(hardcoded_seed=0, set_eager_seed=False))
    sample_mean = tf.reduce_mean(input_tensor=samps, axis=0)
    x = samps - sample_mean
    sample_covariance = tf.matmul(x, x, transpose_a=True) / n

    sample_kl_identity = tf.reduce_mean(
        input_tensor=dist.log_prob(samps) - mvn_identity.log_prob(samps),
        axis=0)
    analytical_kl_identity = tfd.kl_divergence(dist, mvn_identity)

    sample_kl_scaled = tf.reduce_mean(
        input_tensor=dist.log_prob(samps) - mvn_scaled.log_prob(samps), axis=0)
    analytical_kl_scaled = tfd.kl_divergence(dist, mvn_scaled)

    sample_kl_diag = tf.reduce_mean(
        input_tensor=dist.log_prob(samps) - mvn_diag.log_prob(samps), axis=0)
    analytical_kl_diag = tfd.kl_divergence(dist, mvn_diag)

    sample_kl_chol = tf.reduce_mean(
        input_tensor=dist.log_prob(samps) - mvn_chol.log_prob(samps), axis=0)
    analytical_kl_chol = tfd.kl_divergence(dist, mvn_chol)

    n = int(10e3)
    baseline = tfd.MultivariateNormalDiag(
        loc=np.array([-1., 0.25, 1.25], dtype=np.float32),
        scale_diag=np.array([1.5, 0.5, 1.], dtype=np.float32),
        validate_args=True)
    samps = baseline.sample(n, seed=tfp_test_util.test_seed())

    sample_kl_identity_diag_baseline = tf.reduce_mean(
        input_tensor=baseline.log_prob(samps) - mvn_identity.log_prob(samps),
        axis=0)
    analytical_kl_identity_diag_baseline = tfd.kl_divergence(
        baseline, mvn_identity)

    sample_kl_scaled_diag_baseline = tf.reduce_mean(
        input_tensor=baseline.log_prob(samps) - mvn_scaled.log_prob(samps),
        axis=0)
    analytical_kl_scaled_diag_baseline = tfd.kl_divergence(baseline, mvn_scaled)

    sample_kl_diag_diag_baseline = tf.reduce_mean(
        input_tensor=baseline.log_prob(samps) - mvn_diag.log_prob(samps),
        axis=0)
    analytical_kl_diag_diag_baseline = tfd.kl_divergence(baseline, mvn_diag)

    sample_kl_chol_diag_baseline = tf.reduce_mean(
        input_tensor=baseline.log_prob(samps) - mvn_chol.log_prob(samps),
        axis=0)
    analytical_kl_chol_diag_baseline = tfd.kl_divergence(baseline, mvn_chol)

    [
        sample_mean_,
        analytical_mean_,
        sample_covariance_,
        analytical_covariance_,
        analytical_variance_,
        analytical_stddev_,
        scale_,
        sample_kl_identity_,
        analytical_kl_identity_,
        sample_kl_scaled_,
        analytical_kl_scaled_,
        sample_kl_diag_,
        analytical_kl_diag_,
        sample_kl_chol_,
        analytical_kl_chol_,
        sample_kl_identity_diag_baseline_,
        analytical_kl_identity_diag_baseline_,
        sample_kl_scaled_diag_baseline_,
        analytical_kl_scaled_diag_baseline_,
        sample_kl_diag_diag_baseline_,
        analytical_kl_diag_diag_baseline_,
        sample_kl_chol_diag_baseline_,
        analytical_kl_chol_diag_baseline_,
    ] = self.evaluate([
        sample_mean,
        dist.mean(),
        sample_covariance,
        dist.covariance(),
        dist.variance(),
        dist.stddev(),
        scale,
        sample_kl_identity,
        analytical_kl_identity,
        sample_kl_scaled,
        analytical_kl_scaled,
        sample_kl_diag,
        analytical_kl_diag,
        sample_kl_chol,
        analytical_kl_chol,
        sample_kl_identity_diag_baseline,
        analytical_kl_identity_diag_baseline,
        sample_kl_scaled_diag_baseline,
        analytical_kl_scaled_diag_baseline,
        sample_kl_diag_diag_baseline,
        analytical_kl_diag_diag_baseline,
        sample_kl_chol_diag_baseline,
        analytical_kl_chol_diag_baseline,
    ])

    sample_variance_ = np.diag(sample_covariance_)
    sample_stddev_ = np.sqrt(sample_variance_)

    tf.compat.v1.logging.vlog(2, "true_mean:\n{}  ".format(true_mean))
    tf.compat.v1.logging.vlog(2, "sample_mean:\n{}".format(sample_mean_))
    tf.compat.v1.logging.vlog(2,
                              "analytical_mean:\n{}".format(analytical_mean_))

    tf.compat.v1.logging.vlog(2, "true_covariance:\n{}".format(true_covariance))
    tf.compat.v1.logging.vlog(
        2, "sample_covariance:\n{}".format(sample_covariance_))
    tf.compat.v1.logging.vlog(
        2, "analytical_covariance:\n{}".format(analytical_covariance_))

    tf.compat.v1.logging.vlog(2, "true_variance:\n{}".format(true_variance))
    tf.compat.v1.logging.vlog(2,
                              "sample_variance:\n{}".format(sample_variance_))
    tf.compat.v1.logging.vlog(
        2, "analytical_variance:\n{}".format(analytical_variance_))

    tf.compat.v1.logging.vlog(2, "true_stddev:\n{}".format(true_stddev))
    tf.compat.v1.logging.vlog(2, "sample_stddev:\n{}".format(sample_stddev_))
    tf.compat.v1.logging.vlog(
        2, "analytical_stddev:\n{}".format(analytical_stddev_))

    tf.compat.v1.logging.vlog(2, "true_scale:\n{}".format(true_scale))
    tf.compat.v1.logging.vlog(2, "scale:\n{}".format(scale_))

    tf.compat.v1.logging.vlog(
        2, "kl_identity:  analytical:{}  sample:{}".format(
            analytical_kl_identity_, sample_kl_identity_))

    tf.compat.v1.logging.vlog(
        2, "kl_scaled:    analytical:{}  sample:{}".format(
            analytical_kl_scaled_, sample_kl_scaled_))

    tf.compat.v1.logging.vlog(
        2, "kl_diag:      analytical:{}  sample:{}".format(
            analytical_kl_diag_, sample_kl_diag_))

    tf.compat.v1.logging.vlog(
        2, "kl_chol:      analytical:{}  sample:{}".format(
            analytical_kl_chol_, sample_kl_chol_))

    tf.compat.v1.logging.vlog(
        2, "kl_identity_diag_baseline:  analytical:{}  sample:{}".format(
            analytical_kl_identity_diag_baseline_,
            sample_kl_identity_diag_baseline_))

    tf.compat.v1.logging.vlog(
        2, "kl_scaled_diag_baseline:  analytical:{}  sample:{}".format(
            analytical_kl_scaled_diag_baseline_,
            sample_kl_scaled_diag_baseline_))

    tf.compat.v1.logging.vlog(
        2, "kl_diag_diag_baseline:  analytical:{}  sample:{}".format(
            analytical_kl_diag_diag_baseline_, sample_kl_diag_diag_baseline_))

    tf.compat.v1.logging.vlog(
        2, "kl_chol_diag_baseline:  analytical:{}  sample:{}".format(
            analytical_kl_chol_diag_baseline_, sample_kl_chol_diag_baseline_))

    self.assertAllClose(true_mean, sample_mean_, atol=0., rtol=0.02)
    self.assertAllClose(true_mean, analytical_mean_, atol=0., rtol=1e-6)

    self.assertAllClose(true_covariance, sample_covariance_, atol=0., rtol=0.02)
    self.assertAllClose(
        true_covariance, analytical_covariance_, atol=0., rtol=1e-6)

    self.assertAllClose(true_variance, sample_variance_, atol=0., rtol=0.02)
    self.assertAllClose(true_variance, analytical_variance_, atol=0., rtol=1e-6)

    self.assertAllClose(true_stddev, sample_stddev_, atol=0., rtol=0.02)
    self.assertAllClose(true_stddev, analytical_stddev_, atol=0., rtol=1e-6)

    self.assertAllClose(true_scale, scale_, atol=0., rtol=1e-6)

    self.assertAllClose(
        sample_kl_identity_, analytical_kl_identity_, atol=0., rtol=0.02)
    self.assertAllClose(
        sample_kl_scaled_, analytical_kl_scaled_, atol=0., rtol=0.02)
    self.assertAllClose(
        sample_kl_diag_, analytical_kl_diag_, atol=0., rtol=0.02)
    self.assertAllClose(
        sample_kl_chol_, analytical_kl_chol_, atol=0., rtol=0.02)

    self.assertAllClose(
        sample_kl_identity_diag_baseline_,
        analytical_kl_identity_diag_baseline_,
        atol=0.,
        rtol=0.02)
    self.assertAllClose(
        sample_kl_scaled_diag_baseline_,
        analytical_kl_scaled_diag_baseline_,
        atol=0.,
        rtol=0.02)
    self.assertAllClose(
        sample_kl_diag_diag_baseline_,
        analytical_kl_diag_diag_baseline_,
        atol=0.,
        rtol=0.04)
    self.assertAllClose(
        sample_kl_chol_diag_baseline_,
        analytical_kl_chol_diag_baseline_,
        atol=0.,
        rtol=0.02)

  def testImplicitLargeDiag(self):
    mu = np.array([[1., 2, 3],
                   [11, 22, 33]])      # shape: [b, k] = [2, 3]
    u = np.array([[[1., 2],
                   [3, 4],
                   [5, 6]],
                  [[0.5, 0.75],
                   [1, 0.25],
                   [1.5, 1.25]]])      # shape: [b, k, r] = [2, 3, 2]
    m = np.array([[0.1, 0.2],
                  [0.4, 0.5]])         # shape: [b, r] = [2, 2]
    scale = np.stack([
        np.eye(3) + np.matmul(np.matmul(u[0], np.diag(m[0])),
                              np.transpose(u[0])),
        np.eye(3) + np.matmul(np.matmul(u[1], np.diag(m[1])),
                              np.transpose(u[1])),
    ])
    cov = np.stack([np.matmul(scale[0], scale[0].T),
                    np.matmul(scale[1], scale[1].T)])
    tf.compat.v1.logging.vlog(2, "expected_cov:\n{}".format(cov))
    mvn = tfd.MultivariateNormalDiagPlusLowRank(
        loc=mu, scale_perturb_factor=u, scale_perturb_diag=m)
    self.assertAllClose(
        cov, self.evaluate(mvn.covariance()), atol=0., rtol=1e-6)


if __name__ == "__main__":
  tf.test.main()
