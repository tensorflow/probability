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
"""Tests for MCMC diagnostic utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import numpy as np
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.mcmc.diagnostic import _reduce_variance


@test_util.test_all_tf_execution_regimes
class _EffectiveSampleSizeTest(object):

  @property
  def use_static_shape(self):
    raise NotImplementedError(
        "Subclass failed to implement `use_static_shape`.")

  def _check_versus_expected_effective_sample_size(
      self,
      x_,
      expected_ess,
      atol=1e-2,
      rtol=1e-2,
      filter_threshold=None,
      filter_beyond_lag=None,
      filter_beyond_positive_pairs=False):
    x = tf1.placeholder_with_default(
        x_, shape=x_.shape if self.use_static_shape else None)
    ess = tfp.mcmc.effective_sample_size(
        x,
        filter_threshold=filter_threshold,
        filter_beyond_lag=filter_beyond_lag,
        filter_beyond_positive_pairs=filter_beyond_positive_pairs)
    if self.use_static_shape:
      self.assertAllEqual(x.shape[1:], ess.shape)

    ess_ = self.evaluate(ess)
    self.assertAllClose(
        np.ones_like(ess_) * expected_ess, ess_, atol=atol, rtol=rtol)

  def testIidRank1NormalHasFullEssMaxLags10(self):
    # With a length 5000 iid normal sequence, and filter_beyond_lag = 10, we
    # should have a good estimate of ESS, and it should be close to the full
    # sequence length of 5000.
    # The choice of filter_beyond_lag = 10 is a short cutoff, reasonable only
    # since we know the correlation length should be zero right away.
    self._check_versus_expected_effective_sample_size(
        x_=np.random.randn(5000).astype(np.float32),
        expected_ess=5000,
        filter_beyond_lag=10,
        filter_threshold=None,
        rtol=0.3)

  def testIidRank2NormalHasFullEssMaxLags10(self):
    # See similar test for Rank1Normal for reasoning.
    self._check_versus_expected_effective_sample_size(
        x_=np.random.randn(5000, 2).astype(np.float32),
        expected_ess=5000,
        filter_beyond_lag=10,
        filter_threshold=None,
        rtol=0.3)

  def testIidRank1NormalHasFullEssMaxLagThresholdZero(self):
    # With a length 5000 iid normal sequence, and filter_threshold = 0,
    # we should have a super-duper estimate of ESS, and it should be very close
    # to the full sequence length of 5000.
    # The choice of filter_beyond_lag = 0 means we cutoff as soon as the
    # auto-corr is below zero.  This should happen very quickly, due to the fact
    # that the theoretical auto-corr is [1, 0, 0,...]
    self._check_versus_expected_effective_sample_size(
        x_=np.random.randn(5000).astype(np.float32),
        expected_ess=5000,
        filter_beyond_lag=None,
        filter_threshold=0.,
        rtol=0.1)

  def testIidRank2NormalHasFullEssMaxLagThresholdZero(self):
    # See similar test for Rank1Normal for reasoning.
    self._check_versus_expected_effective_sample_size(
        x_=np.random.randn(5000, 2).astype(np.float32),
        expected_ess=5000,
        filter_beyond_lag=None,
        filter_threshold=0.,
        rtol=0.1)

  def testIidRank1NormalHasFullEssMaxLagInitialPositive(self):
    # See similar test for ThresholdZero for background. This time this uses the
    # initial_positive sequence criterion. In this case, initial_positive
    # sequence might be a little more noisy than the threshold case because it
    # will typically not drop the lag-1 auto-correlation.
    self._check_versus_expected_effective_sample_size(
        x_=np.random.randn(5000).astype(np.float32),
        expected_ess=5000,
        filter_beyond_lag=None,
        filter_threshold=None,
        filter_beyond_positive_pairs=True,
        rtol=0.25)

  def testIidRank2NormalHasFullEssMaxLagInitialPositive(self):
    # See similar test for Rank1Normal for reasoning.
    self._check_versus_expected_effective_sample_size(
        x_=np.random.randn(5000, 2).astype(np.float32),
        expected_ess=5000,
        filter_beyond_lag=None,
        filter_threshold=None,
        filter_beyond_positive_pairs=True,
        rtol=0.25)

  def testIidRank1NormalHasFullEssMaxLagInitialPositiveOddLength(self):
    # See similar test for Rank1Normal for reasoning.
    self._check_versus_expected_effective_sample_size(
        x_=np.random.randn(4999).astype(np.float32),
        expected_ess=4999,
        filter_beyond_lag=None,
        filter_threshold=None,
        filter_beyond_positive_pairs=True,
        rtol=0.2)

  def testLength10CorrelationHasEssOneTenthTotalLengthUsingMaxLags50(self):
    # Create x_, such that
    #   x_[i] = iid_x_[0], i = 0,...,9
    #   x_[i] = iid_x_[1], i = 10,..., 19,
    #   and so on.
    iid_x_ = np.random.randn(5000, 1).astype(np.float32)
    x_ = (iid_x_ * np.ones((5000, 10)).astype(np.float32)).reshape((50000,))
    self._check_versus_expected_effective_sample_size(
        x_=x_,
        expected_ess=50000 // 10,
        filter_beyond_lag=50,
        filter_threshold=None,
        rtol=0.2)

  def testLength10CorrelationHasEssOneTenthTotalLengthUsingMaxLagsThresholdZero(
      self):
    # Create x_, such that
    #   x_[i] = iid_x_[0], i = 0,...,9
    #   x_[i] = iid_x_[1], i = 10,..., 19,
    #   and so on.
    iid_x_ = np.random.randn(5000, 1).astype(np.float32)
    x_ = (iid_x_ * np.ones((5000, 10)).astype(np.float32)).reshape((50000,))
    self._check_versus_expected_effective_sample_size(
        x_=x_,
        expected_ess=50000 // 10,
        filter_beyond_lag=None,
        filter_threshold=0.,
        rtol=0.1)

  def testLength10CorrelationHasEssOneTenthTotalLengthUsingMaxLagsInitialPos(
      self):
    # Create x_, such that
    #   x_[i] = iid_x_[0], i = 0,...,9
    #   x_[i] = iid_x_[1], i = 10,..., 19,
    #   and so on.
    iid_x_ = np.random.randn(5000, 1).astype(np.float32)
    x_ = (iid_x_ * np.ones((5000, 10)).astype(np.float32)).reshape((50000,))
    self._check_versus_expected_effective_sample_size(
        x_=x_,
        expected_ess=50000 // 10,
        filter_beyond_lag=None,
        filter_threshold=None,
        filter_beyond_positive_pairs=True,
        rtol=0.15)

  def testListArgs(self):
    # x_ has correlation length 10 ==> ESS = N / 10
    # y_ has correlation length 1  ==> ESS = N
    iid_x_ = np.random.randn(5000, 1).astype(np.float32)
    x_ = (iid_x_ * np.ones((5000, 10)).astype(np.float32)).reshape((50000,))
    y_ = np.random.randn(50000).astype(np.float32)
    states = [x_, x_, y_, y_]
    filter_threshold = [0., None, 0., None]
    filter_beyond_lag = [None, 5, None, 5]

    # See other tests for reasoning on tolerance.
    ess = tfp.mcmc.effective_sample_size(
        states,
        filter_threshold=filter_threshold,
        filter_beyond_lag=filter_beyond_lag)
    ess_ = self.evaluate(ess)
    self.assertAllEqual(4, len(ess_))

    self.assertAllClose(50000 // 10, ess_[0], rtol=0.3)
    self.assertAllClose(50000 // 10, ess_[1], rtol=0.3)
    self.assertAllClose(50000, ess_[2], rtol=0.1)
    self.assertAllClose(50000, ess_[3], rtol=0.1)

  def testMaxLagsThresholdLessThanNeg1SameAsNone(self):
    # Setting both means we filter out items R_k from the auto-correlation
    # sequence if k > filter_beyond_lag OR k >= j where R_j < filter_threshold.

    # x_ has correlation length 10.
    iid_x_ = np.random.randn(500, 1).astype(np.float32)
    x_ = (iid_x_ * np.ones((500, 10)).astype(np.float32)).reshape((5000,))
    x = tf1.placeholder_with_default(
        x_, shape=x_.shape if self.use_static_shape else None)

    ess_none_none = tfp.mcmc.effective_sample_size(
        x, filter_threshold=None, filter_beyond_lag=None)
    ess_none_200 = tfp.mcmc.effective_sample_size(
        x, filter_threshold=None, filter_beyond_lag=200)
    ess_neg2_200 = tfp.mcmc.effective_sample_size(
        x, filter_threshold=-2., filter_beyond_lag=200)
    ess_neg2_none = tfp.mcmc.effective_sample_size(
        x, filter_threshold=-2., filter_beyond_lag=None)
    [ess_none_none_, ess_none_200_, ess_neg2_200_,
     ess_neg2_none_] = self.evaluate(
         [ess_none_none, ess_none_200, ess_neg2_200, ess_neg2_none])

    # filter_threshold=-2 <==> filter_threshold=None.
    self.assertAllClose(ess_none_none_, ess_neg2_none_)
    self.assertAllClose(ess_none_200_, ess_neg2_200_)

  def testMaxLagsArgsAddInAnOrManner(self):
    # Setting both means we filter out items R_k from the auto-correlation
    # sequence if k > filter_beyond_lag OR k >= j where R_j < filter_threshold.

    # x_ has correlation length 10.
    iid_x_ = np.random.randn(500, 1).astype(np.float32)
    x_ = (iid_x_ * np.ones((500, 10)).astype(np.float32)).reshape((5000,))
    x = tf1.placeholder_with_default(
        x_, shape=x_.shape if self.use_static_shape else None)

    ess_1_9 = tfp.mcmc.effective_sample_size(
        x, filter_threshold=1., filter_beyond_lag=9)
    ess_1_none = tfp.mcmc.effective_sample_size(
        x, filter_threshold=1., filter_beyond_lag=None)
    ess_none_9 = tfp.mcmc.effective_sample_size(
        x, filter_threshold=1., filter_beyond_lag=9)
    ess_1_9_, ess_1_none_, ess_none_9_ = self.evaluate(
        [ess_1_9, ess_1_none, ess_none_9])

    # Since R_k = 1 for k < 10, and R_k < 1 for k >= 10,
    # filter_threshold = 1 <==> filter_beyond_lag = 9.
    self.assertAllClose(ess_1_9_, ess_1_none_)
    self.assertAllClose(ess_1_9_, ess_none_9_)

  def testInitialPositiveAndLag(self):
    # We will use the max_lags argument to verify that initial_positive sequence
    # argument does what it should.

    # This sequence begins to have non-positive pairwise sums at lag 38
    x_ = np.linspace(-1., 1., 100).astype(np.float32)
    x = tf1.placeholder_with_default(
        x_, shape=x_.shape if self.use_static_shape else None)

    ess_true_37 = tfp.mcmc.effective_sample_size(
        x,
        filter_beyond_positive_pairs=True,
        filter_threshold=None,
        filter_beyond_lag=37)
    ess_true_none = tfp.mcmc.effective_sample_size(
        x,
        filter_beyond_positive_pairs=True,
        filter_threshold=None,
        filter_beyond_lag=None)
    ess_false_37 = tfp.mcmc.effective_sample_size(
        x,
        filter_beyond_positive_pairs=False,
        filter_threshold=None,
        filter_beyond_lag=37)
    ess_true_37_, ess_true_none_, ess_false_37_ = self.evaluate(
        [ess_true_37, ess_true_none, ess_false_37])

    self.assertAllClose(ess_true_37_, ess_true_none_)
    self.assertAllClose(ess_true_37_, ess_false_37_)

  def testInitialPositiveSuperEfficient(self):
    # Initial positive sequence will correctly estimate the ESS of
    # super-efficient MCMC chains.

    # This sequence has strong anti-autocorrelation, so will get ESS larger than
    # its length.
    x_ = ((np.arange(0, 100) % 2).astype(np.float32) -
          0.5) * np.exp(-np.linspace(0., 10., 100))
    x = tf1.placeholder_with_default(
        x_, shape=x_.shape if self.use_static_shape else None)

    ess = tfp.mcmc.effective_sample_size(
        x, filter_beyond_positive_pairs=True)
    ess_ = self.evaluate(ess)

    self.assertGreater(ess_, 100.)

  def testCrossChainESSWellMixing(self):
    # For multiple well-mixing chains, summing the ESS computed over individual
    # chains will be roughly the same as doing the cross-chain ESS.
    x_ = np.random.randn(500, 4).astype(np.float32)
    x = tf1.placeholder_with_default(
        x_, shape=x_.shape if self.use_static_shape else None)
    ess_per_chain = tfp.mcmc.effective_sample_size(x)
    cross_chain_dims = 1
    if not self.use_static_shape:
      cross_chain_dims = tf1.placeholder_with_default(
          cross_chain_dims, shape=[])
    ess_cross_chain = tfp.mcmc.effective_sample_size(
        x, cross_chain_dims=cross_chain_dims)
    ess_per_chain_, ess_cross_chain_ = self.evaluate(
        [ess_per_chain, ess_cross_chain])

    self.assertAllClose(ess_per_chain_.sum(), ess_cross_chain_, rtol=0.05)

  def testCrossChainESSPoorlyMixing(self):
    # For multiple non-mixing chains, cross-chain ESS will report the number of
    # modes.
    x_ = np.random.randn(500, 4).astype(np.float32) + np.array(
        [-10., -5., 5., 10.])
    x = tf1.placeholder_with_default(
        x_, shape=x_.shape if self.use_static_shape else None)
    cross_chain_dims = 1
    if not self.use_static_shape:
      cross_chain_dims = tf1.placeholder_with_default(
          cross_chain_dims, shape=[])
    ess_cross_chain = tfp.mcmc.effective_sample_size(
        x, cross_chain_dims=cross_chain_dims)
    ess_cross_chain_ = self.evaluate(ess_cross_chain)

    self.assertAllClose(4., ess_cross_chain_, rtol=0.05)

  def testCrossChainEssMultipleDims(self):
    x_ = np.random.randn(500, 2, 2).astype(np.float32)
    x = tf1.placeholder_with_default(
        x_, shape=x_.shape if self.use_static_shape else None)
    ess_per_chain = tfp.mcmc.effective_sample_size(x)
    ess_cross_chain = tfp.mcmc.effective_sample_size(x, cross_chain_dims=[1, 2])
    ess_per_chain_, ess_cross_chain_ = self.evaluate(
        [ess_per_chain, ess_cross_chain])

    self.assertAllClose(ess_per_chain_.sum(), ess_cross_chain_, rtol=0.05)

  def testCrossChainEssListArgs(self):
    x_ = np.random.randn(500, 2, 2).astype(np.float32)
    x = tf1.placeholder_with_default(
        x_, shape=x_.shape if self.use_static_shape else None)
    y_ = np.random.randn(500, 4).astype(np.float32)
    y = tf1.placeholder_with_default(
        y_, shape=y_.shape if self.use_static_shape else None)
    ess_per_chain = tfp.mcmc.effective_sample_size([x, y])
    ess_cross_chain = tfp.mcmc.effective_sample_size([x, y],
                                                     cross_chain_dims=[
                                                         [1, 2],
                                                         1,
                                                     ])
    ess_per_chain_, ess_cross_chain_ = self.evaluate(
        [ess_per_chain, ess_cross_chain])

    self.assertAllClose([e.sum() for e in ess_per_chain_],
                        ess_cross_chain_,
                        rtol=0.05)

  def testCrossChainEssNotEnoughChains(self):
    x_ = np.random.randn(500, 1).astype(np.float32)
    x = tf1.placeholder_with_default(
        x_, shape=x_.shape if self.use_static_shape else None)
    cross_chain_dims = 1
    if not self.use_static_shape:
      cross_chain_dims = tf1.placeholder_with_default(
          cross_chain_dims, shape=[])
    with self.assertRaisesRegexp(Exception, "there must be > 1 chain"):
      self.evaluate(
          tfp.mcmc.effective_sample_size(
              x,
              cross_chain_dims=cross_chain_dims,
              validate_args=True))


@test_util.test_all_tf_execution_regimes
class EffectiveSampleSizeStaticTest(test_util.TestCase,
                                    _EffectiveSampleSizeTest):

  @property
  def use_static_shape(self):
    return True


@test_util.test_all_tf_execution_regimes
class EffectiveSampleSizeDynamicTest(test_util.TestCase,
                                     _EffectiveSampleSizeTest):

  @property
  def use_static_shape(self):
    return False


@test_util.test_all_tf_execution_regimes
class _PotentialScaleReductionTest(object):

  @property
  def use_static_shape(self):
    raise NotImplementedError(
        "Subclass failed to implement `use_static_shape`.")

  def testListOfStatesWhereFirstPassesSecondFails(self):
    """Simple test showing API with two states.  Read first!."""
    n_samples = 1000

    # state_0 is two scalar chains taken from iid Normal(0, 1).  Will pass.
    state_0 = np.random.randn(n_samples, 2)

    # state_1 is three 4-variate chains taken from Normal(0, 1) that have been
    # shifted.  Since every chain is shifted, they are not the same, and the
    # test should fail.
    offset = np.array([1., -1., 2.]).reshape(3, 1)
    state_1 = np.random.randn(n_samples, 3, 4) + offset

    rhat = tfp.mcmc.potential_scale_reduction(
        chains_states=[state_0, state_1], independent_chain_ndims=1)

    self.assertIsInstance(rhat, list)
    rhat_0_, rhat_1_ = self.evaluate(rhat)

    # r_hat_0 should be close to 1, meaning test is passed.
    self.assertAllEqual((), rhat_0_.shape)
    self.assertAllClose(1., rhat_0_, rtol=0.02)

    # r_hat_1 should be greater than 1.2, meaning test has failed.
    self.assertAllEqual((4,), rhat_1_.shape)
    self.assertAllEqual(np.ones_like(rhat_1_).astype(bool), rhat_1_ > 1.2)

  def check_results(self,
                    state_,
                    independent_chain_shape,
                    should_pass,
                    split_chains=False):
    sample_ndims = 1
    independent_chain_ndims = len(independent_chain_shape)
    state = tf1.placeholder_with_default(
        state_, shape=state_.shape if self.use_static_shape else None)

    rhat = tfp.mcmc.potential_scale_reduction(
        state,
        independent_chain_ndims=independent_chain_ndims,
        split_chains=split_chains)

    if self.use_static_shape:
      self.assertAllEqual(
          state_.shape[sample_ndims + independent_chain_ndims:], rhat.shape)

    rhat_ = self.evaluate(rhat)
    if should_pass:
      self.assertAllClose(np.ones_like(rhat_), rhat_, atol=0, rtol=0.02)
    else:
      self.assertAllEqual(np.ones_like(rhat_).astype(bool), rhat_ > 1.2)

  def iid_normal_chains_should_pass_wrapper(self,
                                            sample_shape,
                                            independent_chain_shape,
                                            other_shape,
                                            split_chains=False,
                                            dtype=np.float32):
    """Check results with iid normal chains."""

    state_shape = sample_shape + independent_chain_shape + other_shape
    state_ = np.random.randn(*state_shape).astype(dtype)

    # The "other" dimensions do not have to be identical, just independent, so
    # force them to not be identical.
    if other_shape:
      state_ *= np.random.rand(*other_shape).astype(dtype)

    self.check_results(
        state_,
        independent_chain_shape,
        should_pass=True,
        split_chains=split_chains)

  def testPassingIIDNdimsAreIndependentOneOtherZero(self):
    self.iid_normal_chains_should_pass_wrapper(
        sample_shape=[10000], independent_chain_shape=[4], other_shape=[])

  def testPassingIIDNdimsAreIndependentOneOtherOne(self):
    self.iid_normal_chains_should_pass_wrapper(
        sample_shape=[10000], independent_chain_shape=[3], other_shape=[7])

  def testPassingIIDNdimsAreIndependentOneOtherOneSplitChainsEvenNSamples(self):
    self.iid_normal_chains_should_pass_wrapper(
        sample_shape=[10000],
        independent_chain_shape=[3],
        other_shape=[7],
        split_chains=True)

  def testPassingIIDNdimsAreIndependentOneOtherOneSplitChainsOddNSamples(self):
    # For odd number of samples we must remove last sample.
    self.iid_normal_chains_should_pass_wrapper(
        sample_shape=[10001],
        independent_chain_shape=[3],
        other_shape=[7],
        split_chains=True)

  def testPassingIIDNdimsAreIndependentOneOtherTwo(self):
    self.iid_normal_chains_should_pass_wrapper(
        sample_shape=[10000], independent_chain_shape=[2], other_shape=[5, 7])

  def testPassingIIDNdimsAreIndependentTwoOtherTwo64Bit(self):
    self.iid_normal_chains_should_pass_wrapper(
        sample_shape=[10000],
        independent_chain_shape=[2, 3],
        other_shape=[5, 7],
        dtype=np.float64)

  def offset_normal_chains_should_fail_wrapper(
      self, sample_shape, independent_chain_shape, other_shape):
    """Check results with normal chains that are offset from each other."""

    state_shape = sample_shape + independent_chain_shape + other_shape
    state_ = np.random.randn(*state_shape)

    # Add a significant offset to the different (formerly iid) chains.
    offset = np.linspace(
        0, 2, num=np.prod(independent_chain_shape)).reshape([1] * len(
            sample_shape) + independent_chain_shape + [1] * len(other_shape))
    state_ += offset

    self.check_results(state_, independent_chain_shape, should_pass=False)

  def testFailingOffsetNdimsAreSampleOneIndependentOneOtherOne(self):
    self.offset_normal_chains_should_fail_wrapper(
        sample_shape=[10000], independent_chain_shape=[2], other_shape=[5])

  def testLinearTrendPassesIfNoSplitChains(self):
    # A problem with non-split Rhat is that it does not catch linear trends.
    n_samples = 1000
    n_chains = 10
    state_ = (
        np.random.randn(n_samples, n_chains) +
        np.linspace(0, 1, n_samples).reshape(n_samples, 1))
    self.check_results(
        state_,
        independent_chain_shape=[n_chains],
        should_pass=True,
        split_chains=False)

  def testLinearTrendFailsIfSplitChains(self):
    n_samples = 10000
    n_chains = 10
    state_ = (
        np.random.randn(n_samples, n_chains) +
        np.linspace(0, 10, n_samples).reshape(n_samples, 1))
    self.check_results(
        state_,
        independent_chain_shape=[n_chains],
        should_pass=False,
        split_chains=True)

  def testNotEnoughSamplesNoSplitChainsFailsIfValidateArgs(self):
    input_ = np.random.rand(1, 10)
    x = tf1.placeholder_with_default(
        input_, shape=input_.shape if self.use_static_shape else None)
    with self.assertRaisesError("Must provide at least 2 samples"):
      self.evaluate(
          tfp.mcmc.potential_scale_reduction(
              # Require at least 2 samples...have only 1
              x,
              independent_chain_ndims=1,
              validate_args=True))

  def testNotEnoughSamplesWithSplitChainsFailsIfValidateArgs(self):
    input_ = np.random.rand(3, 10)
    x = tf1.placeholder_with_default(
        input_, shape=input_.shape if self.use_static_shape else None)
    with self.assertRaisesError("Must provide at least 4 samples"):
      self.evaluate(
          tfp.mcmc.potential_scale_reduction(
              # Require at least 4 samples...have only 3
              x,
              independent_chain_ndims=1,
              split_chains=True,
              validate_args=True))


@test_util.test_all_tf_execution_regimes
class PotentialScaleReductionStaticTest(test_util.TestCase,
                                        _PotentialScaleReductionTest):

  @property
  def use_static_shape(self):
    return True

  def assertRaisesError(self, msg):
    return self.assertRaisesRegexp(Exception, msg)

  def testIndependentNdimsLessThanOneRaises(self):
    with self.assertRaisesRegexp(ValueError, "independent_chain_ndims"):
      tfp.mcmc.potential_scale_reduction(
          np.random.rand(2, 3, 4), independent_chain_ndims=0)


@test_util.test_all_tf_execution_regimes
class PotentialScaleReductionDynamicTest(test_util.TestCase,
                                         _PotentialScaleReductionTest):

  @property
  def use_static_shape(self):
    return False

  def assertRaisesError(self, msg):
    if tf.executing_eagerly():
      return self.assertRaisesRegexp(Exception, msg)
    return self.assertRaisesOpError(msg)


@test_util.test_all_tf_execution_regimes
class _ReduceVarianceTest(object):

  @property
  def use_static_shape(self):
    raise NotImplementedError(
        "Subclass failed to implement `use_static_shape`.")

  def check_versus_numpy(self, x_, axis, biased, keepdims):
    x_ = np.asarray(x_)
    x = tf1.placeholder_with_default(
        x_, shape=x_.shape if self.use_static_shape else None)
    var = _reduce_variance(
        x, axis=axis, biased=biased, keepdims=keepdims)
    np_var = np.var(x_, axis=axis, ddof=0 if biased else 1, keepdims=keepdims)

    if self.use_static_shape:
      self.assertAllEqual(np_var.shape, var.shape)

    var_ = self.evaluate(var)
    # We will mask below, which changes shape, so check shape explicitly here.
    self.assertAllEqual(np_var.shape, var_.shape)

    # We get NaN when we divide by zero due to the size being the same as ddof
    nan_mask = np.isnan(np_var)
    if nan_mask.any():
      self.assertTrue(np.isnan(var_[nan_mask]).all())
    self.assertAllClose(np_var[~nan_mask], var_[~nan_mask], atol=0, rtol=0.02)

  def testScalarBiasedTrue(self):
    self.check_versus_numpy(x_=-1.234, axis=None, biased=True, keepdims=False)

  def testScalarBiasedFalse(self):
    # This should result in NaN.
    self.check_versus_numpy(x_=-1.234, axis=None, biased=False, keepdims=False)

  def testShape2x3x4AxisNoneBiasedFalseKeepdimsFalse(self):
    self.check_versus_numpy(
        x_=np.random.randn(2, 3, 4), axis=None, biased=True, keepdims=False)

  def testShape2x3x4Axis1BiasedFalseKeepdimsTrue(self):
    self.check_versus_numpy(
        x_=np.random.randn(2, 3, 4), axis=1, biased=True, keepdims=True)

  def testShape2x3x4x5Axis13BiasedFalseKeepdimsTrue(self):
    self.check_versus_numpy(
        x_=np.random.randn(2, 3, 4, 5), axis=1, biased=True, keepdims=True)

  def testShape2x3x4x5Axis13BiasedFalseKeepdimsFalse(self):
    self.check_versus_numpy(
        x_=np.random.randn(2, 3, 4, 5), axis=1, biased=False, keepdims=False)


@test_util.test_all_tf_execution_regimes
class ReduceVarianceTestStaticShape(test_util.TestCase, _ReduceVarianceTest):

  @property
  def use_static_shape(self):
    return True


@test_util.test_all_tf_execution_regimes
class ReduceVarianceTestDynamicShape(test_util.TestCase, _ReduceVarianceTest):

  @property
  def use_static_shape(self):
    return False


if __name__ == "__main__":
  tf.test.main()
