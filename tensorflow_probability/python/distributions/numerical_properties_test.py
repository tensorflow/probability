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
"""Property-based testing for TFP distributions.

These are distribution properties pertaining to numerics; tested by
default on TF2 graph and eager mode only.

Tests of compatibility with TF platform features are in
platform_compatibility_test.py.

Tests of compatibility with JAX transformations are in
jax_transformation_test.py.
"""

import collections

from absl.testing import parameterized
import hypothesis as hp
from hypothesis import strategies as hps
import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.distributions import hypothesis_testlib as dhps
from tensorflow_probability.python.internal import hypothesis_testlib as tfp_hps
from tensorflow_probability.python.internal import numerics_testing as nt
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.math.psd_kernels import hypothesis_testlib as kernel_hps


WORKING_PRECISION_TEST_BLOCK_LIST = (
    'Masked',  # b/182313283
    # The difficulty concerns Mixtures of component distributions whose samples
    # have different dtypes.
    'Mixture',
    # TODO(b/197680518): ForwardAccumulator does not work under tf1.Graph.
    'PowerSpherical',
    # TODO(b/197680518): ForwardAccumulator does not work under tf1.Graph.
    'SphericalUniform',
    # TODO(b/197680518): ForwardAccumulator does not work under tf1.Graph.
    'VonMisesFisher',
    'ZeroInflatedNegativeBinomial')


NO_NANS_TEST_BLOCK_LIST = (
    'BetaQuotient',  # b/178925774
    'Dirichlet',  # b/169689852
    'ExpRelaxedOneHotCategorical',  # b/169663302
    'RelaxedOneHotCategorical',  # b/169663302
    # Independent log_prob unavoidably emits `nan` if the underlying
    # distribution yields a +inf on one sample and a -inf on another.
    'Independent',
    'LambertWNormal',
    'LogitNormal',  # TODO(axch): Maybe nan problem hints at accuracy problem
    # Mixtures of component distributions whose samples have different dtypes
    # cannot pass validate_args.
    'Mixture',
    'MixtureSameFamily',  # b/169790025
    'OneHotCategorical',  # b/169680869
    # TODO(axch) Re-enable after CDFs of underlying distributions are tested
    # for NaN production.
    'QuantizedDistribution',
    # Sample log_prob unavoidably emits `nan` if the underlying distribution
    # yields a +inf on one sample and a -inf on another.  This can happen even
    # with iid samples, if one sample is at a pole of the distribution, and the
    # other is far enough into the tail to round to +inf.  Weibull with a low
    # concentration is an example of a distribution that can produce this
    # effect.
    'Sample',
    'SinhArcsinh',  # b/183670203
    'TransformedDistribution',  # Bijectors may introduce nans
    # TODO(axch): Edit C++ sampler to reject numerically out-of-bounds samples
    'TruncatedNormal',
)

NANS_EVEN_IN_SAMPLE_LIST = (
    'Mixture',  # b/169847344.  Not a nan, but can't always sample from Mixture
    'SinhArcsinh',  # b/183670203
    'TransformedDistribution',  # Bijectors may introduce nans
)


LOG_PROB_ACCURACY_BLOCK_LIST = (
    # TODO(axch): Understand why each of these has 20+ bad bits; fix
    # or file good bugs.
    'Beta',  # Hypothesis filters too much in 4/100 runs; this is odd.
    'BetaBinomial',
    'BetaQuotient',  # Filters too much in 46/100 runs (nan samples too easy?)
    'Binomial',
    'Categorical',
    'ContinuousBernoulli',
    'Dirichlet',  # Filters too much in 1/100 runs
    'ExponentiallyModifiedGaussian',
    'FiniteDiscrete',
    'GeneralizedExtremeValue',
    'JohnsonSU',
    'Kumaraswamy',
    'LambertWNormal',
    'LogitNormal',  # Filters too much in 6/100 runs (nan samples too easy?)
    'MultivariateNormalDiag',
    'MultivariateNormalTriL',
    'NegativeBinomial',
    'OneHotCategorical',
    'PlackettLuce',
    'SinhArcsinh',  # b/183670203
    'Skellam',
    'StoppingRatioLogistic',  # Filters too much in 61/100 runs; this is odd.
    # TODO(axch): Fix numerics of _cauchy_cdf(x + delta) - _cauchy_cdf(x)
    'TruncatedCauchy',
    # TODO(axch): Is this the same problem as the NoNansTest exclusion?
    'TruncatedNormal',
    'WishartTriL',
    'ZeroInflatedNegativeBinomial')


SLICING_LOGPROB_ATOL = collections.defaultdict(lambda: 1e-5)
SLICING_LOGPROB_ATOL.update({
    'Weibull': 3e-5,
})

SLICING_LOGPROB_RTOL = collections.defaultdict(lambda: 1e-5)
SLICING_LOGPROB_RTOL.update({
    'Weibull': 3e-5,
})


@test_util.test_all_tf_execution_regimes
class LogProbConsistentPrecisionTest(test_util.TestCase):

  @parameterized.named_parameters(
      {'testcase_name': dname, 'dist_name': dname}
      for dname in sorted(list(dhps.INSTANTIABLE_BASE_DISTS.keys())
                          + list(dhps.INSTANTIABLE_META_DISTS)))
  @hp.given(hps.data())
  @tfp_hps.tfp_hp_settings()
  def testDistribution(self, dist_name, data):
    if dist_name in WORKING_PRECISION_TEST_BLOCK_LIST:
      self.skipTest('{} is blocked'.format(dist_name))
    def eligibility_filter(name):
      return name not in WORKING_PRECISION_TEST_BLOCK_LIST
    dist = data.draw(dhps.distributions(
        dist_name=dist_name, eligibility_filter=eligibility_filter,
        enable_vars=False, validate_args=False))
    hp.note('Trying distribution {}'.format(
        self.evaluate_dict(dist.parameters)))
    seed = test_util.test_seed()
    with tfp_hps.no_tf_rank_errors(), kernel_hps.no_pd_errors():
      samples = dist.sample(5, seed=seed)
      self.assertIn(samples.dtype, [tf.float32, tf.int32])
      self.assertEqual(dist.log_prob(samples).dtype, tf.float32)

    def log_prob_function(dist, x):
      return dist.log_prob(x)

    self.assertIsInstance(dist, tf.__internal__.CompositeTensor)
    dist64 = tf.nest.map_structure(
        nt.floating_tensor_to_f64,
        dist,
        expand_composites=True)
    with tfp_hps.no_tf_rank_errors(), kernel_hps.no_pd_errors():
      result64 = log_prob_function(dist64, nt.floating_tensor_to_f64(samples))
    self.assertEqual(result64.dtype, tf.float64)


@test_util.test_all_tf_execution_regimes
class NoNansTest(test_util.TestCase, dhps.TestCase):

  @parameterized.named_parameters(
      {'testcase_name': dname, 'dist_name': dname}
      for dname in sorted(list(dhps.INSTANTIABLE_BASE_DISTS.keys()) +
                          list(dhps.INSTANTIABLE_META_DISTS)))
  @hp.given(hps.data())
  @tfp_hps.tfp_hp_settings()
  def testDistribution(self, dist_name, data):
    if dist_name in NO_NANS_TEST_BLOCK_LIST:
      self.skipTest('{} is blocked'.format(dist_name))
    def eligibility_filter(name):
      return name not in NO_NANS_TEST_BLOCK_LIST
    dist = data.draw(dhps.distributions(dist_name=dist_name, enable_vars=False,
                                        eligibility_filter=eligibility_filter))
    samples = self.check_samples_not_nan(dist)
    self.assume_loc_scale_ok(dist)

    hp.note('Testing on samples {}'.format(samples))
    with tfp_hps.no_tf_rank_errors():
      lp = self.evaluate(dist.log_prob(samples))
      hp.note('Got log_probs {}'.format(lp))
    self.assertAllEqual(np.zeros_like(lp), np.isnan(lp))

  @parameterized.named_parameters(
      {'testcase_name': dname, 'dist_name': dname}
      for dname in sorted(NO_NANS_TEST_BLOCK_LIST)
      if dname not in NANS_EVEN_IN_SAMPLE_LIST)
  @hp.given(hps.data())
  @tfp_hps.tfp_hp_settings()
  def testSampleOnly(self, dist_name, data):
    def eligibility_filter(name):
      # We use this eligibility filter to focus the test's attention
      # on sub-distributions that are not already tested by the sample
      # and log_prob test.  However, we also include some relatively
      # widely used distributions to make sure that at least one legal
      # sub-distribution exists for every meta-distribution we may be
      # testing.
      return ((name in NO_NANS_TEST_BLOCK_LIST
               or name in dhps.QUANTIZED_BASE_DISTS)
              and name not in NANS_EVEN_IN_SAMPLE_LIST)
    dist = data.draw(dhps.distributions(dist_name=dist_name, enable_vars=False,
                                        eligibility_filter=eligibility_filter))
    self.check_samples_not_nan(dist)

  def check_samples_not_nan(self, dist):
    hp.note('Trying distribution {}'.format(
        self.evaluate_dict(dist.parameters)))
    seed = test_util.test_seed(sampler_type='stateless')
    with tfp_hps.no_tf_rank_errors():
      samples = self.evaluate(dist.sample(20, seed=seed))
    self.assertAllEqual(np.zeros_like(samples), np.isnan(samples))
    return samples


# TODO(axch): Testing only in Eager mode because Graph mode barfs on a
# weird tf.constant error inside value_and_gradients.
# @test_util.test_all_tf_execution_regimes
class DistributionAccuracyTest(test_util.TestCase):

  def skip_if_tf1(self):
    # This is a hack to check whether we're running under TF1, which
    # seems to have a less-accurate implementation of some special
    # function.
    # pylint: disable=g-import-not-at-top
    import tensorflow
    if hasattr(tensorflow, 'Session'):
      self.skipTest('TODO(axch): TF1 seems to have an inaccurate base function')

  @parameterized.named_parameters(
      {'testcase_name': dname, 'dist_name': dname}
      for dname in sorted(list(set(dhps.INSTANTIABLE_BASE_DISTS.keys()) -
                               set(LOG_PROB_ACCURACY_BLOCK_LIST))))
  @hp.given(hps.data())
  @tfp_hps.tfp_hp_settings()
  def testLogProbAccuracy(self, dist_name, data):
    self.skip_if_tf1()
    dist = data.draw(dhps.distributions(
        dist_name=dist_name,
        # Accuracy tools can't handle batches (yet?)
        batch_shape=(),
        # Variables presumably do not affect the numerics
        enable_vars=False,
        # Checking that samples pass validations (including in 64-bit
        # arithmetic) is left for another test
        validate_args=False))
    seed = test_util.test_seed(sampler_type='stateless')
    with tfp_hps.no_tf_rank_errors():
      sample = dist.sample(seed=seed)
    if sample.dtype.is_floating:
      hp.assume(self.evaluate(tf.reduce_all(~tf.math.is_nan(sample))))
    hp.note('Testing on sample {}'.format(sample))

    self.assertIsInstance(dist, tf.__internal__.CompositeTensor)
    as_tensors = tf.nest.flatten(dist, expand_composites=True)
    def log_prob_function(tensors, x):
      dist_ = tf.nest.pack_sequence_as(dist, tensors, expand_composites=True)
      return dist_.log_prob(x)

    with tfp_hps.finite_ground_truth_only():
      badness = nt.excess_wrong_bits(log_prob_function, as_tensors, sample)
    # TODO(axch): Lower the acceptable badness to 4, which corresponds
    # to slightly better accuracy than 1e-6 relative error for
    # well-conditioned functions.
    self.assertAllLess(badness, 20)


if __name__ == '__main__':
  # Hypothesis often finds numerical near misses.  Debugging them is much aided
  # by seeing all the digits of every floating point number, instead of the
  # usual default of truncating the printed representation to 8 digits.
  np.set_printoptions(floatmode='unique', precision=None)
  test_util.main()
