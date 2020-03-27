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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
from scipy import stats

import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import test_util

tfd = tfp.distributions
rng = np.random.RandomState(123)


@test_util.test_all_tf_execution_regimes
class QuantizedDistributionTest(test_util.TestCase):

  def testQuantizationOfUniformWithCutoffsHavingNoEffect(self):
    # The Quantized uniform with cutoffs == None divides the real line into:
    # R = ...(-1, 0](0, 1](1, 2](2, 3](3, 4]...
    # j = ...     0     1     2     3     4 ...
    # Since this uniform (below) is supported on [0, 3],
    # it places 1/3 of its mass in the intervals j = 1, 2, 3.
    # Adding a cutoff at y = 0 changes the picture to
    # R = ...(-inf, 0](0, 1](1, 2](2, 3](3, 4]...
    # j = ...     0     1     2     3     4 ...
    # So the QUniform still places 1/3 of its mass in the intervals
    # j = 1, 2, 3.
    # Adding a cutoff at y = 3 changes the picture to
    # R = ...(-1, 0](0, 1](1, 2](2, inf)
    # j = ...     0     1     2     3
    # and the QUniform still places 1/3 of its mass in the intervals
    # j = 1, 2, 3.
    for lcut, ucut in [(None, None), (0.0, None), (None, 3.0), (0.0, 3.0),
                       (-10., 10.)]:
      qdist = tfd.QuantizedDistribution(
          distribution=tfd.Uniform(low=0.0, high=3.0),
          low=lcut,
          high=ucut,
          validate_args=False)

      # pmf
      pmf_n1, pmf_0, pmf_1, pmf_2, pmf_3, pmf_4, pmf_5 = self.evaluate(
          qdist.prob([-1., 0., 1., 2., 3., 4., 5.]))
      # uniform had no mass below -1.
      self.assertAllClose(0., pmf_n1)
      # uniform had no mass below 0.
      self.assertAllClose(0., pmf_0)
      # uniform put 1/3 of its mass in each of (0, 1], (1, 2], (2, 3],
      # which are the intervals j = 1, 2, 3.
      self.assertAllClose(1 / 3, pmf_1)
      self.assertAllClose(1 / 3, pmf_2)
      self.assertAllClose(1 / 3, pmf_3)
      # uniform had no mass in (3, 4] or (4, 5], which are j = 4, 5.
      self.assertAllClose(0 / 3, pmf_4)
      self.assertAllClose(0 / 3, pmf_5)

      # cdf
      cdf_n1, cdf_0, cdf_1, cdf_2, cdf_2p5, cdf_3, cdf_4, cdf_5 = self.evaluate(
          qdist.cdf([-1., 0., 1., 2., 2.5, 3., 4., 5.]))
      self.assertAllClose(0., cdf_n1)
      self.assertAllClose(0., cdf_0)
      self.assertAllClose(1 / 3, cdf_1)
      self.assertAllClose(2 / 3, cdf_2)
      # Note fractional values allowed for cdfs of discrete distributions.
      # And adding 0.5 makes no difference because the quantized dist has
      # mass only on the integers, never in between.
      self.assertAllClose(2 / 3, cdf_2p5)
      self.assertAllClose(3 / 3, cdf_3)
      self.assertAllClose(3 / 3, cdf_4)
      self.assertAllClose(3 / 3, cdf_5)

  def testQuantizationOfUniformWithCutoffsInTheMiddle(self):
    # The uniform is supported on [-3, 3]
    # Consider partitions the real line in intervals
    # ...(-3, -2](-2, -1](-1, 0](0, 1](1, 2](2, 3] ...
    # Before cutoffs, the uniform puts a mass of 1/6 in each interval written
    # above.  Because of cutoffs, the qdist considers intervals and indices
    # ...(-infty, -1](-1, 0](0, infty) ...
    #             -1      0     1
    qdist = tfd.QuantizedDistribution(
        distribution=tfd.Uniform(low=-3., high=3.),
        low=-1.0,
        high=1.0,
        validate_args=False)

    # pmf
    cdf_n3, cdf_n2, cdf_n1, cdf_0, cdf_0p5, cdf_1, cdf_10 = self.evaluate(
        qdist.cdf([-3., -2., -1., 0., 0.5, 1.0, 10.0]))
    # Uniform had no mass on (-4, -3] or (-3, -2]
    self.assertAllClose(0., cdf_n3)
    self.assertAllClose(0., cdf_n2)
    # Uniform had 1/6 of its mass in each of (-3, -2], and (-2, -1], which
    # were collapsed into (-infty, -1], which is now the '-1' interval.
    self.assertAllClose(1 / 3, cdf_n1)
    # The j=0 interval contained mass from (-3, 0], which is 1/2 of the
    # uniform's mass.
    self.assertAllClose(1 / 2, cdf_0)
    # Adding 0.5 makes no difference because the quantized dist has mass on
    # the integers, not in between them.
    self.assertAllClose(1 / 2, cdf_0p5)
    # After applying the cutoff, all mass was either in the interval
    # (0, infty), or below.  (0, infty) is the interval indexed by j=1,
    # so pmf(1) should equal 1.
    self.assertAllClose(1., cdf_1)
    # Since no mass of qdist is above 1,
    # pmf(10) = P[Y <= 10] = P[Y <= 1] = pmf(1).
    self.assertAllClose(1., cdf_10)

  def testQuantizationOfBatchOfUniforms(self):
    batch_shape = (5, 5)
    # The uniforms are supported on [0, 10].  The qdist considers the
    # intervals
    # ... (0, 1](1, 2]...(9, 10]...
    # with the intervals displayed above each holding 1 / 10 of the mass.
    # The qdist will be defined with no cutoffs,
    uniform = tfd.Uniform(
        low=tf.zeros(batch_shape, dtype=tf.float32),
        high=10 * tf.ones(batch_shape, dtype=tf.float32))
    qdist = tfd.QuantizedDistribution(
        distribution=uniform, low=None, high=None, validate_args=True)

    # x is random integers in {-3,...,12}.
    x = rng.randint(-3, 13, size=batch_shape).astype(np.float32)

    # pmf
    # qdist.prob(j) = 1 / 10 for j in {1,...,10}, and 0 otherwise,
    expected_pmf = (1 / 10) * np.ones(batch_shape)
    expected_pmf[x < 1] = 0.
    expected_pmf[x > 10] = 0.
    self.assertAllClose(expected_pmf, self.evaluate(qdist.prob(x)))

    # cdf
    # qdist.cdf(j)
    #    = 0 for j < 1
    #    = j / 10, for j in {1,...,10},
    #    = 1, for j > 10.
    expected_cdf = x.copy() / 10
    expected_cdf[x < 1] = 0.
    expected_cdf[x > 10] = 1.
    self.assertAllClose(expected_cdf, self.evaluate(qdist.cdf(x)))

  def testSamplingFromBatchOfNormals(self):
    batch_shape = (2,)
    normal = tfd.Normal(
        loc=tf.zeros(batch_shape, dtype=tf.float32),
        scale=tf.ones(batch_shape, dtype=tf.float32))

    qdist = tfd.QuantizedDistribution(
        distribution=normal, low=0., high=None, validate_args=True)

    samps = qdist.sample(5000, seed=test_util.test_seed())
    samps_v = self.evaluate(samps)

    # With low = 0, the interval j=0 is (-infty, 0], which holds 1/2
    # of the mass of the normals.
    #
    # NOTE: rtol chosen to be ~5 standard deviations from the expected value,
    # to give a false positive rate of ~1e-6.
    self.assertAllClose([0.5, 0.5], (samps_v == 0).mean(axis=0), rtol=0.071)

    # The interval j=1 is (0, 1], which is from the mean to one standard
    # deviation out.  This should contain 0.6827 / 2 of the mass.
    #
    # NOTE: rtol chosen to be ~5 standard deviations from the expected value,
    # to give a false positive rate of ~1e-6.
    self.assertAllClose(
        [0.6827 / 2, 0.6827 / 2], (samps_v == 1).mean(axis=0), rtol=0.098)

  def testSamplesAgreeWithCdfForSamplesOverLargeRange(self):
    # Consider the cdf for distribution X, F(x).
    # If U ~ Uniform[0, 1], then Y := F^{-1}(U) is distributed like X since
    # P[Y <= y] = P[F^{-1}(U) <= y] = P[U <= F(y)] = F(y).
    # If F is a bijection, we also have Z = F(X) is Uniform.
    #
    # Make an exponential with large mean (= 100).  This ensures we will get
    # quantized values over a large range.  This large range allows us to
    # pretend that the cdf F is a bijection, and hence F(X) is uniform.
    # Note that F cannot be bijection since it is constant between the
    # integers.  Hence, F(X) (see below) will not be uniform exactly.
    qdist = tfd.QuantizedDistribution(
        distribution=tfd.Exponential(rate=0.01), validate_args=True)
    # X ~ QuantizedExponential
    x = qdist.sample(10000, seed=test_util.test_seed())
    # Z = F(X), should be Uniform.
    z = qdist.cdf(x)
    # Compare the CDF of Z to that of a Uniform.
    # dist = maximum distance between P[Z <= a] and P[U <= a].
    # We ignore pvalue, since of course this distribution is not exactly, and
    # with so many sample points we would get a false fail.
    dist, _ = stats.kstest(self.evaluate(z), 'uniform')

    # Since the distribution take values (approximately) in [0, 100], the
    # cdf should have jumps (approximately) every 1/100 of the way up.
    # Assert that the jumps are not more than 2/100.
    self.assertLess(dist, 0.02)

  def testSamplesAgreeWithPdfForSamplesOverSmallRange(self):
    # Testing that samples and pdf agree for a small range is important because
    # it makes sure the bin edges are consistent.

    # Make an exponential with mean 5.
    qdist = tfd.QuantizedDistribution(
        distribution=tfd.Exponential(rate=0.2), validate_args=True)
    # Standard error should be less than 1 / (2 * sqrt(n_samples)).
    # We check for results off by 5.2+ standard errors to have an overall false
    # positive rate of ~1e-6.
    n_samples = 10000
    stddev_err_bound = 5.2 / (2 * np.sqrt(n_samples))
    samps = self.evaluate(
        qdist.sample((n_samples,), seed=test_util.test_seed()))
    # The smallest value the samples can take on is 1, which corresponds to
    # the interval (0, 1].  Recall we use ceiling in the sampling definition.
    self.assertLess(0.5, samps.min())
    x_vals = np.arange(1, 11).astype(np.float32)
    pmf_vals = self.evaluate(qdist.prob(x_vals))
    for ii in range(10):
      self.assertAllClose(
          pmf_vals[ii], (samps == x_vals[ii]).mean(), atol=stddev_err_bound)

  def testNormalCdfAndSurvivalFunction(self):
    # At integer values, the result should be the same as the standard normal.
    batch_shape = (3, 3)
    mu = rng.randn(*batch_shape)
    sigma = rng.rand(*batch_shape) + 1.0
    qdist = tfd.QuantizedDistribution(
        distribution=tfd.Normal(loc=mu, scale=sigma), validate_args=True)
    sp_normal = stats.norm(mu, sigma)

    x = rng.randint(-5, 5, size=batch_shape).astype(np.float64)

    self.assertAllClose(sp_normal.cdf(x), self.evaluate(qdist.cdf(x)))

    self.assertAllClose(
        sp_normal.sf(x), self.evaluate(qdist.survival_function(x)))

  def testNormalLogCdfAndLogSurvivalFunction(self):
    # At integer values, the result should be the same as the standard normal.
    batch_shape = (3, 3)
    mu = rng.randn(*batch_shape)
    sigma = rng.rand(*batch_shape) + 1.0
    qdist = tfd.QuantizedDistribution(
        distribution=tfd.Normal(loc=mu, scale=sigma), validate_args=True)
    sp_normal = stats.norm(mu, sigma)

    x = rng.randint(-10, 10, size=batch_shape).astype(np.float64)

    self.assertAllClose(sp_normal.logcdf(x), self.evaluate(qdist.log_cdf(x)))

    self.assertAllClose(
        sp_normal.logsf(x), self.evaluate(qdist.log_survival_function(x)))

  def testNormalProbWithCutoffs(self):
    # At integer values, the result should be the same as the standard normal.
    qdist = tfd.QuantizedDistribution(
        distribution=tfd.Normal(loc=0., scale=1.),
        low=-2.,
        high=2.,
        validate_args=True)
    sm_normal = stats.norm(0., 1.)
    # These cutoffs create partitions of the real line, and indices:
    # (-inf, -2](-2, -1](-1, 0](0, 1](1, inf)
    #        -2      -1      0     1     2
    # Test interval (-inf, -2], <--> index -2.
    self.assertAllClose(
        sm_normal.cdf(-2), self.evaluate(qdist.prob(-2.)), atol=0)
    # Test interval (-2, -1], <--> index -1.
    self.assertAllClose(
        sm_normal.cdf(-1) - sm_normal.cdf(-2),
        self.evaluate(qdist.prob(-1.)),
        atol=0)
    # Test interval (-1, 0], <--> index 0.
    self.assertAllClose(
        sm_normal.cdf(0) - sm_normal.cdf(-1),
        self.evaluate(qdist.prob(0.)),
        atol=0)
    # Test interval (1, inf), <--> index 2.
    self.assertAllClose(
        1. - sm_normal.cdf(1), self.evaluate(qdist.prob(2.)), atol=0)

  def testNormalLogProbWithCutoffs(self):
    # At integer values, the result should be the same as the standard normal.
    qdist = tfd.QuantizedDistribution(
        distribution=tfd.Normal(loc=0., scale=1.),
        low=-2.,
        high=2.,
        validate_args=True)
    sm_normal = stats.norm(0., 1.)
    # These cutoffs create partitions of the real line, and indices:
    # (-inf, -2](-2, -1](-1, 0](0, 1](1, inf)
    #        -2      -1      0     1     2
    # Test interval (-inf, -2], <--> index -2.
    self.assertAllClose(
        np.log(sm_normal.cdf(-2)), self.evaluate(qdist.log_prob(-2.)), atol=0)
    # Test interval (-2, -1], <--> index -1.
    self.assertAllClose(
        np.log(sm_normal.cdf(-1) - sm_normal.cdf(-2)),
        self.evaluate(qdist.log_prob(-1.)),
        atol=0)
    # Test interval (-1, 0], <--> index 0.
    self.assertAllClose(
        np.log(sm_normal.cdf(0) - sm_normal.cdf(-1)),
        self.evaluate(qdist.log_prob(0.)),
        atol=0)
    # Test interval (1, inf), <--> index 2.
    self.assertAllClose(
        np.log(1. - sm_normal.cdf(1)),
        self.evaluate(qdist.log_prob(2.)),
        atol=0)

  def testLogProbAndGradGivesFiniteResults(self):
    def quantized_log_prob(dtype):
      x = np.arange(-100, 100, 2).astype(dtype)

      def inner_func(mu, sigma):
        qdist = tfd.QuantizedDistribution(
            distribution=tfd.Normal(loc=mu, scale=sigma), validate_args=True)
        return qdist.log_prob(x)

      return inner_func

    for dtype in [np.float32, np.float64]:
      mu = tf.constant(0., dtype=dtype, name='mu')
      sigma = tf.constant(1., dtype=dtype, name='sigma')
      value, grads = self.evaluate(tfp.math.value_and_gradient(
          quantized_log_prob(dtype), [mu, sigma]))
      self.assertAllFinite(value)
      self.assertAllFinite(grads)

  def testProbAndGradGivesFiniteResultsForCommonEvents(self):
    def quantized_log_prob(mu, sigma):
      x = tf.math.ceil(4 * rng.rand(100).astype(np.float32) - 2)

      qdist = tfd.QuantizedDistribution(
          distribution=tfd.Normal(loc=mu, scale=sigma), validate_args=True)
      return qdist.log_prob(x)

    mu = tf.constant(0.0, name='mu')
    sigma = tf.constant(1.0, name='sigma')

    value, grads = self.evaluate(tfp.math.value_and_gradient(
        quantized_log_prob, [mu, sigma]))
    self.assertAllFinite(value)
    self.assertAllFinite(grads)

  def testLowerCutoffMustBeBelowUpperCutoffOrWeRaise(self):
    with self.assertRaisesOpError('must be strictly less'):
      qdist = tfd.QuantizedDistribution(
          distribution=tfd.Normal(loc=0., scale=1.),
          low=1.,  # not strictly less than high.
          high=1.,
          validate_args=True)

      self.evaluate(qdist.sample(seed=test_util.test_seed()))

  def testCutoffsMustBeIntegerValuedIfValidateArgsTrue(self):
    low = tf1.placeholder_with_default(1.5, shape=[])
    with self.assertRaisesOpError('has non-integer components.'):
      qdist = tfd.QuantizedDistribution(
          distribution=tfd.Normal(loc=0., scale=1.),
          low=low,
          high=10.,
          validate_args=True)
      self.evaluate(qdist.sample(seed=test_util.test_seed()))

  def testCutoffsCanBeFloatValuedIfValidateArgsFalse(self):
    qdist = tfd.QuantizedDistribution(
        distribution=tfd.Normal(loc=0., scale=1., validate_args=False),
        low=1.5,
        high=10.11)

    self.assertFalse(qdist.validate_args)  # Default is True.

    # Should not raise
    self.evaluate(qdist.sample(seed=test_util.test_seed()))

  def testDtypeAndShapeInheritedFromBaseDist(self):
    batch_shape = (2, 3)
    qdist = tfd.QuantizedDistribution(
        distribution=tfd.Normal(
            loc=tf.zeros(batch_shape), scale=tf.ones(batch_shape)),
        low=1.0,
        high=10.0,
        validate_args=True)

    self.assertEqual(batch_shape, qdist.batch_shape)
    self.assertAllEqual(batch_shape, self.evaluate(qdist.batch_shape_tensor()))
    self.assertEqual((), qdist.event_shape)
    self.assertAllEqual((), self.evaluate(qdist.event_shape_tensor()))

    samps = qdist.sample(10, seed=test_util.test_seed())
    self.assertEqual((10,) + batch_shape, samps.shape)
    self.assertAllEqual((10,) + batch_shape, self.evaluate(samps).shape)

    y = rng.randint(0, 5, size=batch_shape).astype(np.float32)
    self.assertEqual(batch_shape, qdist.prob(y).shape)

  def testAssertValidSample(self):
    qdist = tfd.QuantizedDistribution(
        distribution=tfd.Uniform(low=-3., high=3.),
        low=-1.0,
        high=1.0,
        validate_args=True)
    with self.assertRaisesOpError('Sample has non-integer components.'):
      self.evaluate(qdist.prob([2., -1.7, 0.]))

  def testVariableBounds(self):
    dist = tfd.Normal(0., scale=2.)
    low = tf.Variable(3.)
    high = tf.Variable(-2.)
    self.evaluate([v.initializer for v in [low, high]])
    with self.assertRaisesOpError('must be strictly less than'):
      d = tfd.QuantizedDistribution(
          dist, low=low, high=high, validate_args=True)
      self.evaluate(d.sample(seed=test_util.test_seed()))

  def testVariableLowAfterMutation(self):
    dist = tfd.Normal(0., scale=2.)
    low = tf.Variable(-4.)
    d = tfd.QuantizedDistribution(dist, low=low, high=5., validate_args=True)
    self.evaluate(low.initializer)
    with self.assertRaisesOpError('must be strictly less than'):
      with tf.control_dependencies([low.assign(6.)]):
        self.evaluate(d.sample(seed=test_util.test_seed()))

  def testVariableHighAfterMutation(self):
    dist = tfd.Normal(0., scale=2.)
    high = tf.Variable(5.)
    d = tfd.QuantizedDistribution(dist, low=-4., high=high, validate_args=True)
    self.evaluate(high.initializer)
    with self.assertRaisesOpError('must be strictly less than'):
      with tf.control_dependencies([high.assign(-6.)]):
        self.evaluate(d.log_prob(1.))

  def testVariableNonInteger(self):
    dist = tfd.Normal(0., scale=2.)
    high = tf.Variable(5.2)
    d = tfd.QuantizedDistribution(dist, low=-2., high=high, validate_args=True)
    self.evaluate(high.initializer)
    with self.assertRaisesOpError('has non-integer components.'):
      self.evaluate(d.log_prob(0.))

  def testVariableNonIntegerAfterMutation(self):
    dist = tfd.Normal(0., scale=2.)
    low = tf.Variable(-2.)
    d = tfd.QuantizedDistribution(dist, low=low, high=4., validate_args=True)
    self.evaluate(low.initializer)
    with self.assertRaisesOpError('has non-integer components.'):
      with tf.control_dependencies([low.assign(-3.3)]):
        self.evaluate(d.sample(seed=test_util.test_seed()))

  def testVariableNonScalar(self):
    dist = tfd.Normal(loc=tf.zeros((1, 4)), scale=2.)
    high = tf.Variable(np.random.randint(10, size=(5, 4)).astype(np.float32),
                       shape=tf.TensorShape(None))
    d = tfd.QuantizedDistribution(dist, high=high, validate_args=True)
    self.evaluate(high.initializer)
    with self.assertRaisesOpError('adds extra batch dimensions'):
      self.evaluate(d.sample(seed=test_util.test_seed()))

  def testVariableNonScalarAfterMutation(self):
    dist = tfd.Normal(0., scale=2.)
    low = tf.Variable(-2., shape=tf.TensorShape(None))
    d = tfd.QuantizedDistribution(dist, low=low, validate_args=True)
    self.evaluate(low.initializer)
    with self.assertRaisesOpError('adds extra batch dimensions'):
      with tf.control_dependencies([low.assign([[-2., 5., 1.]])]):
        self.evaluate(d.sample(seed=test_util.test_seed()))

if __name__ == '__main__':
  tf.test.main()
