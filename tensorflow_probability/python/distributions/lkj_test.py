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
"""Tests for the LKJ distribution."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.distributions.internal import statistical_testing as st
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import test_util


def _det_ok_mask(x, det_bounds, input_output_cholesky=False):
  if input_output_cholesky:
    logdet = 2.0 * tf.reduce_sum(
        tf.math.log(tf.linalg.diag_part(x)), axis=[-1])
  else:
    _, logdet = tf.linalg.slogdet(x)

  return tf.cast(tf.exp(logdet) > det_bounds, dtype=x.dtype)

# Each leaf entry here is a confidence interval for the volume of some
# set of correlation matrices.  To wit, k-by-k correlation matrices
# whose determinant is at least d appear as volume_bounds[k][d].
# These particular confidence intervals were estimated by the
# Clopper-Pearson method applied to 10^7 rejection samples, with an
# error probability of 5e-7.  This computation may be performed by
# executing the correlation_matrix_volumes program with argument
# --num_samples 1e7.  Doing so took about 45 minutes on a standard
# workstation.
volume_bounds = {
    3: {0.01: (04.8334339757361420, 4.845866340472709),
        0.25: (02.9993127232473036, 3.011629093880439),
        0.30: (02.6791373340121916, 2.691146382760893),
        0.35: (02.3763254004846030, 2.3879545568875358),
        0.40: (02.0898224112869355, 2.1010041316917913),
        0.45: (01.8202389505755674, 1.8309117190894892)},
    4: {0.01: (10.983339932556953, 11.060156130783517),
        0.25: (03.4305021152837020, 3.4764695469900464),
        0.30: (02.6624323207206930, 2.703204389589173),
        0.35: (02.0431263321809440, 2.0790437132708752),
        0.40: (01.5447440594930320, 1.5761221057556805),
        0.45: (01.1459065289947180, 1.1730410135527702)},
    5: {0.01: (19.081135276668707, 19.523821224876603),
        0.20: (02.8632254471072285, 3.0376848112309776),
        0.25: (01.8225680180604158, 1.9623522646052605),
        0.30: (01.1299612119639912, 1.2406126830051296),
        0.35: (00.6871928383147943, 0.7740705901566753),
        0.40: (00.4145900446719042, 0.482655106057178)}}


@test_util.test_all_tf_execution_regimes
@parameterized.parameters(np.float32, np.float64)
class LKJTest(test_util.TestCase):

  def testNormConst2D(self, dtype):
    expected = 2.
    # 2x2 correlation matrices are determined by one number between -1
    # and 1, so the volume of density 1 over all of them is 2.
    answer = self.evaluate(
        tfd.LKJ(2, dtype([1.]), validate_args=True)._log_normalization())
    self.assertAllClose(answer, np.log([expected]))

  def testNormConst3D(self, dtype):
    expected = np.pi**2 / 2.
    # 3x3 correlation matrices are determined by the three
    # lower-triangular entries.  In addition to being between -1 and
    # 1, they must also obey the constraint that the determinant of
    # the resulting symmetric matrix is non-negative.  The post
    # https://psychometroscar.com/the-volume-of-a-3-x-3-correlation-matrix/
    # derives (with elementary calculus) that the volume of this set
    # (with respect to Lebesgue^3 measure) is pi^2/2.  The same result
    # is also obtained by Rousseeuw, P. J., & Molenberghs,
    # G. (1994). "The shape of correlation matrices." The American
    # Statistician, 48(4), 276-279.
    answer = self.evaluate(
        tfd.LKJ(3, dtype([1.]), validate_args=True)._log_normalization())
    self.assertAllClose(answer, np.log([expected]))

  def _testSampleLogProbExact(self,
                              concentrations,
                              det_bounds,
                              dim,
                              means,
                              num_samples=int(1e5),
                              dtype=np.float32,
                              target_discrepancy=0.1,
                              input_output_cholesky=False,
                              seed=42):
    # For test methodology see the comment in
    # _testSampleConsistentLogProbInterval, except that this test
    # checks those parameter settings where the true volume is known
    # analytically.
    concentration = np.array(concentrations, dtype=dtype)
    det_bounds = np.array(det_bounds, dtype=dtype)
    means = np.array(means, dtype=dtype)
    # Add a tolerance to guard against some of the importance_weights exceeding
    # the theoretical maximum (importance_maxima) due to numerical inaccuracies
    # while lower bounding the determinant. See corresponding comment in
    # _testSampleConsistentLogProbInterval.
    high_tolerance = 1e-6

    testee_lkj = tfd.LKJ(
        dimension=dim,
        concentration=concentration,
        input_output_cholesky=input_output_cholesky,
        validate_args=True)
    x = testee_lkj.sample(num_samples, seed=seed)
    importance_weights = (
        tf.exp(-testee_lkj.log_prob(x)) * _det_ok_mask(x, det_bounds,
                                                       input_output_cholesky))
    importance_maxima = (1. / det_bounds) ** (concentration - 1) * tf.exp(
        testee_lkj._log_normalization())

    chk1 = st.assert_true_mean_equal_by_dkwm(
        importance_weights, low=0., high=importance_maxima + high_tolerance,
        expected=means, false_fail_rate=1e-6)
    chk2 = assert_util.assert_less(
        st.min_discrepancy_of_true_means_detectable_by_dkwm(
            num_samples,
            low=0.,
            high=importance_maxima + high_tolerance,
            false_fail_rate=1e-6,
            false_pass_rate=1e-6), dtype(target_discrepancy))
    self.evaluate([chk1, chk2])

  def testSampleConsistentLogProb2(self, dtype):
    concentrations = np.array([
        1.00, 1.30, 1.50, 1.70, 1.90, 2.00, 2.10, 2.50, 3.00])
    det_bounds = np.array([
        0.01, 0.25, 0.30, 0.40, 0.50, 0.50, 0.50, 0.70, 0.70])
    exact_volumes = 2 * np.sqrt(1. - det_bounds)

    for input_output_cholesky in [True, False]:
      self._testSampleLogProbExact(
          concentrations,
          det_bounds,
          2,
          exact_volumes,
          num_samples=int(1.1e5),
          dtype=dtype,
          input_output_cholesky=input_output_cholesky,
          target_discrepancy=0.05,
          seed=test_util.test_seed())

  def _testSampleConsistentLogProbInterval(self,
                                           concentrations,
                                           det_bounds,
                                           dim,
                                           num_samples=int(1e5),
                                           dtype=np.float32,
                                           input_output_cholesky=False,
                                           false_fail_rate=1e-6,
                                           target_discrepancy=0.1,
                                           seed=42):
    # Consider the set M of dim x dim correlation matrices whose
    # determinant exceeds some bound (rationale for bound forthwith).
    # - This is a (convex!) shape in dim * (dim - 1) / 2 dimensions
    #   (because a correlation matrix is determined by its lower
    #   triangle, and the main diagonal is all 1s).
    # - Further, M is contained entirely in the [-1,1] cube,
    #   because no correlation can fall outside that interval.
    #
    # We have two different ways to estimate the volume of M:
    # - Importance sampling from the LKJ distribution
    # - Importance sampling from the uniform distribution on the cube
    #
    # This test checks that these two methods agree.  However, because
    # the uniform proposal leads to many rejections (thus slowness),
    # those volumes are computed offline and the confidence intervals
    # are presented to this test procedure in the "volume_bounds"
    # table.
    #
    # Why place a lower bound on the determinant?  Because for eta > 1,
    # the density of LKJ approaches 0 as the determinant approaches 0.
    # However, the test methodology requires an upper bound on the
    # improtance weights produced.  Rejecting matrices with too-small
    # determinant (from both methods) allows me to supply that bound.
    #
    # I considered several alternative regions whose volume I might
    # know analytically (without having to do rejection).
    # - Option a: Some hypersphere guaranteed to be contained inside M.
    #   - Con: I don't know a priori how to find a radius for it.
    #   - Con: I still need a lower bound on the determinants that appear
    #     in this sphere, and I don't know how to compute it.
    # - Option b: Some trapezoid given as the convex hull of the
    #   nearly-extreme correlation matrices (i.e., those that partition
    #   the variables into two strongly anti-correclated groups).
    #   - Con: Would have to dig up n-d convex hull code to implement this.
    #   - Con: Need to compute the volume of that convex hull.
    #   - Con: Need a bound on the determinants of the matrices in that hull.
    # - Option c: Same thing, but with the matrices that make a single pair
    #   of variables strongly correlated (or anti-correlated), and leaves
    #   the others uncorrelated.
    #   - Same cons, except that there is a determinant bound (which
    #     felt pretty loose).
    lows = [dtype(volume_bounds[dim][db][0]) for db in det_bounds]
    highs = [dtype(volume_bounds[dim][db][1]) for db in det_bounds]
    concentration = np.array(concentrations, dtype=dtype)
    det_bounds = np.array(det_bounds, dtype=dtype)
    # Due to possible numerical inaccuracies while lower bounding the
    # determinant, the maximum of the importance weights may exceed the
    # theoretical maximum (importance_maxima). We add a tolerance to guard
    # against this. An alternative would have been to add a threshold while
    # filtering in _det_ok_mask, but that would affect the mean as well.
    high_tolerance = 1e-6

    testee_lkj = tfd.LKJ(
        dimension=dim,
        concentration=concentration,
        input_output_cholesky=input_output_cholesky,
        validate_args=True)
    x = testee_lkj.sample(num_samples, seed=seed)
    importance_weights = (
        tf.exp(-testee_lkj.log_prob(x)) * _det_ok_mask(x, det_bounds,
                                                       input_output_cholesky))
    importance_maxima = (1. / det_bounds) ** (concentration - 1) * tf.exp(
        testee_lkj._log_normalization())
    check1 = st.assert_true_mean_in_interval_by_dkwm(
        samples=importance_weights,
        low=0.,
        high=importance_maxima + high_tolerance,
        expected_low=lows,
        expected_high=highs,
        false_fail_rate=false_fail_rate)
    check2 = assert_util.assert_less(
        st.min_discrepancy_of_true_means_detectable_by_dkwm(
            num_samples,
            low=0.,
            high=importance_maxima + high_tolerance,
            false_fail_rate=false_fail_rate,
            false_pass_rate=false_fail_rate), dtype(target_discrepancy))
    self.evaluate([check1, check2])

  def testSampleConsistentLogProbInterval3(self, dtype):
    # The hardcoded volume boundaries are (5e-7)-confidence intervals
    # of a rejection sampling run.  Ergo, I only have 5e-7 probability
    # mass left for the false fail rate of the test so the aggregate
    # false fail probability is 1e-6.
    concentrations = [
        1.00, 1.30, 1.50, 1.70, 1.90, 2.00, 2.10, 2.50, 3.00]
    det_bounds = [
        0.01, 0.25, 0.25, 0.30, 0.35, 0.35, 0.35, 0.40, 0.45]

    for input_output_cholesky in [True, False]:
      self._testSampleConsistentLogProbInterval(
          concentrations,
          det_bounds,
          3,
          dtype=dtype,
          input_output_cholesky=input_output_cholesky,
          false_fail_rate=5e-7,
          target_discrepancy=0.11,
          seed=test_util.test_seed())

  def testSampleConsistentLogProbInterval4(self, dtype):
    # The hardcoded volume boundaries are (5e-7)-confidence intervals
    # of a rejection sampling run.  Ergo, I only have 5e-7 probability
    # mass left for the false fail rate of the test so the aggregate
    # false fail probability is 1e-6.
    concentrations = [
        1.00, 1.30, 1.50, 1.70, 1.90, 2.00, 2.10, 2.50, 3.00]
    det_bounds = [
        0.01, 0.25, 0.25, 0.30, 0.35, 0.35, 0.35, 0.40, 0.45]
    for input_output_cholesky in [True, False]:
      self._testSampleConsistentLogProbInterval(
          concentrations,
          det_bounds,
          4,
          dtype=dtype,
          input_output_cholesky=input_output_cholesky,
          false_fail_rate=5e-7,
          target_discrepancy=0.22,
          seed=test_util.test_seed())

  def testSampleConsistentLogProbInterval5(self, dtype):
    # The hardcoded volume boundaries are (5e-7)-confidence intervals
    # of a rejection sampling run.  Ergo, I only have 5e-7 probability
    # mass left for the false fail rate of the test so the aggregate
    # false fail probability is 1e-6.
    concentrations = [
        1.00, 1.30, 1.50, 1.70, 1.90, 2.00, 2.10, 2.50, 3.00]
    det_bounds = [
        0.01, 0.20, 0.20, 0.25, 0.30, 0.30, 0.30, 0.35, 0.40]

    for input_output_cholesky in [True, False]:
      self._testSampleConsistentLogProbInterval(
          concentrations,
          det_bounds,
          5,
          dtype=dtype,
          input_output_cholesky=input_output_cholesky,
          false_fail_rate=5e-7,
          target_discrepancy=0.41,
          seed=test_util.test_seed())

  def testDimensionGuard(self, dtype):
    testee_lkj = tfd.LKJ(
        dimension=3, concentration=dtype([1., 4.]), validate_args=True)
    with self.assertRaisesRegexp(ValueError, 'dimension mismatch'):
      testee_lkj.log_prob(dtype(np.eye(4)))

  def testAssertValidCorrelationMatrix(self, dtype):
    lkj = tfd.LKJ(
        dimension=2, concentration=dtype([1., 4.]), validate_args=True)
    with self.assertRaisesOpError('Correlations must be >= -1.'):
      self.evaluate(lkj.log_prob(dtype([[1., -1.3], [-1.3, 1.]])))
    with self.assertRaisesOpError('Correlations must be <= 1.'):
      self.evaluate(lkj.log_prob(dtype([[1., 1.3], [1.3, 1.]])))
    with self.assertRaisesOpError('Self-correlations must be = 1.'):
      self.evaluate(lkj.log_prob(dtype([[0.5, 0.5], [0.5, 1.]])))
    with self.assertRaisesOpError('Correlation matrices must be symmetric.'):
      self.evaluate(lkj.log_prob(dtype([[1., 0.2], [0.3, 1.]])))

  def testZeroDimension(self, dtype):
    testee_lkj = tfd.LKJ(
        dimension=0, concentration=dtype([1., 4.]), validate_args=True)
    results = testee_lkj.sample(sample_shape=[4, 3], seed=test_util.test_seed())
    self.assertEqual(results.shape, [4, 3, 2, 0, 0])

  def testOneDimension(self, dtype):
    testee_lkj = tfd.LKJ(
        dimension=1, concentration=dtype([1., 4.]), validate_args=True)
    results = testee_lkj.sample(sample_shape=[4, 3], seed=test_util.test_seed())
    self.assertEqual(results.shape, [4, 3, 2, 1, 1])

  def testMean(self, dtype):
    testee_lkj = tfd.LKJ(
        dimension=3, concentration=dtype([1., 3., 5.]), validate_args=True)
    num_samples = 20000
    results = testee_lkj.sample(
        sample_shape=[num_samples], seed=test_util.test_seed())
    mean = testee_lkj.mean()
    self.assertEqual(mean.shape, [3, 3, 3])
    check1 = st.assert_true_mean_equal_by_dkwm(
        samples=results, low=-1., high=1.,
        expected=mean,
        false_fail_rate=1e-6)
    check2 = assert_util.assert_less(
        st.min_discrepancy_of_true_means_detectable_by_dkwm(
            num_samples,
            low=-1.,
            high=1.,
            # Smaller false fail rate because of different batch sizes between
            # these two checks.
            false_fail_rate=1e-7,
            false_pass_rate=1e-6),
        # 4% relative error
        0.08)
    self.evaluate([check1, check2])

  def testValidateConcentration(self, dtype):
    dimension = 3
    concentration = tf.Variable(0.5, dtype=dtype)
    d = tfd.LKJ(dimension, concentration, validate_args=True)
    with self.assertRaisesOpError('Argument `concentration` must be >= 1.'):
      self.evaluate([v.initializer for v in d.variables])
      self.evaluate(d.sample(seed=test_util.test_seed()))

  def testValidateConcentrationAfterMutation(self, dtype):
    dimension = 3
    concentration = tf.Variable(1.5, dtype=dtype)
    d = tfd.LKJ(dimension, concentration, validate_args=True)
    self.evaluate([v.initializer for v in d.variables])
    with self.assertRaisesOpError('Argument `concentration` must be >= 1.'):
      with tf.control_dependencies([concentration.assign(0.5)]):
        self.evaluate(d.mean())


class LKJTestGraphOnly(test_util.TestCase):

  def testDimensionGuardDynamicShape(self):
    if tf.executing_eagerly():
      return
    testee_lkj = tfd.LKJ(
        dimension=3, concentration=[1., 4.], validate_args=True)
    with self.assertRaisesOpError('dimension mismatch'):
      self.evaluate(
          testee_lkj.log_prob(
              tf1.placeholder_with_default(tf.eye(4), shape=None)))


if __name__ == '__main__':
  tf.test.main()
