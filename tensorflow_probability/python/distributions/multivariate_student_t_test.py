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
"""Tests for the MultivariateStudentTLinearOperator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_probability.python.distributions import multivariate_student_t as mvt
from tensorflow_probability.python.distributions import student_t
from tensorflow_probability.python.internal import test_case
from tensorflow_probability.python.internal import test_util as tfp_test_util
from tensorflow.python.framework import test_util


@test_util.run_all_in_graph_and_eager_modes
class MultivariateStudentTTestFloat32StaticShape(
    test_case.TestCase, parameterized.TestCase,
    tfp_test_util.VectorDistributionTestHelpers):
  dtype = tf.float32
  use_static_shape = True

  def _input(self, value):
    """Helper to create inputs with varied dtypes an static shapes."""
    value = tf.cast(value, self.dtype)
    return tf.placeholder_with_default(
        value, shape=value.shape if self.use_static_shape else None)

  # pyformat: disable
  # pylint: disable=bad-whitespace
  @parameterized.parameters(
      # loc           df      diag          batch_shape
      ([0., 0.],     1.,     [1., 1.],     []),
      (0.,           1.,     [1., 1.],     []),
      ([[[0., 0.]]], 1.,     [1., 1.],     [1, 1]),
      ([0., 0.],     [[1.]], [1., 1.],     [1, 1]),
      ([0., 0.],     1.,     [[[1., 1.]]], [1, 1]),
      ([[[0., 0.]]], [[1.]], [[[1., 1.]]], [1, 1]),
  )
  # pylint: enable=bad-whitespace
  # pyformat: enable
  def testBroadcasting(self, loc, df, diag, batch_shape):
    # Test that broadcasting works across all 3 parameters.
    loc = self._input(loc)
    df = self._input(df)
    diag = self._input(diag)

    scale = tf.linalg.LinearOperatorDiag(diag, is_positive_definite=True)
    dist = mvt.MultivariateStudentTLinearOperator(
        loc=loc, df=df, scale=scale, validate_args=True)

    sample = dist.sample(3)
    log_prob = dist.log_prob(sample)
    mean = dist.mean()
    mode = dist.mode()
    cov = dist.covariance()
    std = dist.stddev()
    var = dist.variance()
    entropy = dist.entropy()
    if self.use_static_shape:
      self.assertAllEqual([3] + batch_shape + [2], sample.shape)
      self.assertAllEqual([3] + batch_shape, log_prob.shape)
      self.assertAllEqual(batch_shape + [2], mean.shape)
      self.assertAllEqual(batch_shape + [2], mode.shape)
      self.assertAllEqual(batch_shape + [2, 2], cov.shape)
      self.assertAllEqual(batch_shape + [2], std.shape)
      self.assertAllEqual(batch_shape + [2], var.shape)
      self.assertAllEqual(batch_shape, entropy.shape)
      self.assertAllEqual([2], dist.event_shape)
      self.assertAllEqual(batch_shape, dist.batch_shape)

    sample = self.evaluate(sample)
    log_prob = self.evaluate(log_prob)
    mean = self.evaluate(mean)
    mode = self.evaluate(mode)
    cov = self.evaluate(cov)
    std = self.evaluate(std)
    var = self.evaluate(var)
    entropy = self.evaluate(entropy)
    self.assertAllEqual([3] + batch_shape + [2], sample.shape)
    self.assertAllEqual([3] + batch_shape, log_prob.shape)
    self.assertAllEqual(batch_shape + [2], mean.shape)
    self.assertAllEqual(batch_shape + [2], mode.shape)
    self.assertAllEqual(batch_shape + [2, 2], cov.shape)
    self.assertAllEqual(batch_shape + [2], std.shape)
    self.assertAllEqual(batch_shape + [2], var.shape)
    self.assertAllEqual(batch_shape, entropy.shape)
    self.assertAllEqual([2], self.evaluate(dist.event_shape_tensor()))
    self.assertAllEqual(batch_shape, self.evaluate(dist.batch_shape_tensor()))

  def testNonPositiveDf(self):
    with self.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                                 "`df` must be positive"):
      self.evaluate(
          mvt.MultivariateStudentTLinearOperator(
              loc=self._input([0.]),
              df=self._input(0.),
              scale=tf.linalg.LinearOperatorDiag(
                  self._input([1.]), is_positive_definite=True),
              validate_args=True).df)

  def testBadScaleDType(self):
    with self.assertRaisesRegexp(TypeError,
                                 "`scale` must have floating-point dtype."):
      mvt.MultivariateStudentTLinearOperator(
          loc=[0.],
          df=1.,
          scale=tf.linalg.LinearOperatorIdentity(
              num_rows=1, dtype=tf.int32, is_positive_definite=True))

  def testNotPositiveDefinite(self):
    with self.assertRaisesRegexp(ValueError,
                                 "`scale` must be positive definite."):
      mvt.MultivariateStudentTLinearOperator(
          loc=self._input([0.]),
          df=self._input(1.),
          scale=tf.linalg.LinearOperatorDiag(self._input([1.])),
          validate_args=True)

  def testMeanAllDefined(self):
    dist = mvt.MultivariateStudentTLinearOperator(
        loc=self._input([0., 0.]),
        df=self._input(2.),
        scale=tf.linalg.LinearOperatorDiag(self._input([1., 1.])))
    mean = self.evaluate(dist.mean())
    self.assertAllClose([0., 0.], mean)

  def testMeanSomeUndefinedNaNAllowed(self):
    dist = mvt.MultivariateStudentTLinearOperator(
        loc=self._input([[0., 0.], [1., 1.]]),
        df=self._input([1., 2.]),
        scale=tf.linalg.LinearOperatorDiag(self._input([[1., 1.], [1., 1.]])),
        allow_nan_stats=True)
    mean = self.evaluate(dist.mean())
    self.assertAllClose([[np.nan, np.nan], [1., 1.]], mean)

  def testMeanSomeUndefinedNaNNotAllowed(self):
    dist = mvt.MultivariateStudentTLinearOperator(
        loc=self._input([[0., 0.], [1., 1.]]),
        df=self._input([1., 2.]),
        scale=tf.linalg.LinearOperatorDiag(self._input([[1., 1.], [1., 1.]])),
        allow_nan_stats=False)
    with self.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                                 "mean not defined for components of df <= 1"):
      self.evaluate(dist.mean())

  def testMode(self):
    dist = mvt.MultivariateStudentTLinearOperator(
        loc=[0., 0.], df=2., scale=tf.linalg.LinearOperatorDiag([[1., 1.]]))
    mode = self.evaluate(dist.mode())
    self.assertAllClose([[0., 0.]], mode)

  # pyformat: disable
  # pylint: disable=bad-whitespace
  @parameterized.parameters(
      # diag      full                  expected_mvn_cov
      ([2., 2.], None,                 [[4., 0.], [0., 4.]]),
      (None,     [[2., 1.], [1., 2.]], [[5., 4.], [4., 5.]]),
  )
  # pyformat: enable
  # pylint: enable=bad-whitespace
  def testCovarianceAllDefined(self,
                               diag=None,
                               full=None,
                               expected_mvn_cov=None):
    if diag is not None:
      scale = tf.linalg.LinearOperatorDiag(self._input(diag))
    else:
      scale = tf.linalg.LinearOperatorFullMatrix(self._input(full))
    dist = mvt.MultivariateStudentTLinearOperator(
        loc=self._input([0., 0.]), df=self._input(3.), scale=scale)
    cov = self.evaluate(dist.covariance())
    self.assertAllClose(np.array(expected_mvn_cov) * 3. / (3. - 2.), cov)

  def testCovarianceSomeUndefinedNaNAllowed(self):
    scale = tf.linalg.LinearOperatorDiag(self._input([2., 2.]))
    dist = mvt.MultivariateStudentTLinearOperator(
        loc=self._input([0., 0.]),
        df=self._input([2., 1.]),
        scale=scale,
        allow_nan_stats=True)
    cov = self.evaluate(dist.covariance())
    self.assertAllClose(np.full([2, 2], np.inf), cov[0])
    self.assertAllClose(np.full([2, 2], np.nan), cov[1])

  def testCovarianceSomeUndefinedNaNNotAllowed(self):
    scale = tf.linalg.LinearOperatorDiag(self._input([2., 2.]))
    dist = mvt.MultivariateStudentTLinearOperator(
        loc=self._input([0., 0.]),
        df=self._input(1.),
        scale=scale,
        allow_nan_stats=False)
    with self.assertRaisesRegexp(
        tf.errors.InvalidArgumentError,
        "covariance not defined for components of df <= 1"):
      self.evaluate(dist.covariance())

  # pyformat: disable
  # pylint: disable=bad-whitespace
  @parameterized.parameters(
      # diag     full                  update       expected_mvn_var
      ([2., 2.], None,                 None,        [4., 4.]),
      (None,     [[2., 1.], [1., 2.]], None,        [5., 5.]),
      ([2., 2.], None,                 [[1.],[1.]], [10., 10.]),
  )
  # pylint: enable=bad-whitespace
  # pyformat: enable
  def testVarianceStdAllDefined(self,
                                diag=None,
                                full=None,
                                update=None,
                                expected_mvn_var=None):
    if diag is not None:
      scale = tf.linalg.LinearOperatorDiag(self._input(diag))
    elif full is not None:
      scale = tf.linalg.LinearOperatorFullMatrix(self._input(full))
    if update is not None:
      scale = tf.linalg.LinearOperatorLowRankUpdate(scale, self._input(update))

    dist = mvt.MultivariateStudentTLinearOperator(
        loc=self._input([0., 0.]), df=self._input(3.), scale=scale)
    var = self.evaluate(dist.variance())
    std = self.evaluate(dist.stddev())
    # df = 3, so we expect the variance of the MVT to exceed MVN by a factor of
    # 3 / (3 - 2) = 3.
    self.assertAllClose(np.array(expected_mvn_var) * 3., var)
    self.assertAllClose(np.sqrt(np.array(expected_mvn_var) * 3.), std)

  def testVarianceStdSomeUndefinedNaNAllowed(self):
    scale = tf.linalg.LinearOperatorDiag(self._input([2., 2.]))
    dist = mvt.MultivariateStudentTLinearOperator(
        loc=self._input([0., 0.]),
        df=self._input([2., 1.]),
        scale=scale,
        allow_nan_stats=True)
    var = self.evaluate(dist.variance())
    std = self.evaluate(dist.stddev())
    self.assertAllClose([np.inf, np.inf], var[0])
    self.assertAllClose([np.nan, np.nan], var[1])
    self.assertAllClose([np.inf, np.inf], std[0])
    self.assertAllClose([np.nan, np.nan], std[1])

  def testVarianceStdSomeUndefinedNaNNotAllowed(self):
    scale = tf.linalg.LinearOperatorDiag(self._input([2., 2.]))
    dist = mvt.MultivariateStudentTLinearOperator(
        loc=self._input([0., 0.]),
        df=self._input(1.),
        scale=scale,
        allow_nan_stats=False)
    with self.assertRaisesRegexp(
        tf.errors.InvalidArgumentError,
        "variance not defined for components of df <= 1"):
      self.evaluate(dist.variance())
    with self.assertRaisesRegexp(
        tf.errors.InvalidArgumentError,
        "standard deviation not defined for components of df <= 1"):
      self.evaluate(dist.stddev())

  def testEntropy(self):
    scale = tf.linalg.LinearOperatorDiag(self._input([2., 2.]))
    dist = mvt.MultivariateStudentTLinearOperator(
        loc=self._input([0., 0.]), df=self._input([2., 3.]), scale=scale)
    # From Kotz S. and Nadarajah S. (2004). Multivariate t Distributions and
    # Their Applications. Cambridge University Press. p22.
    self.assertAllClose(
        [0.5 * np.log(16.) + 3.83788, 0.5 * np.log(16.) + 3.50454],
        dist.entropy())

  def testSamplingConsistency(self):
    # pyformat: disable
    scale = tf.linalg.LinearOperatorFullMatrix(self._input(
        [[2., -1.],
         [-1., 2.]]))
    # pyformat: enable
    dist = mvt.MultivariateStudentTLinearOperator(
        loc=self._input([1., 2.]), df=self._input(5.), scale=scale)
    self.run_test_sample_consistent_mean_covariance(
        sess_run_fn=self.evaluate, dist=dist)

  def testSamplingDeterministic(self):
    # pyformat: disable
    scale = tf.linalg.LinearOperatorFullMatrix(self._input(
        [[2., -1.],
         [-1., 2.]]))
    # pyformat: enable
    tf.set_random_seed(2)
    dist1 = mvt.MultivariateStudentTLinearOperator(
        loc=[1., 2.], df=5., scale=scale)
    samples1 = self.evaluate(dist1.sample(100, seed=1))
    tf.set_random_seed(2)
    dist2 = mvt.MultivariateStudentTLinearOperator(
        loc=[1., 2.], df=5., scale=scale)
    samples2 = self.evaluate(dist2.sample(100, seed=1))
    self.assertAllClose(samples1, samples2)

  def testSamplingFullyReparameterized(self):
    df = self._input(2.)
    loc = self._input([1., 2.])
    diag = self._input([3., 4.])
    with tf.GradientTape() as tape:
      tape.watch(df)
      tape.watch(loc)
      tape.watch(diag)
      scale = tf.linalg.LinearOperatorDiag(diag)
      dist = mvt.MultivariateStudentTLinearOperator(loc=loc, df=df, scale=scale)
      samples = dist.sample(100)
    grad_df, grad_loc, grad_diag = tape.gradient(samples, [df, loc, diag])
    self.assertIsNotNone(grad_df)
    self.assertIsNotNone(grad_loc)
    self.assertIsNotNone(grad_diag)

  def testSamplingSmallDfNoNaN(self):
    scale = tf.linalg.LinearOperatorDiag(self._input([1., 1.]))
    dist = mvt.MultivariateStudentTLinearOperator(
        loc=self._input([0., 0.]),
        df=self._input([1e-1, 1e-5, 1e-10, 1e-20]),
        scale=scale)
    samples = dist.sample(int(2e5), seed=1)
    log_probs = dist.log_prob(samples)
    samples, log_probs = self.evaluate([samples, log_probs])
    self.assertTrue(np.all(np.isfinite(samples)))
    self.assertTrue(np.all(np.isfinite(log_probs)))

  def testLogProb(self):
    # Test that numerically integrating over some portion of the domain yields a
    # normalization constant of close to 1.
    # pyformat: disable
    scale = tf.linalg.LinearOperatorFullMatrix(
        self._input([[1.,  -0.5],
                     [-0.5, 1.]]))
    # pyformat: enable
    dist = mvt.MultivariateStudentTLinearOperator(
        loc=self._input([1., 1.]), df=self._input(5.), scale=scale)

    spacings = tf.cast(tf.linspace(-20., 20., 100), self.dtype)
    x, y = tf.meshgrid(spacings, spacings)
    points = tf.concat([x[..., tf.newaxis], y[..., tf.newaxis]], -1)
    log_probs = dist.log_prob(points)
    normalization = tf.exp(
        tf.reduce_logsumexp(log_probs)) * (spacings[1] - spacings[0])**2
    self.assertAllClose(1., self.evaluate(normalization), atol=1e-3)

    mode_log_prob = dist.log_prob(dist.mode())
    self.assertTrue(np.all(self.evaluate(mode_log_prob >= log_probs)))

  @parameterized.parameters(1., 3., 10.)
  def testHypersphereVolume(self, radius):
    # pyformat: disable
    scale = tf.linalg.LinearOperatorFullMatrix(
        self._input([[1.,  -0.5],
                     [-0.5, 1.]]))
    # pyformat: enable
    dist = mvt.MultivariateStudentTLinearOperator(
        loc=self._input([1., 1.]), df=self._input(4.), scale=scale)
    self.run_test_sample_consistent_log_prob(
        sess_run_fn=self.evaluate,
        dist=dist,
        radius=radius,
        num_samples=int(5e6),
        rtol=0.05)

  def testLogProbSameFor1D(self):
    # 1D MVT is exactly a regular Student's T distribution.
    t_dist = student_t.StudentT(
        df=self._input(5.), loc=self._input(2.), scale=self._input(3.))
    scale = tf.linalg.LinearOperatorDiag([self._input(3.)])
    mvt_dist = mvt.MultivariateStudentTLinearOperator(
        loc=[self._input(2.)], df=self._input(5.), scale=scale)

    test_points = tf.cast(tf.linspace(-10.0, 10.0, 100), self.dtype)

    t_log_probs = self.evaluate(t_dist.log_prob(test_points))
    mvt_log_probs = self.evaluate(
        mvt_dist.log_prob(test_points[..., tf.newaxis]))

    self.assertAllClose(t_log_probs, mvt_log_probs)


class MultivariateStudentTTestFloat64StaticShape(
    MultivariateStudentTTestFloat32StaticShape):
  dtype = tf.float64
  use_static_shape = True


class MultivariateStudentTTestFloat32DynamicShape(
    MultivariateStudentTTestFloat32StaticShape):
  dtype = tf.float32
  use_static_shape = False


if __name__ == "__main__":
  tf.test.main()
