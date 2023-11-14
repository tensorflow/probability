# Copyright 2020 The TensorFlow Probability Authors.
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
"""Tests for LambertWDistributions."""

import math
# Dependency imports
from absl.testing import parameterized
import numpy as np
from scipy import stats
import tensorflow.compat.v2 as tf
from tensorflow_probability.python.bijectors import lambertw_transform
from tensorflow_probability.python.distributions import lambertw_f
from tensorflow_probability.python.distributions import normal
from tensorflow_probability.python.distributions import student_t
from tensorflow_probability.python.distributions import transformed_distribution
from tensorflow_probability.python.distributions import uniform
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.internal import tf_keras


@test_util.test_all_tf_execution_regimes
class LambertWDistributionTest(test_util.TestCase):

  @parameterized.named_parameters(
      ("Normal", normal.Normal, {"loc": 0., "scale": 1.}),
      ("T", student_t.StudentT, {"df": 5, "loc": 0., "scale": 1.}),
      ("Uniform", uniform.Uniform, {
          "low": -math.sqrt(3.), "high": math.sqrt(3.)}))
  def testLambertWDistributionOutputType(self,
                                         distribution_constructor,
                                         distribution_kwargs):
    """Tests that the output of lambertw_distribution is of the correct type."""
    shift = 1.0
    scale = 2.0
    tail = 1.0

    lambertw_distribution = lambertw_f.LambertWDistribution(
        distribution=distribution_constructor(**distribution_kwargs),
        shift=shift,
        scale=scale,
        tailweight=tail)
    self.assertIsInstance(
        lambertw_distribution, transformed_distribution.TransformedDistribution)
    samples = lambertw_distribution.sample(10, seed=test_util.test_seed())
    self.assertAllEqual(samples.shape, [10])


@test_util.test_all_tf_execution_regimes
class LambertWNormalTest(test_util.TestCase):

  def testShapes(self):
    loc = tf.ones([2], dtype=tf.float32)
    scale = tf.ones([3, 1], dtype=tf.float32)
    tailweight = tf.zeros([4, 1, 1], dtype=tf.float32)
    lwn = lambertw_f.LambertWNormal(
        loc=loc,
        scale=scale,
        tailweight=tailweight,
        validate_args=True)
    self.assertAllEqual(lwn.batch_shape, [4, 3, 2])

  def testBatchSamplesAreIndependent(self):
    num_samples = 1000
    lwn = lambertw_f.LambertWNormal(loc=0., scale=1., tailweight=[0., 0.])
    xs = lwn.sample(num_samples, seed=test_util.test_seed())
    cov = 1. / num_samples * tf.matmul(xs, xs, transpose_a=True)
    self.assertAllClose(cov, tf.eye(2), atol=0.2)

  def testSampleMethod(self):
    """Tests that samples can be correctly transformed into gaussian samples."""
    tailweight = .1
    loc = 2.
    scale = 3.
    lwn = lambertw_f.LambertWNormal(loc=loc, scale=scale, tailweight=tailweight)
    samples = lwn.sample(100, seed=test_util.test_seed)

    self.assertAllEqual(samples.shape, [100])
    gaussianized_samples = lambertw_transform.LambertWTail(
        shift=loc, scale=scale, tailweight=tailweight).inverse(samples)
    _, p = stats.normaltest(self.evaluate(gaussianized_samples))
    self.assertGreater(p, .05)

  def testProbMethod(self):
    """Tests that Lambert W pdf is the same as Normal pdf when tail is zero."""
    tailweight = 0.
    loc = 2.
    scale = 3.
    lwn = lambertw_f.LambertWNormal(loc=loc, scale=scale, tailweight=tailweight)
    values = np.random.normal(loc=2., scale=3., size=10)
    normal_pdf = normal.Normal(loc, scale).prob(values)
    lambertw_normal_pdf = lwn.prob(values)
    self.assertAllClose(normal_pdf, lambertw_normal_pdf)

  def testCDFMethod(self):
    """Tests that output of the cdf method is correct."""
    tailweight = 0.
    loc = 2.
    scale = 3.
    lwn = lambertw_f.LambertWNormal(loc=loc, scale=scale, tailweight=tailweight)

    values = np.random.uniform(low=0.0, high=1.0, size=10)
    quantiles = normal.Normal(loc, scale).quantile(values)
    heavy_tailed_values = lambertw_transform.LambertWTail(
        shift=loc, scale=scale, tailweight=tailweight)(quantiles)
    lambertw_normal_values = lwn.cdf(heavy_tailed_values)
    self.assertAllClose(values, lambertw_normal_values)

  def testQuantileMethod(self):
    """Tests that quantiles are correct."""
    tailweight = 0.
    loc = 2.
    scale = 3.
    lwn = lambertw_f.LambertWNormal(loc=loc, scale=scale, tailweight=tailweight)

    values = np.random.uniform(low=0.0, high=1.0, size=5)
    normal_quantiles = normal.Normal(loc, scale).quantile(values)
    transformed_normal_quantiles = lambertw_transform.LambertWTail(
        shift=loc, scale=scale, tailweight=tailweight)(normal_quantiles)
    lambertw_quantiles = lwn.quantile(values)
    self.assertAllClose(transformed_normal_quantiles, lambertw_quantiles)

  @parameterized.named_parameters(
      ("0_0", 0.0),
      ("0_1", 0.1),
      ("1_5", 1.5))
  def testQuantileInverseOfCDF(self, delta):
    lwn = lambertw_f.LambertWNormal(loc=2., scale=3., tailweight=delta)
    samples_ = self.evaluate(lwn.sample(5, seed=test_util.test_seed()))
    cdf_vals = lwn.cdf(samples_)
    self.assertAllClose(lwn.quantile(cdf_vals), samples_, rtol=1e-4)

  @parameterized.named_parameters(
      ("0_0", 0., True),
      ("0_1", 0.1, True),
      ("0_4", 0.4, True),
      ("0_8", 0.8, True),
      ("1_5", 1.5, False))
  def testMeanMode(self, delta, mean_exists):
    loc = 2.
    scale = 3.
    lwn = lambertw_f.LambertWNormal(loc=loc, scale=scale, tailweight=delta)
    self.assertAllClose(lwn.mode(), loc)
    if mean_exists:
      self.assertAllClose(lwn.mean(), loc)
    else:
      self.assertAllClose(lwn.mean(), np.nan)

  @parameterized.named_parameters(
      ("0_0", 0., 1.0),
      ("0_1", 0.1, 1. / (1. - 2 * 0.1)**(3./2.)),
      ("0_4", 0.4, 1. / (1. - 2 * 0.4)**(3./2.)),
      ("0_8", 0.8, np.inf),
      ("1_5", 1.5, np.nan))
  def testVariance(self, delta, variance_multiplier):
    loc = 2.
    scale = 3.
    lwn = lambertw_f.LambertWNormal(loc=loc, scale=scale, tailweight=delta)
    if np.isnan(variance_multiplier):
      self.assertAllClose(lwn.variance(), np.nan)
    else:
      self.assertAllClose(lwn.variance(), scale**2 * variance_multiplier)

  @test_util.disable_test_for_backend(
      disable_numpy=True, disable_jax=True,
      reason="Keras/Layers are not supported in JAX or Numpy.")
  def testWorksInDistributionLayerAndNegloglik(self):
    """Test that distribution works as layer and in gradient optimization."""
    x = np.random.uniform(size=(100, 1))
    y = 2 + -1 * x
    lwn = lambertw_f.LambertWNormal(tailweight=0.5, loc=0.2, scale=1.0)
    eps = self.evaluate(lwn.sample(x.shape, seed=test_util.test_seed()))
    y += eps

    def dist_lambda(t):
      return lambertw_f.LambertWNormal(
          loc=t[..., :1],
          scale=1e-3 + tf.math.softplus(t[..., 1:2]),
          tailweight=1e-3 + tf.math.softplus(t[..., 2:]))

    from tensorflow_probability.python.layers import distribution_layer  # pylint:disable=g-import-not-at-top
    dist_layer = distribution_layer.DistributionLambda(dist_lambda)
    model = tf_keras.Sequential([
        tf_keras.layers.Dense(10, "relu"),
        tf_keras.layers.Dense(5, "selu"),
        tf_keras.layers.Dense(1 + 1 + 1),
        dist_layer])
    negloglik = lambda y, p_y: -p_y.log_prob(y)
    if tf.__internal__.tf2.enabled() and tf.executing_eagerly():
      optimizer = tf_keras.optimizers.Adam(learning_rate=0.01)
    else:
      optimizer = tf_keras.optimizers.legacy.Adam(learning_rate=0.01)

    model.compile(optimizer=optimizer, loss=negloglik)

    model.fit(x, y, epochs=1, verbose=True, batch_size=32, validation_split=0.2)
    self.assertGreater(model.history.history["val_loss"][0], -np.Inf)


if __name__ == "__main__":
  test_util.main()
