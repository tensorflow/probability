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
"""Tests for MixtureSameFamily distribution."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.internal import test_case
from tensorflow_probability.python.internal import test_util as tfp_test_util
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import

tfd = tfp.distributions


class _MixtureSameFamilyTest(tfp_test_util.VectorDistributionTestHelpers):

  def testSampleAndLogProbUnivariateShapes(self):
    gm = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(
            probs=self._build_tensor([0.3, 0.7])),
        components_distribution=tfd.Normal(
            loc=self._build_tensor([-1., 1]),
            scale=self._build_tensor([0.1, 0.5])))
    x = gm.sample([4, 5], seed=tfp_test_util.test_seed())
    log_prob_x = gm.log_prob(x)
    self.assertAllEqual([4, 5], self._shape(x))
    self.assertAllEqual([4, 5], self._shape(log_prob_x))

  def testSampleAndLogProbBatch(self):
    gm = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(
            probs=self._build_tensor([[0.3, 0.7]])),
        components_distribution=tfd.Normal(
            loc=self._build_tensor([[-1., 1]]),
            scale=self._build_tensor([[0.1, 0.5]])))
    x = gm.sample([4, 5], seed=tfp_test_util.test_seed())
    log_prob_x = gm.log_prob(x)
    self.assertAllEqual([4, 5, 1], self._shape(x))
    self.assertAllEqual([4, 5, 1], self._shape(log_prob_x))

  def testSampleAndLogProbShapesBroadcastMix(self):
    mix_probs = self._build_tensor([.3, .7])
    bern_probs = self._build_tensor([[.4, .6], [.25, .75]])
    bm = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(probs=mix_probs),
        components_distribution=tfd.Bernoulli(probs=bern_probs))
    x = bm.sample([4, 5], seed=tfp_test_util.test_seed())
    log_prob_x = bm.log_prob(x)
    x_ = self.evaluate(x)
    self.assertAllEqual([4, 5, 2], self._shape(x))
    self.assertAllEqual([4, 5, 2], self._shape(log_prob_x))
    self.assertAllEqual(
        np.ones_like(x_, dtype=np.bool), np.logical_or(x_ == 0., x_ == 1.))

  def testSampleAndLogProbMultivariateShapes(self):
    gm = self._build_mvndiag_mixture(
        probs=[0.3, 0.7],
        loc=[[-1., 1], [1, -1]],
        scale_identity_multiplier=[1., 0.5])
    x = gm.sample([4, 5], seed=tfp_test_util.test_seed())
    log_prob_x = gm.log_prob(x)
    self.assertAllEqual([4, 5, 2], self._shape(x))
    self.assertAllEqual([4, 5], self._shape(log_prob_x))

  def testSampleAndLogProbBatchMultivariateShapes(self):
    gm = self._build_mvndiag_mixture(
        probs=[0.3, 0.7],
        loc=[[[-1., 1], [1, -1]], [[0., 1], [1, 0]]],
        scale_identity_multiplier=[1., 0.5])
    x = gm.sample([4, 5], seed=tfp_test_util.test_seed())
    log_prob_x = gm.log_prob(x)
    self.assertAllEqual([4, 5, 2, 2], self._shape(x))
    self.assertAllEqual([4, 5, 2], self._shape(log_prob_x))

  def testSampleConsistentLogProb(self):
    gm = self._build_mvndiag_mixture(
        probs=[0.3, 0.7],
        loc=[[-1., 1], [1, -1]],
        scale_identity_multiplier=[1., 0.5])
    # Ball centered at component0's mean.
    self.run_test_sample_consistent_log_prob(
        self.evaluate, gm, radius=1., center=[-1., 1], rtol=0.02)
    # Larger ball centered at component1's mean.
    self.run_test_sample_consistent_log_prob(
        self.evaluate, gm, radius=1., center=[1., -1], rtol=0.02)

  def testLogCdf(self):
    gm = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(
            probs=self._build_tensor([0.3, 0.7])),
        components_distribution=tfd.Normal(
            loc=self._build_tensor([-1., 1]),
            scale=self._build_tensor([0.1, 0.5])))
    x = gm.sample(10, seed=tfp_test_util.test_seed())
    actual_log_cdf = gm.log_cdf(x)
    expected_log_cdf = tf.reduce_logsumexp(
        input_tensor=(gm.mixture_distribution.logits_parameter() +
                      gm.components_distribution.log_cdf(x[..., tf.newaxis])),
        axis=1)
    actual_log_cdf_, expected_log_cdf_ = self.evaluate(
        [actual_log_cdf, expected_log_cdf])
    self.assertAllClose(actual_log_cdf_, expected_log_cdf_, rtol=1e-6, atol=0.0)

  def testSampleConsistentMeanCovariance(self):
    gm = self._build_mvndiag_mixture(
        probs=[0.3, 0.7],
        loc=[[-1., 1], [1, -1]],
        scale_identity_multiplier=[1., 0.5])
    self.run_test_sample_consistent_mean_covariance(self.evaluate, gm)

  def testVarianceConsistentCovariance(self):
    gm = self._build_mvndiag_mixture(
        probs=[0.3, 0.7],
        loc=[[-1., 1], [1, -1]],
        scale_identity_multiplier=[1., 0.5])
    cov_, var_ = self.evaluate([gm.covariance(), gm.variance()])
    self.assertAllClose(cov_.diagonal(), var_, atol=0.)

  def testReparameterizationOfNonReparameterizedComponents(self):
    with self.assertRaises(ValueError):
      tfd.MixtureSameFamily(
          mixture_distribution=tfd.Categorical(
              logits=self._build_tensor([-0.3, 0.4])),
          components_distribution=tfd.Bernoulli(
              logits=self._build_tensor([0.1, -0.1])),
          reparameterize=True)

  def testSecondGradientIsDisabled(self):
    if not self.use_static_shape:
      return

    # Testing using GradientTape in both eager and graph modes.
    # GradientTape does not support some control flow ops in graph mode, which
    # is not a problem here as this code does not use any control flow.
    logits = self._build_tensor([[0.1, 0.5]])
    with tf.GradientTape() as g:
      g.watch(logits)
      with tf.GradientTape() as gg:
        gg.watch(logits)
        mixture = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(
                logits=logits),
            components_distribution=tfd.Normal(
                loc=self._build_tensor([[0.4, 0.25]]),
                scale=self._build_tensor([[0.1, 0.5]])),
            reparameterize=True)

        sample = mixture.sample()
      grad = gg.gradient(sample, logits)

    with self.assertRaises(LookupError):
      g.gradient(grad, logits)

  def _testMixtureReparameterizationGradients(
      self, mixture_func, parameters, function, num_samples):
    assert function in ["mean", "variance"]

    if not self.use_static_shape:
      return

    def sample_estimate(*parameters):
      mixture = mixture_func(*parameters)
      values = mixture.sample(num_samples, seed=tfp_test_util.test_seed())
      if function == "variance":
        values = tf.math.squared_difference(values, mixture.mean())
      return tf.reduce_mean(input_tensor=values, axis=0)

    def exact(*parameters):
      mixture = mixture_func(*parameters)
      # Normal mean does not depend on the scale, so add 0 * variance
      # to avoid None gradients. Also do the same for variance, just in case.
      if function == "variance":
        return mixture.variance() + 0 * mixture.mean()
      elif function == "mean":
        return mixture.mean() + 0 * mixture.variance()

    _, actual = tfp.math.value_and_gradient(sample_estimate, parameters)
    _, expected = tfp.math.value_and_gradient(exact, parameters)
    self.assertAllClose(actual, expected, atol=0.1, rtol=0.2)

  def testReparameterizationGradientsNormalScalarComponents(self):
    def mixture_func(logits, loc, scale):
      return tfd.MixtureSameFamily(
          mixture_distribution=tfd.Categorical(logits=logits),
          components_distribution=tfd.Normal(loc=loc, scale=scale),
          reparameterize=True)

    for function in ["mean", "variance"]:
      self._testMixtureReparameterizationGradients(
          mixture_func,
          [self._build_tensor([[0.1, 0.5]]),  # logits
           self._build_tensor([[0.4, 0.25]]),  # loc
           self._build_tensor([[0.1, 0.5]])],  # scale
          function,
          num_samples=10000)

  def testReparameterizationGradientsNormalVectorComponents(self):
    def mixture_func(logits, loc, scale):
      return tfd.MixtureSameFamily(
          mixture_distribution=tfd.Categorical(logits=logits),
          components_distribution=tfd.Independent(
              tfd.Normal(loc=loc, scale=scale), reinterpreted_batch_ndims=1),
          reparameterize=True)

    for function in ["mean", "variance"]:
      self._testMixtureReparameterizationGradients(
          mixture_func,
          [self._build_tensor([0.5, -0.2, 0.1]),  # logits
           self._build_tensor([[-1., 1], [0.5, -1], [-1., 0.5]]),  # mean
           self._build_tensor([[0.1, 0.5], [0.3, 0.5], [0.2, 0.3]])],  # scale
          function,
          num_samples=20000)

  def testReparameterizationGradientsNormalMatrixComponents(self):
    def mixture_func(logits, loc, scale):
      return tfd.MixtureSameFamily(
          mixture_distribution=tfd.Categorical(logits=logits),
          components_distribution=tfd.Independent(
              tfd.Normal(loc=loc, scale=scale), reinterpreted_batch_ndims=2),
          reparameterize=True)

    for function in ["mean", "variance"]:
      self._testMixtureReparameterizationGradients(
          mixture_func,
          [self._build_tensor([0.7, 0.2, 0.1]),  # logits
           self._build_tensor([[[-1., 1]], [[0.5, -1]], [[-1., 0.5]]]),  # mean
           # scale
           self._build_tensor([[[0.1, 0.5]], [[0.3, 0.5]], [[0.2, 0.3]]])],
          function,
          num_samples=50000)

  def testReparameterizationGradientsExponentialScalarComponents(self):
    def mixture_func(logits, rate):
      return tfd.MixtureSameFamily(
          mixture_distribution=tfd.Categorical(logits=logits),
          components_distribution=tfd.Exponential(rate=rate),
          reparameterize=True)

    for function in ["mean", "variance"]:
      self._testMixtureReparameterizationGradients(
          mixture_func,
          [self._build_tensor([0.7, 0.2, 0.1]),  # logits
           self._build_tensor([1., 0.5, 1.])],  # rate
          function,
          num_samples=10000)

  def testDeterministicSampling(self):
    seed = tfp_test_util.test_seed()
    tf.compat.v1.set_random_seed(seed)
    dist = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(logits=[0., 0.]),
        components_distribution=tfd.Normal(loc=[0., 200.], scale=[1., 1.]))
    sample_1 = self.evaluate(dist.sample([100], seed=seed))
    tf.compat.v1.set_random_seed(seed)
    sample_2 = self.evaluate(dist.sample([100], seed=seed))
    self.assertAllClose(sample_1, sample_2)

  def _shape(self, x):
    if self.use_static_shape:
      return tensorshape_util.as_list(x.shape)
    else:
      return self.evaluate(tf.shape(input=x))

  def _build_mvndiag_mixture(self, probs, loc, scale_identity_multiplier):
    components_distribution = tfd.MultivariateNormalDiag(
        loc=self._build_tensor(loc),
        scale_identity_multiplier=self._build_tensor(
            scale_identity_multiplier))

    # Use a no-op `Independent` wrapper to possibly create dynamic ndims.
    wrapped_components_distribution = tfd.Independent(
        components_distribution,
        reinterpreted_batch_ndims=self._build_tensor(0, dtype=np.int32))
    # Lambda ensures that the covariance fn sees `self=components_distribution`.
    wrapped_components_distribution._covariance = (
        lambda: components_distribution.covariance())  # pylint: disable=unnecessary-lambda

    gm = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(
            probs=self._build_tensor(probs)),
        components_distribution=wrapped_components_distribution)
    return gm

  def _build_tensor(self, ndarray, dtype=None):
    # Enforce parameterized dtype and static/dynamic testing.
    ndarray = np.asarray(ndarray).astype(
        dtype if dtype is not None else self.dtype)
    if self.use_static_shape:
      return tf.convert_to_tensor(value=ndarray)
    else:
      return tf.compat.v1.placeholder_with_default(input=ndarray, shape=None)


@test_util.run_all_in_graph_and_eager_modes
class MixtureSameFamilyTestStatic32(
    _MixtureSameFamilyTest,
    test_case.TestCase):
  use_static_shape = True
  dtype = np.float32


@test_util.run_all_in_graph_and_eager_modes
class MixtureSameFamilyTestDynamic32(
    _MixtureSameFamilyTest,
    test_case.TestCase):
  use_static_shape = False
  dtype = np.float32


@test_util.run_all_in_graph_and_eager_modes
class MixtureSameFamilyTestStatic64(
    _MixtureSameFamilyTest,
    test_case.TestCase):
  use_static_shape = True
  dtype = np.float64

if __name__ == "__main__":
  tf.test.main()
