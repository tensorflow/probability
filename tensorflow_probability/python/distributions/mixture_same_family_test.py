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
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import hypothesis_testlib as tfp_hps
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.internal import test_util

tfd = tfp.distributions


class _MixtureSameFamilyTest(test_util.VectorDistributionTestHelpers):

  def testSampleAndLogProbUnivariateShapes(self):
    gm = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(
            probs=self._build_tensor([0.3, 0.7])),
        components_distribution=tfd.Normal(
            loc=self._build_tensor([-1., 1]),
            scale=self._build_tensor([0.1, 0.5])),
        validate_args=True)
    x = gm.sample([4, 5], seed=test_util.test_seed())
    log_prob_x = gm.log_prob(x)
    self.assertAllEqual([4, 5], self._shape(x))
    self.assertAllEqual([4, 5], self._shape(log_prob_x))

  def testSampleAndLogProbBatch(self):
    gm = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(
            probs=self._build_tensor([[0.3, 0.7]])),
        components_distribution=tfd.Normal(
            loc=self._build_tensor([[-1., 1]]),
            scale=self._build_tensor([[0.1, 0.5]])),
        validate_args=True)
    x = gm.sample([4, 5], seed=test_util.test_seed())
    log_prob_x = gm.log_prob(x)
    self.assertAllEqual([4, 5, 1], self._shape(x))
    self.assertAllEqual([4, 5, 1], self._shape(log_prob_x))

  def testSampleAndLogProbShapesBroadcastMix(self):
    mix_probs = self._build_tensor([.3, .7])
    bern_probs = self._build_tensor([[.4, .6], [.25, .75]])
    bm = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(probs=mix_probs),
        components_distribution=tfd.Bernoulli(probs=bern_probs),
        validate_args=True)
    x = bm.sample([4, 5], seed=test_util.test_seed())
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
    x = gm.sample([4, 5], seed=test_util.test_seed())
    log_prob_x = gm.log_prob(x)
    self.assertAllEqual([4, 5, 2], self._shape(x))
    self.assertAllEqual([4, 5], self._shape(log_prob_x))

  def testSampleAndLogProbBatchMultivariateShapes(self):
    gm = self._build_mvndiag_mixture(
        probs=[0.3, 0.7],
        loc=[[[-1., 1], [1, -1]], [[0., 1], [1, 0]]],
        scale_identity_multiplier=[1., 0.5])
    x = gm.sample([4, 5], seed=test_util.test_seed())
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
            scale=self._build_tensor([0.1, 0.5])),
        validate_args=True)
    x = gm.sample(10, seed=test_util.test_seed())
    actual_log_cdf = gm.log_cdf(x)
    expected_log_cdf = tf.reduce_logsumexp(
        (gm.mixture_distribution.logits_parameter() +
         gm.components_distribution.log_cdf(x[..., tf.newaxis])),
        axis=1)
    actual_log_cdf_, expected_log_cdf_ = self.evaluate(
        [actual_log_cdf, expected_log_cdf])
    self.assertAllClose(actual_log_cdf_, expected_log_cdf_, rtol=1e-6, atol=0.0)

  def testCovarianceWithBatch(self):
    d = self._build_mvndiag_mixture(
        probs=[0.2, 0.3, 0.5],
        loc=np.zeros((2, 1, 5, 3, 4)),
        scale_identity_multiplier=[1., 0.75, 0.5])
    self.assertAllEqual((2, 1, 5, 4, 4), self.evaluate(d.covariance()).shape)

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
          reparameterize=True,
          validate_args=True)

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
            mixture_distribution=tfd.Categorical(logits=logits),
            components_distribution=tfd.Normal(
                loc=self._build_tensor([[0.4, 0.25]]),
                scale=self._build_tensor([[0.1, 0.5]])),
            reparameterize=True,
            validate_args=True)

        sample = mixture.sample(seed=test_util.test_seed())
      grad = gg.gradient(sample, logits)

    with self.assertRaises(LookupError):
      g.gradient(grad, logits)

  def _testMixtureReparameterizationGradients(
      self, mixture_func, parameters, function, num_samples):
    assert function in ['mean', 'variance']

    if not self.use_static_shape:
      return

    def sample_estimate(*parameters):
      mixture = mixture_func(*parameters)
      values = mixture.sample(num_samples, seed=test_util.test_seed())
      if function == 'variance':
        values = tf.math.squared_difference(values, mixture.mean())
      return tf.reduce_mean(values, axis=0)

    def exact(*parameters):
      mixture = mixture_func(*parameters)
      # Normal mean does not depend on the scale, so add 0 * variance
      # to avoid None gradients. Also do the same for variance, just in case.
      if function == 'variance':
        return mixture.variance() + 0 * mixture.mean()
      elif function == 'mean':
        return mixture.mean() + 0 * mixture.variance()

    _, actual = tfp.math.value_and_gradient(sample_estimate, parameters)
    _, expected = tfp.math.value_and_gradient(exact, parameters)
    self.assertAllClose(actual, expected, atol=0.1, rtol=0.2)

  def testReparameterizationGradientsNormalScalarComponents(self):
    def mixture_func(logits, loc, scale):
      return tfd.MixtureSameFamily(
          mixture_distribution=tfd.Categorical(logits=logits),
          components_distribution=tfd.Normal(loc=loc, scale=scale),
          reparameterize=True,
          validate_args=True)

    for function in ['mean', 'variance']:
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
          reparameterize=True,
          validate_args=True)

    for function in ['mean', 'variance']:
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
          reparameterize=True,
          validate_args=True)

    for function in ['mean', 'variance']:
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
          reparameterize=True,
          validate_args=True)

    for function in ['mean', 'variance']:
      self._testMixtureReparameterizationGradients(
          mixture_func,
          [self._build_tensor([0.7, 0.2, 0.1]),  # logits
           self._build_tensor([1., 0.5, 1.])],  # rate
          function,
          num_samples=10000)

  def testDeterministicSampling(self):
    seed = test_util.test_seed()
    dist = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(logits=[0., 0.]),
        components_distribution=tfd.Normal(loc=[0., 200.], scale=[1., 1.]),
        validate_args=True)
    tf.random.set_seed(seed)
    sample_1 = self.evaluate(dist.sample([100], seed=seed))
    tf.random.set_seed(seed)
    sample_2 = self.evaluate(dist.sample([100], seed=seed))
    self.assertAllClose(sample_1, sample_2)

  @test_util.tf_tape_safety_test
  def testGradientsThroughParams(self):
    logits = self._build_variable([1., 2., 3.])
    loc = self._build_variable([0., 0., 0])
    scale = self._build_variable(1.)
    dist = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(logits=logits),
        components_distribution=tfd.Logistic(loc=loc, scale=scale),
        validate_args=True)
    with tf.GradientTape() as tape:
      loss = -dist.log_prob([5., 4.])
    grad = tape.gradient(loss, dist.trainable_variables)
    self.assertLen(grad, 3)
    self.assertAllNotNone(grad)

    logits = self._build_variable(np.zeros((4, 4, 5)))
    loc = self._build_variable(np.zeros((4, 4, 5, 2, 3)))
    scale = self._build_variable(1.)
    dist = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(logits=logits),
        components_distribution=tfd.Independent(
            tfd.Logistic(loc=loc, scale=scale),
            reinterpreted_batch_ndims=self._build_tensor(2, dtype=np.int32)),
        validate_args=True)
    with tf.GradientTape() as tape:
      loss = -dist.log_prob(np.zeros((4, 4, 2, 3)))
    grad = tape.gradient(loss, dist.trainable_variables)
    self.assertLen(grad, 3)
    self.assertAllNotNone(grad)

  def testExcessiveConcretizationOfParams(self):
    logits = tfp_hps.defer_and_count_usage(
        self._build_variable(np.zeros((4, 4, 5)), name='logits'))
    concentration = tfp_hps.defer_and_count_usage(
        self._build_variable(np.zeros((4, 4, 5, 3)), name='concentration'))
    dist = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(logits=logits),
        components_distribution=tfd.Dirichlet(concentration=concentration),
        validate_args=True)

    # Many methods use mixture_distribution and components_distribution at most
    # once, and thus incur no extra reads/concretizations of parameters.

    for method in ('batch_shape_tensor', 'event_shape_tensor',
                   'mean'):
      with tfp_hps.assert_no_excessive_var_usage(method, max_permissible=2):
        getattr(dist, method)()

    with tfp_hps.assert_no_excessive_var_usage('sample', max_permissible=2):
      dist.sample(seed=test_util.test_seed())

    for method in ('log_prob', 'prob'):
      with tfp_hps.assert_no_excessive_var_usage(method, max_permissible=2):
        getattr(dist, method)(np.ones((4, 4, 3)) / 3.)

    # TODO(b/140579567): The `variance()` and `covariance()` methods require
    # calling both:
    #  - `self.components_distribution.mean()`
    #  - `self.components_distribution.variance()` or `.covariance()`
    # Thus, these methods incur an additional concretization (or two if
    # `validate_args=True` for `self.components_distribution`).

    for method in ('variance', 'covariance'):
      with tfp_hps.assert_no_excessive_var_usage(method, max_permissible=3):
        getattr(dist, method)()

    # TODO(b/140579567): When event ndims is not known statically, several
    # methods call `self.components_distribution.event_shape_tensor()` to
    # determine the number of event dimensions.  Depending on the underlying
    # distribution, this would likely incur additional concretizations of the
    # parameters of `self.components_distribution`.  The methods are:
    #  - `log_cdf` and `cdf`
    #  - `log_prob` and `prob`
    #  - `mean` and `variance`
    #  - `sample`
    #
    # NOTE: `Distribution.survival_function` and `log_survival_function` will
    # call `Distribution.cdf` and `Distribution.log_cdf`, resulting in one
    # additional call to `_parameter_control_dependencies`, and thus an
    # additional concretizations of the underlying distribution parameters.

  def testExcessiveConcretizationOfParamsWithReparameterization(self):
    logits = tfp_hps.defer_and_count_usage(self._build_variable(
        np.zeros(5), name='logits', static_rank=True))
    loc = tfp_hps.defer_and_count_usage(self._build_variable(
        np.zeros((4, 4, 5)), name='loc', static_rank=True))
    scale = tfp_hps.defer_and_count_usage(self._build_variable(
        1., name='scale', static_rank=True))
    dist = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(logits=logits),
        components_distribution=tfd.Logistic(loc=loc, scale=scale),
        reparameterize=True, validate_args=True)

    # TODO(b/140579567): With reparameterization, there are additional reads of
    # the parameters of the underlying mixture and components distributions when
    # sampling, from calls in `_distributional_transform` to:
    #
    #  - `self.mixture_distribution.logits_parameter`
    #  - `self.components_distribution.log_prob`
    #  - `self.components_distribution.cdf`
    #
    # NOTE: In the unlikely case that samples have a statically-known rank but
    # the rank of `self.components_distribution.event_shape` is not known
    # statically, there can be additional reads in `_distributional_transform`
    # from calling `self.components_distribution.is_scalar_event`.

    with tfp_hps.assert_no_excessive_var_usage('sample', max_permissible=4):
      dist.sample(seed=test_util.test_seed())

  @test_util.tf_tape_safety_test
  def testSampleGradientsThroughParams(self):
    logits = self._build_variable(np.zeros(5), static_rank=True)
    loc = self._build_variable(np.zeros((4, 5, 2, 3)), static_rank=True)
    scale = self._build_variable(1., static_rank=True)
    dist = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(logits=logits),
        components_distribution=tfd.Independent(
            tfd.Logistic(loc=loc, scale=scale), reinterpreted_batch_ndims=2),
        reparameterize=True, validate_args=True)
    with tf.GradientTape() as tape:
      loss = tf.reduce_sum(dist.sample(2, seed=test_util.test_seed()))
    grad = tape.gradient(loss, dist.trainable_variables)
    self.assertLen(grad, 3)
    self.assertAllNotNone(grad)

  def _shape(self, x):
    if self.use_static_shape:
      return tensorshape_util.as_list(x.shape)
    else:
      return self.evaluate(tf.shape(x))

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
        components_distribution=wrapped_components_distribution,
        validate_args=True)
    return gm

  def _build_tensor(self, ndarray, dtype=None):
    # Enforce parameterized dtype and static/dynamic testing.
    if dtype is None:
      dtype = self.dtype
    ndarray = np.array(ndarray, dtype=dtype)
    if self.use_static_shape:
      return tf.convert_to_tensor(ndarray)
    else:
      return tf1.placeholder_with_default(ndarray, shape=None)

  def _build_variable(self, ndarray, name=None, dtype=None, static_rank=False):
    if dtype is None:
      dtype = self.dtype
    ndarray = np.array(ndarray, dtype=dtype)
    if self.use_static_shape:
      return tf.Variable(ndarray, name=name, dtype=dtype)
    elif static_rank:
      return tf.Variable(ndarray, name=name, dtype=dtype,
                         shape=tf.TensorShape([None] * len(ndarray.shape)))
    else:
      return tf.Variable(ndarray, name=name, dtype=dtype,
                         shape=tf.TensorShape(None))


@test_util.test_all_tf_execution_regimes
class MixtureSameFamilyTestStatic32(
    _MixtureSameFamilyTest,
    test_util.TestCase):
  use_static_shape = True
  dtype = np.float32


@test_util.test_all_tf_execution_regimes
class MixtureSameFamilyTestDynamic32(
    _MixtureSameFamilyTest,
    test_util.TestCase):
  use_static_shape = False
  dtype = np.float32

  def testMatchingComponentsSizeAssertions(self):
    logits = self._build_variable(np.zeros(5))
    loc = self._build_variable(np.zeros((4, 5, 2, 3)), static_rank=True)
    scale = self._build_variable(1.)
    dist = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(logits=logits),
        components_distribution=tfd.Independent(
            tfd.Logistic(loc=loc, scale=scale), reinterpreted_batch_ndims=2),
        validate_args=True)

    self.evaluate([v.initializer for v in [logits, loc, scale]])
    self.evaluate(dist.mean())

    msg = ('`mixture_distribution` components.* does not equal '
           r'`components_distribution.batch_shape\[-1\]`')
    with self.assertRaisesRegex(Exception, msg):
      with tf.control_dependencies([loc.assign(np.zeros((4, 7, 2, 3)))]):
        self.evaluate(dist.mean())

  def testMatchingBatchShapeAssertions(self):
    logits = self._build_variable(np.zeros(5))
    loc = self._build_variable(np.zeros((4, 5, 2, 3)), static_rank=True)
    scale = self._build_variable(1.)
    dist = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(logits=logits),
        components_distribution=tfd.Independent(
            tfd.Logistic(loc=loc, scale=scale), reinterpreted_batch_ndims=2),
        validate_args=True)

    self.evaluate([v.initializer for v in [logits, loc, scale]])
    self.evaluate(dist.sample(seed=test_util.test_seed()))

    msg = ('`mixture_distribution.batch_shape`.* is not compatible with '
           '`components_distribution.batch_shape')
    with self.assertRaisesRegex(Exception, msg):
      with tf.control_dependencies([logits.assign(np.zeros((4, 3, 5)))]):
        self.evaluate(dist.sample(seed=test_util.test_seed()))


@test_util.test_all_tf_execution_regimes
class MixtureSameFamilyTestStatic64(
    _MixtureSameFamilyTest,
    test_util.TestCase):
  use_static_shape = True
  dtype = np.float64

if __name__ == '__main__':
  tf.test.main()
