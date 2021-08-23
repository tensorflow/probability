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
"""Tests for the Independent distribution."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os

# Dependency imports

from absl.testing import parameterized
import numpy as np
from scipy import stats as sp_stats
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.internal import hypothesis_testlib as tfp_hps
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.internal import test_util

tfd = tfp.distributions

JAX_MODE = False


@test_util.test_all_tf_execution_regimes
class IndependentDistributionTest(test_util.TestCase):

  def assertRaises(self, error_class, msg):
    if tf.executing_eagerly():
      return self.assertRaisesRegexp(error_class, msg)
    return self.assertRaisesOpError(msg)

  def testSampleAndLogProbUnivariate(self):
    loc = np.float32([-1., 1])
    scale = np.float32([0.1, 0.5])
    ind = tfd.Independent(
        distribution=tfd.Normal(loc=loc, scale=scale),
        reinterpreted_batch_ndims=1,
        validate_args=True)

    x = ind.sample([4, 5], seed=test_util.test_seed(hardcoded_seed=42))
    log_prob_x = ind.log_prob(x)
    x_, actual_log_prob_x = self.evaluate([x, log_prob_x])

    self.assertEqual([], ind.batch_shape)
    self.assertEqual([2], ind.event_shape)
    self.assertEqual([4, 5, 2], x.shape)
    self.assertEqual([4, 5], log_prob_x.shape)

    expected_log_prob_x = sp_stats.norm(loc, scale).logpdf(x_).sum(-1)
    self.assertAllClose(
        expected_log_prob_x, actual_log_prob_x, rtol=1e-5, atol=1e-6)

  def testSampleAndLogProbMultivariate(self):
    loc = np.float32([[-1., 1], [1, -1]])
    scale = np.float32([1., 0.5])
    ind = tfd.Independent(
        distribution=tfd.MultivariateNormalDiag(
            loc=loc, scale_identity_multiplier=scale),
        reinterpreted_batch_ndims=1,
        validate_args=True)

    x = ind.sample([4, 5], seed=test_util.test_seed())
    log_prob_x = ind.log_prob(x)
    x_, actual_log_prob_x = self.evaluate([x, log_prob_x])

    self.assertEqual([], ind.batch_shape)
    self.assertEqual([2, 2], ind.event_shape)
    self.assertEqual([4, 5, 2, 2], x.shape)
    self.assertEqual([4, 5], log_prob_x.shape)

    expected_log_prob_x = sp_stats.norm(
        loc, scale[:, None]).logpdf(x_).sum(-1).sum(-1)
    self.assertAllClose(
        expected_log_prob_x, actual_log_prob_x, rtol=1e-6, atol=0.)

  def testCdfMultivariate(self):
    ind = tfd.Independent(
        distribution=tfd.Normal(loc=tf.zeros([3]), scale=1.),
        reinterpreted_batch_ndims=1,
        validate_args=True)

    cdfs = ind.cdf([[-50., 0., 0.], [0., 0., 0.], [50., 0., 0.], [50., 0., 50.],
                    [50., 50., 50.]])
    log_cdfs = ind.log_cdf([[0., 0., 0.], [50., 0., 0.], [50., 0., 50.],
                            [50., 50., 50.]])
    cdfs_, log_cdfs_ = self.evaluate([cdfs, log_cdfs])
    self.assertAllClose([0, .5**3, .5**2, .5, 1.], cdfs_)
    self.assertAllClose([np.log(.5) * 3, np.log(.5) * 2, np.log(.5), 0.],
                        log_cdfs_)

  def testSampleConsistentStats(self):
    loc = np.float32([[-1., 1], [1, -1]])
    scale = np.float32([1., 0.5])
    n_samp = 1e4
    ind = tfd.Independent(
        distribution=tfd.MultivariateNormalDiag(
            loc=loc, scale_identity_multiplier=scale),
        reinterpreted_batch_ndims=1,
        validate_args=True)

    x = ind.sample(int(n_samp), seed=test_util.test_seed(hardcoded_seed=42))
    sample_mean = tf.reduce_mean(x, axis=0)
    sample_var = tf.reduce_mean(
        tf.math.squared_difference(x, sample_mean), axis=0)
    sample_std = tf.sqrt(sample_var)
    sample_entropy = -tf.reduce_mean(ind.log_prob(x), axis=0)

    [
        sample_mean_,
        sample_var_,
        sample_std_,
        sample_entropy_,
        actual_mean_,
        actual_var_,
        actual_std_,
        actual_entropy_,
        actual_mode_,
    ] = self.evaluate([
        sample_mean,
        sample_var,
        sample_std,
        sample_entropy,
        ind.mean(),
        ind.variance(),
        ind.stddev(),
        ind.entropy(),
        ind.mode(),
    ])

    # Bounds chosen so that the probability of each sample mean/variance/stddev
    # differing by more than the given tolerance is roughly 1e-6.
    self.assertAllClose(sample_mean_, actual_mean_, rtol=0.049, atol=0.)
    self.assertAllClose(sample_var_, actual_var_, rtol=0.07, atol=0.)
    self.assertAllClose(sample_std_, actual_std_, rtol=0.035, atol=0.)

    self.assertAllClose(sample_entropy_, actual_entropy_, rtol=0.015, atol=0.)
    self.assertAllClose(loc, actual_mode_, rtol=1e-6, atol=0.)

  def testEventNdimsIsStaticWhenPossible(self):
    ind = tfd.Independent(
        distribution=tfd.Normal(
            loc=tf1.placeholder_with_default([2.], shape=None),
            scale=tf1.placeholder_with_default(1., shape=None)),
        reinterpreted_batch_ndims=1,
        validate_args=True)
    # Even though `event_shape` is not static, event_ndims must equal
    # `reinterpreted_batch_ndims + rank(distribution.event_shape)`.
    self.assertEqual(tensorshape_util.rank(ind.event_shape), 1)

  def testKLRaises(self):
    ind1 = tfd.Independent(
        distribution=tfd.Normal(
            loc=np.float32([-1., 1]), scale=np.float32([0.1, 0.5])),
        reinterpreted_batch_ndims=1,
        validate_args=True)
    ind2 = tfd.Independent(
        distribution=tfd.Normal(
            loc=np.float32(-1), scale=np.float32(0.5)),
        reinterpreted_batch_ndims=0, validate_args=True)

    with self.assertRaisesRegexp(
        ValueError, 'Event shapes do not match'):
      tfd.kl_divergence(ind1, ind2)

    ind1 = tfd.Independent(
        distribution=tfd.Normal(
            loc=np.float32([-1., 1]), scale=np.float32([0.1, 0.5])),
        reinterpreted_batch_ndims=1, validate_args=True)
    ind2 = tfd.Independent(
        distribution=tfd.MultivariateNormalDiag(
            loc=np.float32([-1., 1]), scale_diag=np.float32([0.1, 0.5])),
        reinterpreted_batch_ndims=0, validate_args=True)

    with self.assertRaisesRegexp(
        NotImplementedError, 'different event shapes'):
      tfd.kl_divergence(ind1, ind2)

  def testKlWithDynamicShapes(self):
    dist1 = tfd.Independent(
        tfd.Normal(loc=np.zeros((4, 5, 2, 3)), scale=1., validate_args=True),
        reinterpreted_batch_ndims=2, validate_args=True)

    loc2 = tf.Variable(np.zeros((4, 5, 2, 3)), shape=tf.TensorShape(None))
    scale2 = tf.Variable(np.ones([]), shape=tf.TensorShape(None))
    ndims2 = tf.Variable(2, trainable=False, shape=tf.TensorShape(None))
    dist2 = tfd.Independent(
        tfd.Normal(loc=loc2, scale=scale2, validate_args=True),
        reinterpreted_batch_ndims=ndims2, validate_args=True)

    self.evaluate([v.initializer for v in dist1.variables]
                  + [v.initializer for v in dist2.variables])
    kl = self.evaluate(dist1.kl_divergence(dist2))
    self.assertAllEqual((4, 5), kl.shape)

    with tf.control_dependencies([loc2.assign(np.zeros((4, 5, 3, 2)))]):
      with self.assertRaisesRegexp(Exception, 'Event shapes do not match'):
        self.evaluate(dist1.kl_divergence(dist2))

  def testKLScalarToMultivariate(self):
    normal1 = tfd.Normal(
        loc=np.float32([-1., 1]), scale=np.float32([0.1, 0.5]))
    ind1 = tfd.Independent(distribution=normal1, reinterpreted_batch_ndims=1,
                           validate_args=True)

    normal2 = tfd.Normal(
        loc=np.float32([-3., 3]), scale=np.float32([0.3, 0.3]))
    ind2 = tfd.Independent(distribution=normal2, reinterpreted_batch_ndims=1,
                           validate_args=True)

    normal_kl = tfd.kl_divergence(normal1, normal2)
    ind_kl = tfd.kl_divergence(ind1, ind2)
    self.assertAllClose(
        self.evaluate(tf.reduce_sum(normal_kl, axis=-1)),
        self.evaluate(ind_kl))

  def testKLIdentity(self):
    normal1 = tfd.Normal(
        loc=np.float32([-1., 1]), scale=np.float32([0.1, 0.5]))
    # This is functionally just a wrapper around normal1,
    # and doesn't change any outputs.
    ind1 = tfd.Independent(distribution=normal1, reinterpreted_batch_ndims=0,
                           validate_args=True)

    normal2 = tfd.Normal(
        loc=np.float32([-3., 3]), scale=np.float32([0.3, 0.3]))
    # This is functionally just a wrapper around normal2,
    # and doesn't change any outputs.
    ind2 = tfd.Independent(distribution=normal2, reinterpreted_batch_ndims=0,
                           validate_args=True)

    normal_kl = tfd.kl_divergence(normal1, normal2)
    ind_kl = tfd.kl_divergence(ind1, ind2)
    self.assertAllClose(
        self.evaluate(normal_kl), self.evaluate(ind_kl))

  def testKLMultivariateToMultivariate(self):
    # (1, 1, 2) batch of MVNDiag
    mvn1 = tfd.MultivariateNormalDiag(
        loc=np.float32([[[[-1., 1, 3.], [2., 4., 3.]]]]),
        scale_diag=np.float32([[[0.2, 0.1, 5.], [2., 3., 4.]]]))
    ind1 = tfd.Independent(distribution=mvn1, reinterpreted_batch_ndims=2,
                           validate_args=True)

    # (1, 1, 2) batch of MVNDiag
    mvn2 = tfd.MultivariateNormalDiag(
        loc=np.float32([[[[-2., 3, 2.], [1., 3., 2.]]]]),
        scale_diag=np.float32([[[0.1, 0.5, 3.], [1., 2., 1.]]]))

    ind2 = tfd.Independent(distribution=mvn2, reinterpreted_batch_ndims=2,
                           validate_args=True)

    mvn_kl = tfd.kl_divergence(mvn1, mvn2)
    ind_kl = tfd.kl_divergence(ind1, ind2)
    self.assertAllClose(
        self.evaluate(tf.reduce_sum(mvn_kl, axis=[-1, -2])),
        self.evaluate(ind_kl))

  def _testMnistLike(self, static_shape):
    sample_shape = [4, 5]
    batch_shape = [10]
    image_shape = [28, 28, 1]
    logits = 3 * np.random.random_sample(
        batch_shape + image_shape).astype(np.float32) - 1

    def expected_log_prob(x, logits):
      return (x * logits - np.log1p(np.exp(logits))).sum(-1).sum(-1).sum(-1)

    logits_ph = tf1.placeholder_with_default(
        logits, shape=logits.shape if static_shape else None)
    ind = tfd.Independent(
        distribution=tfd.Bernoulli(logits=logits_ph), validate_args=True)
    x = ind.sample(sample_shape, seed=test_util.test_seed())
    log_prob_x = ind.log_prob(x)
    [
        x_,
        actual_log_prob_x,
        ind_batch_shape,
        ind_event_shape,
        x_shape,
        log_prob_x_shape,
    ] = self.evaluate([
        x,
        log_prob_x,
        ind.batch_shape_tensor(),
        ind.event_shape_tensor(),
        tf.shape(x),
        tf.shape(log_prob_x),
    ])

    if static_shape:
      ind_batch_shape = ind.batch_shape
      ind_event_shape = ind.event_shape
      x_shape = x.shape
      log_prob_x_shape = log_prob_x.shape

    self.assertAllEqual(batch_shape, ind_batch_shape)
    self.assertAllEqual(image_shape, ind_event_shape)
    self.assertAllEqual(sample_shape + batch_shape + image_shape, x_shape)
    self.assertAllEqual(sample_shape + batch_shape, log_prob_x_shape)
    self.assertAllClose(
        expected_log_prob(x_, logits), actual_log_prob_x, rtol=1.5e-6, atol=0.)

  def testMnistLikeStaticShape(self):
    self._testMnistLike(static_shape=True)

  def testMnistLikeDynamicShape(self):
    self._testMnistLike(static_shape=False)

  def testSlicingScalarDistZeroReinterpretedDims(self):
    """Verifies a failure scenario identified by hypothesis testing.

    Calling self.copy(distribution=sliced_underlying) without explicitly
    specifying reinterpreted_batch_ndims allowed the default fallback logic of
    rank(underlying.batch_shape)-1 to take over, which we don't want in the
    slice case.
    """
    d = tfd.Independent(tfd.Bernoulli(logits=0), validate_args=True)
    self.assertAllEqual([], d[...].batch_shape)
    self.assertAllEqual([], d[...].event_shape)
    self.assertAllEqual([1], d[tf.newaxis].batch_shape)
    self.assertAllEqual([], d[tf.newaxis].event_shape)
    self.assertAllEqual([1], d[..., tf.newaxis].batch_shape)
    self.assertAllEqual([], d[..., tf.newaxis].event_shape)
    self.assertAllEqual([1, 1], d[tf.newaxis, ..., tf.newaxis].batch_shape)
    self.assertAllEqual([], d[tf.newaxis, ..., tf.newaxis].event_shape)

  def testSlicingGeneral(self):
    d = tfd.Independent(tfd.Bernoulli(logits=tf.zeros([5, 6])),
                        validate_args=True)
    self.assertAllEqual([5], d.batch_shape)
    self.assertAllEqual([6], d.event_shape)
    self.assertAllEqual([1, 5], d[tf.newaxis].batch_shape)
    self.assertAllEqual([6], d[tf.newaxis].event_shape)

    d = tfd.Independent(tfd.Bernoulli(logits=tf.zeros([4, 5, 6])),
                        validate_args=True)
    self.assertAllEqual([4], d.batch_shape)
    self.assertAllEqual([5, 6], d.event_shape)
    self.assertAllEqual([1, 3], d[tf.newaxis, ..., :3].batch_shape)
    self.assertAllEqual([5, 6], d[tf.newaxis, ..., :3].event_shape)

    d = tfd.Independent(tfd.Bernoulli(logits=tf.zeros([4, 5, 6])),
                        reinterpreted_batch_ndims=1, validate_args=True)
    self.assertAllEqual([4, 5], d.batch_shape)
    self.assertAllEqual([6], d.event_shape)
    self.assertAllEqual([1, 4, 3], d[tf.newaxis, ..., :3].batch_shape)
    self.assertAllEqual([6], d[tf.newaxis, ..., :3].event_shape)

  def testSlicingSubclasses(self):
    self.skipTest('b/183457779')

    class IndepBern1d(tfd.Independent):

      def __init__(self, logits):
        super(IndepBern1d, self).__init__(tfd.Bernoulli(logits=logits,
                                                        dtype=tf.float32),
                                          reinterpreted_batch_ndims=1)
        self._parameters = {'logits': logits}

    d = IndepBern1d(tf.zeros([4, 5, 6]))
    with self.assertRaisesRegexp(NotImplementedError,
                                 'does not support batch slicing'):
      d[:3]  # pylint: disable=pointless-statement

    class IndepBern1dSliceable(IndepBern1d):

      def _params_event_ndims(self):
        return dict(logits=1)

    d_sliceable = IndepBern1dSliceable(tf.zeros([4, 5, 6]))
    self.assertAllEqual([3, 5], d_sliceable[:3].batch_shape)
    self.assertAllEqual([6], d_sliceable[:3].event_shape)

  @test_util.tf_tape_safety_test
  def testGradientsThroughParams(self):
    loc = tf.Variable(np.zeros((4, 5, 2, 3)), shape=tf.TensorShape(None))
    scale = tf.Variable(np.ones([]), shape=tf.TensorShape(None))
    ndims = tf.Variable(2, trainable=False, shape=tf.TensorShape(None))
    dist = tfd.Independent(
        tfd.Logistic(loc=loc, scale=scale),
        reinterpreted_batch_ndims=ndims, validate_args=True)
    with tf.GradientTape() as tape:
      loss = -dist.log_prob(np.zeros((4, 5, 2, 3)))
    grad = tape.gradient(loss, dist.trainable_variables)
    self.assertLen(grad, 2)
    self.assertAllNotNone(grad)

  def testExcessiveConcretizationOfParams(self):
    loc = tfp_hps.defer_and_count_usage(
        tf.Variable(np.zeros((4, 2, 2)), shape=tf.TensorShape(None)))
    scale = tfp_hps.defer_and_count_usage(
        tf.Variable(np.ones([]), shape=tf.TensorShape(None)))
    ndims = tf.Variable(1, trainable=False, shape=tf.TensorShape(None))
    dist = tfd.Independent(
        tfd.Logistic(loc=loc, scale=scale, validate_args=True),
        reinterpreted_batch_ndims=ndims, validate_args=True)

    # TODO(b/140579567): All methods of `dist` may require four concretizations
    # of parameters `loc` and `scale`:
    #  - `Independent._parameter_control_dependencies` calls
    #    `Logistic.batch_shape_tensor`, which:
    #    * Reads `loc`, `scale` in `Logistic._parameter_control_dependencies`.
    #    * Reads `loc`, `scale` in `Logistic._batch_shape_tensor`.
    #  - The method `dist.m` will call `dist.self.m`, which:
    #    * Reads `loc`, `scale` in `Logistic._parameter_control_dependencies`.
    #    * Reads `loc`, `scale` in the implementation of method  `Logistic._m`.
    #
    # NOTE: If `dist.distribution` had dynamic batch shape and event shape,
    # there could be two more reads of the parameters of `dist.distribution`
    # in `dist.event_shape_tensor`, from calling
    # `dist.distribution.event_shape_tensor()`.

    for method in ('batch_shape_tensor', 'event_shape_tensor',
                   'mode', 'stddev', 'entropy'):
      with tfp_hps.assert_no_excessive_var_usage(method, max_permissible=4):
        getattr(dist, method)()

    with tfp_hps.assert_no_excessive_var_usage('sample', max_permissible=4):
      dist.sample(seed=test_util.test_seed())

    for method in ('log_prob', 'log_cdf', 'prob', 'cdf'):
      with tfp_hps.assert_no_excessive_var_usage(method, max_permissible=4):
        getattr(dist, method)(np.zeros((3, 4, 2, 2)))

    # `Distribution.survival_function` and `Distribution.log_survival_function`
    # will call `Distribution.cdf` and `Distribution.log_cdf`, resulting in
    # one additional call to `Independent._parameter_control_dependencies`,
    # and thus two additional concretizations of the parameters.

    for method in ('survival_function', 'log_survival_function'):
      with tfp_hps.assert_no_excessive_var_usage(method, max_permissible=6):
        getattr(dist, method)(np.zeros((3, 4, 2, 2)))

  def testExcessiveConcretizationWithDefaultReinterpretedBatchNdims(self):
    loc = tfp_hps.defer_and_count_usage(
        tf.Variable(np.zeros((5, 2, 3)), shape=tf.TensorShape(None)))
    scale = tfp_hps.defer_and_count_usage(
        tf.Variable(np.ones([]), shape=tf.TensorShape(None)))
    dist = tfd.Independent(
        tfd.Logistic(loc=loc, scale=scale, validate_args=True),
        reinterpreted_batch_ndims=None, validate_args=True)

    for method in ('event_shape_tensor', 'mean', 'variance'):
      with tfp_hps.assert_no_excessive_var_usage(method, max_permissible=4):
        getattr(dist, method)()

    with tfp_hps.assert_no_excessive_var_usage('batch_shape_tensor',
                                               max_permissible=10):
      # Automatic inference of batch shape requires additional concretizations.
      dist.batch_shape_tensor()

    with tfp_hps.assert_no_excessive_var_usage('sample', max_permissible=6):
      dist.sample(seed=test_util.test_seed())

    # In addition to the four reads of `loc`, `scale` described above in
    # `testExcessiveConcretizationOfParams`, the methods below have two more
    # reads of these parameters -- from computing a default value for
    # `reinterpreted_batch_ndims`, which requires calling
    # `dist.distribution.batch_shape_tensor()`.

    for method in ('log_prob', 'log_cdf', 'prob', 'cdf'):
      with tfp_hps.assert_no_excessive_var_usage(method, max_permissible=6):
        getattr(dist, method)(np.zeros((4, 5, 2, 3)))

    with tfp_hps.assert_no_excessive_var_usage('entropy', max_permissible=6):
      dist.entropy()

    # `Distribution.survival_function` and `Distribution.log_survival_function`
    # will call `Distribution.cdf` and `Distribution.log_cdf`, resulting in
    # one additional call to `Independent._parameter_control_dependencies`,
    # and thus two additional concretizations of the parameters.

    for method in ('survival_function', 'log_survival_function'):
      with tfp_hps.assert_no_excessive_var_usage(method, max_permissible=8):
        getattr(dist, method)(np.zeros((4, 5, 2, 3)))

  @test_util.disable_test_for_backend(
      disable_numpy=True, disable_jax=True,
      reason='NumpyVariable does not correctly handle unknown shapes. '
      'And shape sizes are known statically in JAX.')
  def testChangingVariableShapes(self):
    if not tf.executing_eagerly():
      return

    loc = tf.Variable(np.zeros((4, 5, 2, 3)), shape=tf.TensorShape(None))
    scale = tf.Variable(np.ones([]), shape=tf.TensorShape(None))
    dist = tfd.Independent(
        tfd.Logistic(loc=loc, scale=scale),
        reinterpreted_batch_ndims=None, validate_args=True)

    self.assertAllEqual((4,), dist.batch_shape_tensor())

    loc.assign(np.zeros((3, 7, 1, 1, 1)))
    self.assertAllEqual((3,), dist.batch_shape_tensor())
    self.assertAllEqual(
        (2, 3), tf.shape(dist.log_prob(np.zeros((2, 3, 7, 1, 1, 1)))))

  @parameterized.named_parameters(dict(testcase_name=''),
                                  dict(testcase_name='_jit', jit=True))
  def test_kahan_precision(self, jit=False):
    maybe_jit = lambda f: f
    if jit:
      self.skip_if_no_xla()
      maybe_jit = tf.function(jit_compile=True)
    stream = test_util.test_seed_stream()
    n = 20_000
    samps = tfd.Poisson(rate=1.).sample(n, seed=stream())
    log_rate = tf.fill([n], tfd.Normal(0, .2).sample(seed=stream()))
    pois = tfd.Poisson(log_rate=log_rate)
    lp_fn = maybe_jit(tfd.Independent(pois, reinterpreted_batch_ndims=1,
                                      experimental_use_kahan_sum=True).log_prob)
    lp = lp_fn(samps)
    pois64 = tfd.Poisson(log_rate=tf.cast(log_rate, tf.float64))
    lp64 = tfd.Independent(pois64, reinterpreted_batch_ndims=1).log_prob(
        tf.cast(samps, tf.float64))
    # Evaluate together to ensure we use the same samples.
    lp, lp64 = self.evaluate((tf.cast(lp, tf.float64), lp64))
    # Fails ~75% CPU, 1-75% GPU --vary_seed runs w/o experimental_use_kahan_sum.
    self.assertAllClose(lp64, lp, rtol=0., atol=.01)

  @parameterized.named_parameters(dict(testcase_name=''),
                                  dict(testcase_name='_jit', jit=True))
  def test_kahan_precision_bijector(self, jit=False):
    maybe_jit = lambda f: f
    if jit:
      self.skip_if_no_xla()
      maybe_jit = tf.function(jit_compile=True)

    def ldj_fn(x, dist):
      bij = dist.experimental_default_event_space_bijector()
      y = bij.inverse(x) + 0.
      return bij.inverse_log_det_jacobian(
          x, event_ndims=1), bij.forward_log_det_jacobian(
              y, event_ndims=1)

    stream = test_util.test_seed_stream()
    n = 20_000
    samps = tfd.Exponential(rate=1.).sample(n, seed=stream())
    rate = tf.fill([n], tfd.LogNormal(0, .2).sample(seed=stream()))
    exp = tfd.Exponential(rate=rate)
    ldj32_fn = maybe_jit(
        functools.partial(
            ldj_fn,
            dist=tfd.Independent(
                exp,
                reinterpreted_batch_ndims=1,
                experimental_use_kahan_sum=True),
        ))
    ldj32 = ldj32_fn(samps)
    exp64 = tfd.Exponential(rate=tf.cast(rate, tf.float64))
    ldj64 = ldj_fn(
        tf.cast(samps, tf.float64),
        dist=tfd.Independent(exp64, reinterpreted_batch_ndims=1))
    # Evaluate together to ensure we use the same samples.
    ldj32, ldj64 = self.evaluate((ldj32, ldj64))
    self.assertAllCloseNested(ldj64, ldj32, rtol=0., atol=.002)

  def testLargeLogProbDiff(self):
    b = 15
    n = 5_000
    d0 = tfd.Independent(tfd.Normal(tf.fill([b, n], 0.), tf.fill([n], .1)),
                         reinterpreted_batch_ndims=1,
                         experimental_use_kahan_sum=True)
    d1 = tfd.Independent(tfd.Normal(tf.fill([b, n], 1e-5), tf.fill([n], .1)),
                         reinterpreted_batch_ndims=1,
                         experimental_use_kahan_sum=True)
    strm = test_util.test_seed_stream()
    x0 = self.evaluate(  # overdispersed
        tfd.Normal(0, 2).sample([b, n], seed=strm()))
    x1 = self.evaluate(  # overdispersed, perturbed
        x0 + tfd.Normal(0, 1e-6).sample(x0.shape, seed=strm()))
    d0_64 = d0.copy(distribution=tfd.Normal(
        tf.cast(d0.distribution.loc, tf.float64),
        tf.cast(d0.distribution.scale, tf.float64)))
    d1_64 = d1.copy(distribution=tfd.Normal(
        tf.cast(d1.distribution.loc, tf.float64),
        tf.cast(d1.distribution.scale, tf.float64)))
    self.assertNotAllZero(d0.log_prob(x0) < -1_000_000)
    self.assertAllClose(
        d0_64.log_prob(tf.cast(x0, tf.float64)) -
        d1_64.log_prob(tf.cast(x1, tf.float64)),
        tfp.experimental.distributions.log_prob_ratio(d0, x0, d1, x1),
        rtol=0., atol=0.01)
    # In contrast: the below fails consistently w/ errors around 0.5-1.0
    # self.assertAllClose(
    #     d0_64.log_prob(tf.cast(x0, tf.float64)) -
    #     d1_64.log_prob(tf.cast(x1, tf.float64)),
    #     d0.log_prob(x0) - d1.log_prob(x1),
    #     rtol=0., atol=0.007)

if __name__ == '__main__':
  # TODO(b/173158845): XLA:CPU reassociates away the Kahan correction term.
  os.environ['XLA_FLAGS'] = '--xla_cpu_enable_fast_math=false'
  test_util.main()
