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
"""Tests for TransformedDistribution."""

import copy

# Dependency imports
from absl.testing import parameterized  # pylint: disable=unused-import

import numpy as np
from scipy import stats
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.internal import hypothesis_testlib as tfp_hps
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.internal import test_util


class DummyMatrixTransform(tfb.Bijector):
  """Tractable matrix transformation.

  This is a non-sensical bijector that has forward/inverse_min_event_ndims=2.
  The main use is to check that transformed distribution calculations are done
  appropriately.
  """

  def __init__(self):
    parameters = dict(locals())
    super(DummyMatrixTransform, self).__init__(
        forward_min_event_ndims=2,
        is_constant_jacobian=False,
        parameters=parameters,
        validate_args=False,
        name='dummy')

  def _forward(self, x):
    return x

  def _inverse(self, y):
    return y

  # Note: These jacobians don't make sense.
  def _forward_log_det_jacobian(self, x):
    return -tf.linalg.det(x)

  def _inverse_log_det_jacobian(self, x):
    return tf.linalg.det(x)


class _ChooseLocation(tfp.bijectors.Bijector):
  """A Bijector which chooses between one of two location parameters."""

  def __init__(self, loc, name='ChooseLocation'):
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      self._loc = tf.convert_to_tensor(loc, name='loc')
      super(_ChooseLocation, self).__init__(
          is_constant_jacobian=True,
          validate_args=False,
          forward_min_event_ndims=0,
          parameters=parameters,
          name=name)

  def _forward(self, x, z=0.):
    return x + self._gather_loc(z)

  def _inverse(self, x, z=0.):
    return x - self._gather_loc(z)

  def _inverse_log_det_jacobian(self, x, event_ndims, z=None):
    return 0.

  def _gather_loc(self, z=0.):
    z = tf.convert_to_tensor(z)
    z = tf.cast((1 + z) / 2, tf.int32)
    return tf.gather(self._loc, z)


@test_util.test_all_tf_execution_regimes
class TransformedDistributionTest(test_util.TestCase):

  def _make_unimplemented(self, name):
    def _unimplemented(self, *args):  # pylint: disable=unused-argument
      raise NotImplementedError('{} not implemented'.format(name))
    return _unimplemented

  def testTransformedDistribution(self):
    mu = 3.0
    sigma = 2.0
    # Note: the Jacobian callable only works for this example; more generally
    # you may or may not need a reduce_sum.
    log_normal = tfd.TransformedDistribution(
        distribution=tfd.Normal(loc=mu, scale=sigma),
        bijector=tfb.Exp(),
        validate_args=True)
    sp_dist = stats.lognorm(s=sigma, scale=np.exp(mu))

    # sample
    sample = log_normal.sample(100000, seed=test_util.test_seed())
    self.assertAllEqual([], log_normal.event_shape)
    self.assertAllEqual([], self.evaluate(log_normal.event_shape_tensor()))
    self.assertAllClose(
        sp_dist.mean(), np.mean(self.evaluate(sample)), atol=0.0, rtol=0.05)

    # pdf, log_pdf, cdf, etc...
    # The mean of the lognormal is around 148.
    test_vals = np.linspace(0.1, 1000., num=20).astype(np.float32)
    for func in [[log_normal.log_prob, sp_dist.logpdf],
                 [log_normal.prob, sp_dist.pdf],
                 [log_normal.log_cdf, sp_dist.logcdf],
                 [log_normal.cdf, sp_dist.cdf],
                 [log_normal.survival_function, sp_dist.sf],
                 [log_normal.log_survival_function, sp_dist.logsf]]:
      actual = func[0](test_vals)
      expected = func[1](test_vals)
      self.assertAllClose(
          expected, self.evaluate(actual), atol=0, rtol=0.01)

  def testNonInjectiveTransformedDistribution(self):
    mu = 1.
    sigma = 2.0
    abs_normal = tfd.TransformedDistribution(
        distribution=tfd.Normal(loc=mu, scale=sigma),
        bijector=tfb.AbsoluteValue(),
        validate_args=True)
    sp_normal = stats.norm(mu, sigma)

    # sample
    sample = abs_normal.sample(100000, seed=test_util.test_seed())
    self.assertAllEqual([], abs_normal.event_shape)
    sample_ = self.evaluate(sample)
    self.assertAllEqual([], self.evaluate(abs_normal.event_shape_tensor()))

    # Abs > 0, duh!
    np.testing.assert_array_less(0, sample_)

    # Let X ~ Normal(mu, sigma), Y := |X|, then
    # P[Y < 0.77] = P[-0.77 < X < 0.77]
    self.assertAllClose(
        sp_normal.cdf(0.77) - sp_normal.cdf(-0.77),
        (sample_ < 0.77).mean(), rtol=0.01)

    # p_Y(y) = p_X(-y) + p_X(y),
    self.assertAllClose(
        sp_normal.pdf(1.13) + sp_normal.pdf(-1.13),
        self.evaluate(abs_normal.prob(1.13)))

    # Log[p_Y(y)] = Log[p_X(-y) + p_X(y)]
    self.assertAllClose(
        np.log(sp_normal.pdf(2.13) + sp_normal.pdf(-2.13)),
        self.evaluate(abs_normal.log_prob(2.13)))

  def testQuantile(self):
    logit_normal = tfd.TransformedDistribution(
        distribution=tfd.Normal(loc=0., scale=1.),
        bijector=tfb.Sigmoid(),
        validate_args=True)
    grid = [0., 0.25, 0.5, 0.75, 1.]
    q = logit_normal.quantile(grid)
    cdf = logit_normal.cdf(q)
    cdf_ = self.evaluate(cdf)
    self.assertAllClose(grid, cdf_, rtol=1e-6, atol=0.)

  def testCdfDescending(self):
    td = tfd.TransformedDistribution(
        distribution=tfd.Normal(loc=0., scale=[1., 1.]),
        bijector=tfb.Shift(shift=1.)(tfb.Scale(scale=[2., -2.])),
        validate_args=True)
    nd = tfd.Normal(loc=1., scale=2., validate_args=True)
    self.assertAllEqual(tf.ones(td.batch_shape, dtype=tf.bool),
                        td.cdf(nd.quantile(.8)) < td.cdf(nd.quantile(.9)))

  def testCdfDescendingChained(self):
    bij1 = tfb.Shift(shift=1.)(tfb.Scale(scale=[1., -2.]))
    bij2 = tfb.Shift(shift=1.)(tfb.Scale(scale=[[3.], [-5.]]))
    bij3 = tfb.Shift(shift=1.)(tfb.Scale(scale=[[[7.]], [[-11.]]]))
    for chain in bij2(bij1), bij3(bij2(bij1)):
      td = tfd.TransformedDistribution(
          distribution=tfd.Normal(loc=0., scale=tf.ones([2, 2, 2])),
          bijector=chain,
          validate_args=True)
      nd = tfd.Normal(loc=1., scale=3., validate_args=True)
      self.assertAllEqual(tf.ones(td.batch_shape, dtype=tf.bool),
                          td.cdf(nd.quantile(.4)) < td.cdf(nd.quantile(.6)),
                          msg=chain.name)

  def testSfDescending(self):
    td = tfd.TransformedDistribution(
        distribution=tfd.Normal(loc=0., scale=[1., 1.]),
        bijector=tfb.Shift(shift=1.)(tfb.Scale(scale=[2., -2.])),
        validate_args=True)
    nd = tfd.Normal(loc=1., scale=2., validate_args=True)
    self.assertAllEqual(tf.ones(td.batch_shape, dtype=tf.bool),
                        td.survival_function(nd.quantile(.8)) >
                        td.survival_function(nd.quantile(.9)))

  def testQuantileDescending(self):
    td = tfd.TransformedDistribution(
        distribution=tfd.Normal(loc=0., scale=[1., 1.]),
        bijector=tfb.Shift(shift=1.)(tfb.Scale(scale=[2., -2.])),
        validate_args=True)
    self.assertAllEqual(tf.ones(td.batch_shape, dtype=tf.bool),
                        td.quantile(.8) < td.quantile(.9))

  def testCachedSamples(self):
    class ExpForwardOnly(tfb.Bijector):

      def __init__(self):
        parameters = dict(locals())
        super(ExpForwardOnly, self).__init__(
            forward_min_event_ndims=0,
            parameters=parameters)

      def _forward(self, x):
        return tf.exp(x)

      def _forward_log_det_jacobian(self, x):
        return tf.convert_to_tensor(x)

    exp_forward_only = ExpForwardOnly()

    mu = 3.0
    sigma = 0.02
    log_normal = tfd.TransformedDistribution(
        distribution=tfd.Normal(loc=mu, scale=sigma),
        bijector=exp_forward_only,
        validate_args=True)

    sample = log_normal.sample([2, 3], seed=test_util.test_seed())
    sample_val, log_pdf_val = self.evaluate(
        [sample, log_normal.log_prob(sample)])
    expected_log_pdf = stats.lognorm.logpdf(
        sample_val, s=sigma, scale=np.exp(mu))
    self.assertAllClose(expected_log_pdf, log_pdf_val, rtol=1e-4, atol=0.)

    # Check that nesting TransformedDistributions preserves caching.
    identity_log_normal = tfd.TransformedDistribution(
        distribution=log_normal,
        bijector=tfb.Identity(),
        validate_args=True)
    identity_log_normal.log_prob(
        identity_log_normal.sample([2, 3], seed=test_util.test_seed()))

  def testSampleAndLogprob(self):
    class ExpForwardOnly(tfb.Bijector):

      def __init__(self):
        super(ExpForwardOnly, self).__init__(forward_min_event_ndims=0)

      def _forward(self, x):
        return tf.exp(x)

      def _forward_log_det_jacobian(self, x):
        return tf.convert_to_tensor(value=x)

    exp_forward_only = ExpForwardOnly()

    mu = 3.0
    sigma = 0.02
    log_normal = tfd.TransformedDistribution(
        distribution=tfd.Normal(loc=mu, scale=sigma),
        bijector=exp_forward_only)

    sample, log_pdf = self.evaluate(log_normal.experimental_sample_and_log_prob(
        [2, 3], seed=test_util.test_seed()))
    expected_log_pdf = stats.lognorm.logpdf(
        sample, s=sigma, scale=np.exp(mu))
    self.assertAllClose(expected_log_pdf, log_pdf, rtol=1e-4, atol=0.)

    sample, log_pdf = self.evaluate(
        log_normal.experimental_sample_and_log_prob(seed=test_util.test_seed()))
    expected_log_pdf = stats.lognorm.logpdf(
        sample, s=sigma, scale=np.exp(mu))
    self.assertAllClose(expected_log_pdf, log_pdf, rtol=1e-4, atol=0.)

    sample2 = self.evaluate(log_normal.sample(seed=test_util.test_seed()))
    self.assertAllClose(sample, sample2, rtol=1e-4)

  def testCachedSamplesInvert(self):
    class ExpInverseOnly(tfb.Bijector):

      def __init__(self):
        parameters = dict(locals())
        super(ExpInverseOnly, self).__init__(
            inverse_min_event_ndims=0,
            parameters=parameters)

      def _inverse(self, y):
        return tf.math.log(y)

      def _inverse_log_det_jacobian(self, y):
        return -tf.math.log(y)

    exp_inverse_only = ExpInverseOnly()

    log_forward_only = tfb.Invert(exp_inverse_only)

    # The log bijector isn't defined over the whole real line, so we make
    # sigma sufficiently small so that the draws are positive.
    mu = 2.
    sigma = 1e-2
    exp_normal = tfd.TransformedDistribution(
        distribution=tfd.Normal(loc=mu, scale=sigma),
        bijector=log_forward_only,
        validate_args=True)

    sample = exp_normal.sample(
        [2, 3], seed=test_util.test_seed(hardcoded_seed=42))
    sample_val, log_pdf_val = self.evaluate(
        [sample, exp_normal.log_prob(sample)])
    expected_log_pdf = sample_val + stats.norm.logpdf(
        np.exp(sample_val), loc=mu, scale=sigma)
    self.assertAllClose(expected_log_pdf, log_pdf_val, atol=0., rtol=1e-5)

  def testShapeChangingBijector(self):
    softmax = tfb.SoftmaxCentered()
    standard_normal = tfd.MultivariateNormalDiag(loc=0., scale_diag=[1.])
    multi_logit_normal = tfd.TransformedDistribution(
        distribution=standard_normal,
        bijector=softmax,
        validate_args=True)
    x = [[[-np.log(3.)], [0.]], [[np.log(3)], [np.log(5)]]]
    x = np.float32(x)
    y = self.evaluate(softmax.forward(x))
    expected_log_pdf = -0.5 * np.log(2) + (
        np.squeeze(stats.norm(loc=0., scale=1.).logpdf(x)) -
        np.sum(np.log(y), axis=-1))
    self.assertAllClose(expected_log_pdf,
                        self.evaluate(multi_logit_normal.log_prob(y)))
    self.assertAllClose(
        [1, 2, 3, 2],
        self.evaluate(tf.shape(multi_logit_normal.sample(
            [1, 2, 3], seed=test_util.test_seed()))))
    self.assertAllEqual([2], multi_logit_normal.event_shape)
    self.assertAllEqual([2],
                        self.evaluate(multi_logit_normal.event_shape_tensor()))

  def testCastLogDetJacobian(self):
    """Test log_prob when Jacobian and log_prob dtypes do not match."""

    # Create an identity bijector whose jacobians have dtype int32
    int_identity = tfb.Inline(
        forward_fn=tf.identity,
        inverse_fn=tf.identity,
        inverse_log_det_jacobian_fn=(lambda y: tf.cast(0, tf.int32)),
        forward_log_det_jacobian_fn=(lambda x: tf.cast(0, tf.int32)),
        forward_min_event_ndims=0,
        is_constant_jacobian=True)
    normal = tfd.TransformedDistribution(
        distribution=tfd.Normal(loc=0., scale=1.),
        bijector=int_identity,
        validate_args=True)

    y = normal.sample(seed=test_util.test_seed())
    self.evaluate(normal.log_prob(y))
    self.evaluate(normal.prob(y))
    self.evaluate(normal.mean())
    self.evaluate(normal.entropy())

  def testMode(self):
    dist = tfd.TransformedDistribution(
        tfd.Beta(
            concentration1=[5., 10.],
            concentration0=15.,
            validate_args=True),
        tfb.Shift(2., validate_args=True)(tfb.Scale(10., validate_args=True)),
        validate_args=True)
    self.assertAllClose(2. + 10. * dist.distribution.mode(),
                        self.evaluate(dist.mode()),
                        atol=0., rtol=1e-6)

  def testMean(self):
    shift = np.array([[-1, 0, 1], [-1, -2, -3]], dtype=np.float32)
    diag = np.array([[1, 2, 3], [2, 3, 2]], dtype=np.float32)
    fake_mvn = tfd.TransformedDistribution(
        tfd.MultivariateNormalDiag(
            loc=tf.zeros_like(shift),
            scale_diag=tf.ones_like(diag),
            validate_args=True),
        tfb.Chain([
            tfb.Shift(shift=shift),
            tfb.ScaleMatvecLinearOperator(
                scale=tf.linalg.LinearOperatorDiag(diag, is_non_singular=True))
        ], validate_args=True),
        validate_args=True)
    self.assertAllClose(shift, self.evaluate(fake_mvn.mean()))

  def testStddev(self):
    base_stddev = 2.
    shift = np.array([[-1, 0, 1], [-1, -2, -3]], dtype=np.float32)
    scale = np.array([[1, -2, 3], [2, -3, 2]], dtype=np.float32)
    expected_stddev = tf.abs(base_stddev * scale)
    normal = tfd.TransformedDistribution(
        distribution=tfd.Normal(loc=tf.zeros_like(shift),
                                scale=base_stddev * tf.ones_like(scale),
                                validate_args=True),
        bijector=tfb.Chain([tfb.Shift(shift=shift),
                            tfb.Scale(scale=scale)],
                           validate_args=True),
        validate_args=True)
    self.assertAllClose(expected_stddev, normal.stddev())
    self.assertAllClose(expected_stddev**2, normal.variance())

    split_normal = tfd.TransformedDistribution(
        distribution=tfd.Independent(normal, reinterpreted_batch_ndims=1),
        bijector=tfb.Split(3),
        validate_args=True)
    self.assertAllCloseNested(tf.split(expected_stddev,
                                       num_or_size_splits=3,
                                       axis=-1),
                              split_normal.stddev())

    scaled_normal = tfd.TransformedDistribution(
        distribution=tfd.Independent(normal, reinterpreted_batch_ndims=1),
        bijector=tfb.ScaleMatvecTriL([[1., 0.], [-1., 2.]]),
        validate_args=True)
    with self.assertRaisesRegex(
        NotImplementedError, 'is a multivariate transformation'):
      scaled_normal.stddev()

  def testEntropy(self):
    shift = np.array([[-1, 0, 1], [-1, -2, -3]], dtype=np.float32)
    diag = np.array([[1, 2, 3], [2, 3, 2]], dtype=np.float32)
    actual_mvn_entropy = np.concatenate(
        [[stats.multivariate_normal(shift[i], np.diag(diag[i]**2)).entropy()]
         for i in range(len(diag))])
    fake_mvn = tfd.TransformedDistribution(
        tfd.MultivariateNormalDiag(
            loc=tf.zeros_like(shift),
            scale_diag=tf.ones_like(diag),
            validate_args=True),
        tfb.Chain([
            tfb.Shift(shift=shift),
            tfb.ScaleMatvecLinearOperator(
                scale=tf.linalg.LinearOperatorDiag(diag, is_non_singular=True))
        ], validate_args=True),
        validate_args=True)
    self.assertAllClose(actual_mvn_entropy, self.evaluate(fake_mvn.entropy()))

  def testScalarBatchScalarEventIdentityScale(self):
    exp2 = tfd.TransformedDistribution(
        tfd.Exponential(rate=0.25),
        bijector=tfb.Scale(scale=2.),
        validate_args=True)
    log_prob = exp2.log_prob(1.)
    log_prob_ = self.evaluate(log_prob)
    base_log_prob = -0.5 * 0.25 + np.log(0.25)
    ildj = np.log(2.)
    self.assertAllClose(base_log_prob - ildj, log_prob_, rtol=1e-6, atol=0.)

  def testTransformedKLDifferentBijectorFails(self):
    d1 = tfd.TransformedDistribution(
        tfd.Exponential(rate=0.25),
        bijector=tfb.Scale(scale=2.),
        validate_args=True)
    d2 = tfd.TransformedDistribution(
        tfd.Exponential(rate=0.25),
        bijector=tfb.Scale(scale=3.),
        validate_args=True)
    with self.assertRaisesRegex(
        NotImplementedError, r'their bijectors are not equal'):
      tfd.kl_divergence(d1, d2)

  def testTransformedNormalNormalKL(self):
    batch_size = 6
    mu_a = np.array([3.0] * batch_size).astype(np.float32)
    sigma_a = np.array([1.0, 2.0, 3.0, 1.5, 2.5, 3.5]).astype(np.float32)
    mu_b = np.array([-3.0] * batch_size).astype(np.float32)
    sigma_b = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0]).astype(np.float32)

    n_a = tfd.Normal(loc=mu_a, scale=sigma_a, validate_args=True)
    n_b = tfd.Normal(loc=mu_b, scale=sigma_b, validate_args=True)

    kl_expected = ((mu_a - mu_b)**2 / (2 * sigma_b**2) + 0.5 * (
        (sigma_a**2 / sigma_b**2) - 1 - 2 * np.log(sigma_a / sigma_b)))

    bij1 = tfb.Shift(shift=1.)(tfb.Scale(scale=2.))
    bij2 = (tfb.Shift(shift=np.array(2., dtype=np.float32))
            (tfb.Scale(scale=np.array(3., dtype=np.float32))))
    bij3 = tfb.Tanh()
    for chain in bij2(bij1), bij3(bij2(bij1)):
      td_a = tfd.TransformedDistribution(
          distribution=n_a,
          bijector=chain,
          validate_args=True)
      td_b = tfd.TransformedDistribution(
          distribution=n_b,
          bijector=copy.copy(chain),
          validate_args=True)

      kl = tfd.kl_divergence(td_a, td_b)
      kl_val = self.evaluate(kl)

      x = td_a.sample(int(1e5), seed=test_util.test_seed())
      kl_sample = tf.reduce_mean(td_a.log_prob(x) - td_b.log_prob(x), axis=0)
      kl_sample_ = self.evaluate(kl_sample)

      self.assertEqual(kl.shape, (batch_size,))
      self.assertAllClose(kl_val, kl_expected)
      self.assertAllClose(kl_expected, kl_sample_, atol=0.0, rtol=1e-2)

  def testLogProbRatio(self):
    nsamp = 5
    nbatch = 3
    dim = 5000
    d0 = tfd.MultivariateNormalDiag(tf.fill([nbatch, dim], 0.),
                                    tf.fill([dim], .1))
    d1 = tfd.MultivariateNormalDiag(tf.fill([nbatch, dim], 1e-5),
                                    d0.scale.diag)
    strm = test_util.test_seed_stream()
    x0 = self.evaluate(  # overdispersed
        tfd.Normal(0, 2).sample([nsamp, nbatch, dim], seed=strm()))
    x1 = self.evaluate(  # overdispersed + perturbed
        x0 + tfd.Normal(0, 1e-6).sample(x0.shape, seed=strm()))
    d0_64 = tfd.MultivariateNormalDiag(
        tf.cast(d0.loc, tf.float64), tf.cast(d0.scale.diag, tf.float64))
    d1_64 = tfd.MultivariateNormalDiag(
        tf.cast(d1.loc, tf.float64), tf.cast(d1.scale.diag, tf.float64))
    oracle_64 = (d0_64.log_prob(tf.cast(x0, tf.float64)) -
                 d1_64.log_prob(tf.cast(x1, tf.float64)))
    # For a sense of the order of magnitude log_probs we're dealing with:
    self.assertNotAllZero(d0.log_prob(x0) < -1_000_000.)
    self.assertAllClose(
        oracle_64,
        tfp.experimental.distributions.log_prob_ratio(d0, x0, d1, x1),
        rtol=0., atol=0.007)
    # In contrast, this test fails with max-abs-error around 0.05 to 0.1
    # self.assertAllClose(
    #     oracle_64,
    #     d0.copy(experimental_use_kahan_sum=True).log_prob(x0) -
    #     d1.copy(experimental_use_kahan_sum=True).log_prob(x1),
    #     rtol=0., atol=0.007)
    # In contrast, this test fails with max-abs-error around 0.8 to 1.5
    # self.assertAllClose(
    #     oracle_64, d0.log_prob(x0) - d1.log_prob(x1),
    #     rtol=0., atol=0.007)


@test_util.test_all_tf_execution_regimes
class ScalarToMultiTest(test_util.TestCase):

  def testNormalBatchBroadcasting(self):
    td = tfd.TransformedDistribution(
        distribution=tfd.Normal(0., 1.),
        bijector=tfb.Shift([1., 1., 1.]))

    self.assertAllEqual(td.event_shape, [])
    self.assertAllEqual(td.event_shape_tensor(), [])
    self.assertAllEqual(td.batch_shape, [3])
    self.assertAllEqual(td.batch_shape_tensor(), [3])

    x = self.evaluate(td.sample(seed=test_util.test_seed()))
    self.assertAllEqual(x.shape, [3])
    self.assertAllEqual(td.log_prob(x).shape, [3])
    # Check that we got a different sample for each batch element.
    self.assertLen(np.unique(x), 3)

    x, lp = self.evaluate(
        td.experimental_sample_and_log_prob(seed=test_util.test_seed()))
    self.assertAllEqual(x.shape, [3])
    self.assertAllEqual(lp.shape, [3])
    # Check that we got a different sample for each batch element.
    self.assertLen(np.unique(x), 3)

  @parameterized.named_parameters(
      {'testcase_name': 'static',
       'event_shape': [3],
       'shift': np.array([-1, 0, 1], dtype=np.float32),
       'tril': np.array([[2, 0, 0],  # Shape [3, 3].
                         [3, 2, 0],
                         [4, 3, 2]], dtype=np.float32),
       'dynamic_shape': False},
      {'testcase_name': 'batch_static',
       'event_shape': [3],
       'shift': np.ones([4, 1, 3], dtype=np.float32),
       'tril': np.array([[[1., 0, 0],  # Shape [2, 3, 3].
                          [2, 1, 0],
                          [3, 2, 1]],
                         [[2, 0, 0],
                          [3, 2, 0],
                          [4, 3, 2]]], dtype=np.float32),
       'dynamic_shape': False},
      {'testcase_name': 'batch_dynamic',
       'event_shape': [3],
       'shift': np.ones([4, 1, 3], dtype=np.float32),
       'tril': np.array([[[1., 0, 0],  # Shape [2, 3, 3].
                          [2, 1, 0],
                          [3, 2, 1]],
                         [[2, 0, 0],
                          [3, 2, 0],
                          [4, 3, 2]]], dtype=np.float32),
       'dynamic_shape': True})
  def testMVN(self, event_shape, shift, tril, dynamic_shape):
    if dynamic_shape and tf.executing_eagerly():
      self.skipTest('Eager execution does not support dynamic shape.')
    as_tensor = tf.convert_to_tensor
    if dynamic_shape:
      as_tensor = lambda v, name: tf1.placeholder_with_default(  # pylint: disable=g-long-lambda
          v, shape=None, name='dynamic_' + name)

    fake_mvn = tfd.TransformedDistribution(
        distribution=tfd.Sample(
            tfd.Normal(loc=as_tensor(0., name='loc'),
                       scale=as_tensor(1., name='scale'),
                       validate_args=True),
            sample_shape=as_tensor(np.int32(event_shape), name='event_shape')),
        bijector=tfb.Chain(
            [tfb.Shift(shift=as_tensor(shift, name='shift')),
             tfb.ScaleMatvecTriL(scale_tril=as_tensor(tril, name='scale_tril'))
             ]), validate_args=True)

    base_dist = fake_mvn.distribution
    expected_mean = tf.linalg.matvec(
        tril, tf.broadcast_to(base_dist.mean(), shift.shape)) + shift
    expected_cov = tf.linalg.matmul(
        tril,
        tf.matmul(
            tf.linalg.diag(tf.broadcast_to(base_dist.variance(), shift.shape)),
            tril,
            adjoint_b=True))
    expected_batch_shape = ps.shape(expected_mean)[:-1]

    if dynamic_shape:
      self.assertAllEqual(tf.TensorShape(None), fake_mvn.event_shape)
      self.assertAllEqual(tf.TensorShape(None), fake_mvn.batch_shape)
    else:
      self.assertAllEqual(event_shape, fake_mvn.event_shape)
      self.assertAllEqual(expected_batch_shape, fake_mvn.batch_shape)

    # Ensure sample works by checking first, second moments.
    num_samples = 7e3
    y = fake_mvn.sample(int(num_samples), seed=test_util.test_seed())
    x = y[0:5, ...]
    self.assertAllClose(expected_mean, tf.reduce_mean(y, axis=0),
                        atol=0.1, rtol=0.1)
    self.assertAllClose(expected_cov, tfp.stats.covariance(y, sample_axis=0),
                        atol=0., rtol=0.1)

    self.assertAllEqual(event_shape, fake_mvn.event_shape_tensor())
    self.assertAllEqual(expected_batch_shape, fake_mvn.batch_shape_tensor())
    self.assertAllEqual(
        ps.concat([[5], expected_batch_shape, event_shape], axis=0),
        ps.shape(x))
    self.assertAllClose(expected_mean, fake_mvn.mean())

  @parameterized.named_parameters(('static', False), ('dynamic', True))
  def testMatrixEvent(self, dynamic_shape):
    if dynamic_shape and tf.executing_eagerly():
      self.skipTest('Eager execution does not support dynamic shape.')
    as_tensor = tf.convert_to_tensor
    if dynamic_shape:
      as_tensor = lambda v, name: tf1.placeholder_with_default(  # pylint: disable=g-long-lambda
          v, shape=None, name='dynamic_' + name)

    loc = 0.
    scale = 2.
    fake_mvn = tfd.TransformedDistribution(
        distribution=tfd.Sample(
            tfd.Normal(loc=as_tensor([loc] * 2, name='loc'),
                       scale=as_tensor(scale, name='scale')),
            sample_shape=as_tensor(np.int32([2, 3, 3]), name='event_shape')),
        bijector=DummyMatrixTransform(),
        validate_args=True)

    def actual_mvn_log_prob(x):
      # This distribution is the normal PDF, reduced over the
      # last 3 dimensions + a jacobian term which corresponds
      # to the determinant of x.
      return (np.sum(stats.norm(loc, scale).logpdf(x), axis=(-1, -2, -3)) +
              np.sum(np.linalg.det(x), axis=-1))

    if dynamic_shape:
      self.assertAllEqual(tf.TensorShape(None), fake_mvn.event_shape)
      self.assertAllEqual(tf.TensorShape(None), fake_mvn.batch_shape)
    else:
      self.assertAllEqual([2, 3, 3], fake_mvn.event_shape)
      self.assertAllEqual([2], fake_mvn.batch_shape)

    # Ensure all other functions work as intended.
    x = self.evaluate(fake_mvn.sample(5, seed=test_util.test_seed()))
    self.assertAllEqual([5, 2, 2, 3, 3], x.shape)
    self.assertAllEqual([2, 3, 3], fake_mvn.event_shape_tensor())
    self.assertAllEqual([2], fake_mvn.batch_shape_tensor())
    self.assertAllClose(
        actual_mvn_log_prob(x), fake_mvn.log_prob(x), atol=0., rtol=1e-6)
    # With this many dimensions and samples, the direct space probability
    # may underflow.
    self.assertAllClose(
        np.exp(actual_mvn_log_prob(x)),
        fake_mvn.prob(x),
        atol=1e-12, rtol=1e-5)

  @parameterized.named_parameters(
      {'testcase_name': 'scalar_static',
       'batch_shape': [],
       'shapes_are_dynamic': False},
      {'testcase_name': 'batch_static',
       'batch_shape': [2],
       'shapes_are_dynamic': True},
      {'testcase_name': 'scalar_dynamic',
       'batch_shape': [],
       'shapes_are_dynamic': False},
      {'testcase_name': 'batch_dynamic',
       'batch_shape': [2],
       'shapes_are_dynamic': True})
  def testEmptyEvent(self, batch_shape, shapes_are_dynamic):
    # Verify that zero-dimensional multivariate Normal distributions still
    # return reasonable shapes and a log-prob of 0.0.

    event_shape = [0]
    loc = tf.zeros(batch_shape + event_shape)
    scale_diag = tf.ones(batch_shape + event_shape)

    if shapes_are_dynamic:
      loc = tf.Variable(
          loc, shape=tf.TensorShape(None), name='dynamic_loc')
      scale_diag = tf.Variable(
          scale_diag, shape=tf.TensorShape(None), name='dynamic_scale_diag')
      self.evaluate([loc.initializer, scale_diag.initializer])

    mvn = tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale_diag)

    self.assertAllEqual(self.evaluate(mvn.event_shape_tensor()),
                        event_shape)
    self.assertAllEqual(self.evaluate(mvn.batch_shape_tensor()),
                        batch_shape)
    if not shapes_are_dynamic:
      self.assertAllEqual(
          tensorshape_util.as_list(mvn.event_shape), event_shape)
      self.assertAllEqual(
          tensorshape_util.as_list(mvn.batch_shape), batch_shape)

    for sample_shape in ([3], []):
      sample_ = self.evaluate(mvn.sample(
          sample_shape, seed=test_util.test_seed()))
      self.assertAllEqual(sample_.shape,
                          sample_shape + batch_shape + event_shape)
      self.assertAllEqual(
          self.evaluate(mvn.log_prob(sample_)),
          np.zeros(sample_shape + batch_shape))

  def testConditioning(self):
    conditional_normal = tfd.TransformedDistribution(
        distribution=tfd.Normal(loc=0., scale=1.),
        bijector=_ChooseLocation(loc=[-100., 100.]))
    z = [-1, +1, -1, -1, +1]
    self.assertAllClose(
        np.sign(
            self.evaluate(
                conditional_normal.sample(
                    5, seed=test_util.test_seed(),
                    bijector_kwargs={'z': z}))), z)

  def testSupportBijectorOutsideRange(self):
    log_normal = tfd.TransformedDistribution(
        distribution=tfd.Normal(loc=1., scale=2.),
        bijector=tfb.Exp(),
        validate_args=True)
    x = np.array([-4.2, -1e-6, -1.3])
    bijector_inverse_x = (
        log_normal.experimental_default_event_space_bijector().inverse(x))
    self.assertAllNan(self.evaluate(bijector_inverse_x))

  def test_unknown_event_rank(self):
    if tf.executing_eagerly():
      self.skipTest('Eager execution.')
    unknown_rank_dist = tfd.Independent(
        tfd.Normal(loc=tf.ones([2, 1, 3]), scale=2.),
        reinterpreted_batch_ndims=tf1.placeholder_with_default(1, shape=[]))
    td = tfd.TransformedDistribution(
        distribution=unknown_rank_dist,
        bijector=tfb.Scale(1.),
        validate_args=True)
    self.assertEqual(td.batch_shape, tf.TensorShape(None))
    self.assertEqual(td.event_shape, tf.TensorShape(None))
    self.assertAllEqual(td.batch_shape_tensor(), [2, 1])
    self.assertAllEqual(td.event_shape_tensor(), [3])

    joint_td = tfd.TransformedDistribution(
        distribution=tfd.JointDistributionSequentialAutoBatched(
            [unknown_rank_dist, unknown_rank_dist]),
        bijector=tfb.Invert(tfb.Split(2)),
        validate_args=True)
    # Note that the current behavior is conservative; we could also correctly
    # return a batch shape of `[]` in this case.
    self.assertEqual(joint_td.batch_shape, tf.TensorShape(None))
    self.assertEqual(joint_td.event_shape, tf.TensorShape(None))
    self.assertAllEqual(joint_td.batch_shape_tensor(), [])
    self.assertAllEqual(joint_td.event_shape_tensor(), [2, 1, 6])


@test_util.test_all_tf_execution_regimes
class ExcessiveConcretizationTest(test_util.TestCase):

  def setUp(self):
    super(ExcessiveConcretizationTest, self).setUp()

    self.max_permissible = {
        'mean': 2,
        'sample': 2,
        'log_cdf': 2,
        'cdf': 2,
        'survival_function': 2,
        'log_survival_function': 2,

        # extra concretizations primarily of bijector parameters
        'entropy': 5,
        'log_prob': 6,
        'prob': 6,
        'quantile': 2,

        'event_shape_tensor': 2,
        'batch_shape_tensor': 3,
    }

    self.shape = None

  def testExcessiveConcretizationOfParams(self):
    shape_kwargs = {'shape': self.shape} if self.shape else {}
    loc = tfp_hps.defer_and_count_usage(
        tf.Variable(0., name='loc', dtype=tf.float32, **shape_kwargs))
    scale = tfp_hps.defer_and_count_usage(
        tf.Variable(2., name='scale', dtype=tf.float32, **shape_kwargs))
    bij_scale = tfp_hps.defer_and_count_usage(
        tf.Variable(2., name='bij_scale', dtype=tf.float32, **shape_kwargs))
    dist = tfd.TransformedDistribution(
        distribution=tfd.Normal(loc=loc, scale=scale, validate_args=True),
        bijector=tfb.Scale(scale=bij_scale, validate_args=True),
        validate_args=True)

    for method in ('mean', 'entropy', 'event_shape_tensor',
                   'batch_shape_tensor'):
      with tfp_hps.assert_no_excessive_var_usage(
          method, max_permissible=self.max_permissible[method]):
        getattr(dist, method)()

    with tfp_hps.assert_no_excessive_var_usage(
        'sample', max_permissible=self.max_permissible['sample']):
      dist.sample(seed=test_util.test_seed())

    for method in ('log_prob', 'prob'):
      with tfp_hps.assert_no_excessive_var_usage(
          method, max_permissible=self.max_permissible[method]):
        getattr(dist, method)(np.ones((4, 3, 5, 2, 2)) / 3.)


@test_util.test_all_tf_execution_regimes
class ExcessiveConcretizationTestUnknownShape(ExcessiveConcretizationTest):

  def setUp(self):
    super(ExcessiveConcretizationTestUnknownShape, self).setUp()

    self.max_permissible = {

        # extra concretizations primarily of base distribution parameters
        'mean': 2,
        'sample': 15,  # Unknown shape forces BatchBroadcast wrapping.
        'log_cdf': 2,
        'cdf': 2,
        'survival_function': 2,
        'log_survival_function': 2,
        'entropy': 5,
        'log_prob': 6,
        'prob': 6,
        'quantile': 2,
        'event_shape_tensor': 2,
        'batch_shape_tensor': 3,
    }

    self.shape = tf.TensorShape(None)


@test_util.test_all_tf_execution_regimes
class MultipartBijectorsTest(test_util.TestCase):

  @parameterized.named_parameters(
      {'testcase_name': 'split_sizes_known',
       'known_split_sizes': [1, 3, 2]},
      {'testcase_name': 'split_size_unknown',
       'known_split_sizes': [1, -1, 2]}
      )
  def test_transform_parts_to_vector(self, known_split_sizes):
    batch_shape = [4, 2]
    true_split_sizes = [1, 3, 2]

    # Create a joint distribution with parts of the specified sizes.
    seed = test_util.test_seed_stream()
    component_dists = tf.nest.map_structure(
        lambda size: tfd.MultivariateNormalDiag(  # pylint: disable=g-long-lambda
            loc=tf.random.normal(batch_shape + [size], seed=seed()),
            scale_diag=tf.exp(
                tf.random.normal(batch_shape + [size], seed=seed()))),
        true_split_sizes)
    base_dist = tfd.JointDistributionSequential(component_dists)

    # Transform to a vector-valued distribution by concatenating the parts.
    bijector = tfb.Invert(tfb.Split(known_split_sizes, axis=-1))

    concat_dist = tfd.TransformedDistribution(base_dist, bijector)
    self.assertAllEqual(concat_dist.event_shape, [sum(true_split_sizes)])
    self.assertAllEqual(self.evaluate(concat_dist.event_shape_tensor()),
                        [sum(true_split_sizes)])
    self.assertAllEqual(concat_dist.batch_shape, batch_shape)
    self.assertAllEqual(self.evaluate(concat_dist.batch_shape_tensor()),
                        batch_shape)

    # Since the Split bijector has (constant) unit Jacobian, the transformed
    # entropy and mean/mode should match the base entropy and (split) base
    # mean/mode.
    self.assertAllEqual(*self.evaluate(
        (base_dist.entropy(), concat_dist.entropy())))

    self.assertAllEqual(*self.evaluate(
        (concat_dist.mean(), bijector.forward(base_dist.mean()))))
    self.assertAllEqual(*self.evaluate(
        (concat_dist.mode(), bijector.forward(base_dist.mode()))))

    # Since the Split bijector has zero Jacobian, the transformed `log_prob`
    # and `prob` should match the base distribution.
    sample_shape = [3]
    x = base_dist.sample(sample_shape, seed=seed())
    y = bijector.forward(x)
    for attr in ('log_prob', 'prob'):
      base_attr = getattr(base_dist, attr)(x)
      concat_attr = getattr(concat_dist, attr)(y)
      self.assertAllClose(*self.evaluate((base_attr, concat_attr)))

    # Test that `.sample()` works and returns a result of the expected structure
    # and shape.
    y_sampled = concat_dist.sample(sample_shape, seed=seed())
    self.assertAllEqual(y.shape, y_sampled.shape)

  @parameterized.named_parameters(
      {'testcase_name': 'split_sizes_known',
       'known_split_sizes': [1, 3, 2]},
      {'testcase_name': 'split_size_unknown',
       'known_split_sizes': [1, -1, 2]}
      )
  def test_transform_vector_to_parts(self, known_split_sizes):
    batch_shape = [4, 2]
    true_split_sizes = [1, 3, 2]

    base_event_size = sum(true_split_sizes)
    base_dist = tfd.MultivariateNormalDiag(
        loc=tf.random.normal(
            batch_shape + [base_event_size], seed=test_util.test_seed()),
        scale_diag=tf.exp(tf.random.normal(
            batch_shape + [base_event_size], seed=test_util.test_seed())))

    bijector = tfb.Split(known_split_sizes, axis=-1)
    split_dist = tfd.TransformedDistribution(base_dist, bijector)

    self.assertRegex(
        str(split_dist),
        '{}.*batch_shape.*event_shape.*dtype'.format(split_dist.name))

    expected_event_shape = [np.array([s]) for s in true_split_sizes]
    output_event_shape = [np.array(s) for s in split_dist.event_shape]
    self.assertAllEqual(output_event_shape, expected_event_shape)
    self.assertAllEqual(self.evaluate(split_dist.event_shape_tensor()),
                        expected_event_shape)
    self.assertAllEqual(split_dist.batch_shape, batch_shape)
    self.assertAllEqual(self.evaluate(split_dist.batch_shape_tensor()),
                        batch_shape)

    # Since the Split bijector has (constant) unit Jacobian, the transformed
    # entropy and mean/mode should match the base entropy and (split) base
    # mean/mode.
    self.assertAllEqual(*self.evaluate(
        (base_dist.entropy(), split_dist.entropy())))
    self.assertAllEqualNested(
        *self.evaluate((split_dist.mean(),
                        bijector.forward(base_dist.mean()))))
    self.assertAllEqualNested(
        *self.evaluate((split_dist.mode(),
                        bijector.forward(base_dist.mode()))))

    # Since the Split bijector has zero Jacobian, the transformed `log_prob`
    # and `prob` should match the base distribution.
    sample_shape = [3]
    x = base_dist.sample(sample_shape, seed=test_util.test_seed())
    y = bijector.forward(x)
    for attr in ('log_prob', 'prob'):
      split_attr = getattr(split_dist, attr)(y)
      base_attr = getattr(base_dist, attr)(x)
      self.assertAllClose(*self.evaluate((base_attr, split_attr)), rtol=1e-5)

    # Test that `.sample()` works and returns a result of the expected structure
    # and shape.
    y_sampled = split_dist.sample(sample_shape, seed=test_util.test_seed())
    self.assertAllEqual([x.shape for x in y], [x.shape for x in y_sampled])

  @parameterized.named_parameters(
      {'testcase_name': 'sequential',
       'split_sizes': [1, 3, 2]},
      {'testcase_name': 'named',
       'split_sizes': {'a': 1, 'b': 3, 'c': 2}},)
  def test_transform_joint_to_joint(self, split_sizes):
    dist_batch_shape = tf.nest.pack_sequence_as(
        split_sizes,
        [tensorshape_util.constant_value_as_shape(s)
         for s in [[2, 3], [2, 1], [1, 3]]])
    bijector_batch_shape = [1, 3]

    # Build a joint distribution with parts of the specified sizes.
    seed = test_util.test_seed_stream()
    component_dists = tf.nest.map_structure(
        lambda size, batch_shape: tfd.MultivariateNormalDiag(  # pylint: disable=g-long-lambda
            loc=tf.random.normal(batch_shape + [size], seed=seed()),
            scale_diag=tf.random.uniform(
                minval=1., maxval=2.,
                shape=batch_shape + [size], seed=seed())),
        split_sizes, dist_batch_shape)
    if isinstance(split_sizes, dict):
      base_dist = tfd.JointDistributionNamed(component_dists)
    else:
      base_dist = tfd.JointDistributionSequential(component_dists)

    # Transform the distribution by applying a separate bijector to each part.
    bijectors = [tfb.Exp(),
                 tfb.Scale(
                     tf.random.uniform(
                         minval=1., maxval=2.,
                         shape=bijector_batch_shape, seed=seed())),
                 tfb.Reshape([2, 1])]
    bijector = tfb.JointMap(tf.nest.pack_sequence_as(split_sizes, bijectors),
                            validate_args=True)

    # Transform a joint distribution that has different batch shape components
    transformed_dist = tfd.TransformedDistribution(base_dist, bijector)

    self.assertRegex(
        str(transformed_dist),
        '{}.*batch_shape.*event_shape.*dtype'.format(transformed_dist.name))

    self.assertAllEqualNested(
        transformed_dist.event_shape,
        bijector.forward_event_shape(base_dist.event_shape))
    self.assertAllEqualNested(*self.evaluate((
        transformed_dist.event_shape_tensor(),
        bijector.forward_event_shape_tensor(base_dist.event_shape_tensor()))))

    # Test that the batch shape components of the input are the same as those of
    # the output.
    self.assertAllEqualNested(transformed_dist.batch_shape, dist_batch_shape)
    self.assertAllEqualNested(
        self.evaluate(transformed_dist.batch_shape_tensor()), dist_batch_shape)
    self.assertAllEqualNested(dist_batch_shape, base_dist.batch_shape)

  @parameterized.named_parameters(
      {'testcase_name': 'sequential',
       'split_sizes': [1, 3, 2]},
      {'testcase_name': 'named',
       'split_sizes': {'a': 1, 'b': 3, 'c': 2}},)
  @test_util.numpy_disable_test_missing_functionality('vectorized_map')
  def test_transform_autobatched_joint_to_joint(self, split_sizes):
    dist_batch_shape = tf.nest.pack_sequence_as(
        split_sizes,
        [tensorshape_util.constant_value_as_shape(s)
         for s in [[2, 3], [2, 1], [1, 3]]])
    bijector_batch_shape = [1, 3]

    # Build a joint distribution with parts of the specified sizes.
    seed = test_util.test_seed_stream()
    component_dists = tf.nest.map_structure(
        lambda size, batch_shape: tfd.MultivariateNormalDiag(  # pylint: disable=g-long-lambda
            loc=tf.random.normal(batch_shape + [size], seed=seed()),
            scale_diag=tf.random.uniform(
                minval=1., maxval=2.,
                shape=batch_shape + [size], seed=seed())),
        split_sizes, dist_batch_shape)
    if isinstance(split_sizes, dict):
      base_dist = tfd.JointDistributionNamedAutoBatched(
          component_dists, batch_ndims=2)
    else:
      base_dist = tfd.JointDistributionSequentialAutoBatched(
          component_dists, batch_ndims=2)

    # Transform the distribution by applying a separate bijector to each part.
    bijectors = [tfb.Exp(),
                 tfb.Scale(
                     tf.random.uniform(
                         minval=1., maxval=2.,
                         shape=bijector_batch_shape, seed=seed())),
                 tfb.Reshape([2, 1])]
    bijector = tfb.JointMap(tf.nest.pack_sequence_as(split_sizes, bijectors),
                            validate_args=True)

    transformed_dist = tfd.TransformedDistribution(base_dist, bijector)

    self.assertRegex(
        str(transformed_dist),
        '{}.*batch_shape.*event_shape.*dtype'.format(transformed_dist.name))

    self.assertAllEqualNested(
        transformed_dist.event_shape,
        bijector.forward_event_shape(base_dist.event_shape))
    self.assertAllEqualNested(*self.evaluate((
        transformed_dist.event_shape_tensor(),
        bijector.forward_event_shape_tensor(base_dist.event_shape_tensor()))))

    self.assertAllEqualNested(transformed_dist.batch_shape, [2, 3])
    self.assertAllEqualNested(
        self.evaluate(transformed_dist.batch_shape_tensor()), [2, 3])

    # Check transformed `log_prob` against the base distribution.
    sample_shape = [3]
    sample = base_dist.sample(sample_shape, seed=seed())
    x = tf.nest.map_structure(tf.zeros_like, sample)
    y = bijector.forward(x)
    base_logprob = base_dist.log_prob(x)
    event_ndims = tf.nest.map_structure(lambda s: s.ndims,
                                        transformed_dist.event_shape)
    ildj = bijector.inverse_log_det_jacobian(y, event_ndims=event_ndims)

    (transformed_logprob,
     base_logprob_plus_ildj,
     log_transformed_prob
    ) = self.evaluate([
        transformed_dist.log_prob(y),
        base_logprob + ildj,
        tf.math.log(transformed_dist.prob(y))
    ])
    self.assertAllClose(base_logprob_plus_ildj, transformed_logprob)
    self.assertAllClose(transformed_logprob, log_transformed_prob)

    # Test that `.sample()` works and returns a result of the expected structure
    # and shape.
    y_sampled = transformed_dist.sample(sample_shape, seed=seed())
    self.assertAllEqual(tf.nest.map_structure(lambda y: y.shape, y),
                        tf.nest.map_structure(lambda y: y.shape, y_sampled))

    # Test that a `Restructure` bijector applied to a `JointDistribution` works
    # as expected.
    num_components = len(split_sizes)
    input_keys = (split_sizes.keys() if isinstance(split_sizes, dict)
                  else range(num_components))
    output_keys = [str(i) for i in range(num_components)]
    output_structure = {k: v for k, v in zip(output_keys, input_keys)}
    restructure = tfb.Restructure(output_structure)
    restructured_dist = tfd.TransformedDistribution(
        base_dist, bijector=restructure, validate_args=True)

    # Check that attributes of the restructured distribution have the same
    # nested structure as the `output_structure` of the bijector. Pass a no-op
    # as the `assert_fn` since the contents of the structures are not
    # required to be the same.
    noop_assert_fn = lambda *_: None
    self.assertAllAssertsNested(
        noop_assert_fn, restructured_dist.event_shape, output_structure)
    self.assertAllAssertsNested(
        noop_assert_fn, restructured_dist.batch_shape, output_structure)
    self.assertAllAssertsNested(
        noop_assert_fn,
        self.evaluate(restructured_dist.event_shape_tensor()),
        output_structure)
    self.assertAllAssertsNested(
        noop_assert_fn,
        self.evaluate(restructured_dist.batch_shape_tensor()),
        output_structure)
    self.assertAllAssertsNested(
        noop_assert_fn,
        self.evaluate(restructured_dist.sample(seed=test_util.test_seed())))


if __name__ == '__main__':
  test_util.main()
