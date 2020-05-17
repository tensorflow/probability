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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
from absl.testing import parameterized

import numpy as np
from scipy import stats
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.internal import hypothesis_testlib as tfp_hps
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.internal import test_util
from tensorflow.python.util import nest  # pylint: disable=g-direct-tensorflow-import


class DummyMatrixTransform(tfb.Bijector):
  """Tractable matrix transformation.

  This is a non-sensical bijector that has forward/inverse_min_event_ndims=2.
  The main use is to check that transformed distribution calculations are done
  appropriately.
  """

  def __init__(self):
    super(DummyMatrixTransform, self).__init__(
        forward_min_event_ndims=2,
        is_constant_jacobian=False,
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
    with tf.name_scope(name) as name:
      self._loc = tf.convert_to_tensor(loc, name='loc')
      super(_ChooseLocation, self).__init__(
          is_constant_jacobian=True,
          validate_args=False,
          forward_min_event_ndims=0,
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

  def _cls(self):
    return tfd.TransformedDistribution

  def _make_unimplemented(self, name):
    def _unimplemented(self, *args):  # pylint: disable=unused-argument
      raise NotImplementedError('{} not implemented'.format(name))
    return _unimplemented

  def testTransformedDistribution(self):
    mu = 3.0
    sigma = 2.0
    # Note: the Jacobian callable only works for this example; more generally
    # you may or may not need a reduce_sum.
    log_normal = self._cls()(
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
    abs_normal = self._cls()(
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
    logit_normal = self._cls()(
        distribution=tfd.Normal(loc=0., scale=1.),
        bijector=tfb.Sigmoid(),
        validate_args=True)
    grid = [0., 0.25, 0.5, 0.75, 1.]
    q = logit_normal.quantile(grid)
    cdf = logit_normal.cdf(q)
    cdf_ = self.evaluate(cdf)
    self.assertAllClose(grid, cdf_, rtol=1e-6, atol=0.)

  def testCdfDescending(self):
    td = self._cls()(
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
      td = self._cls()(
          distribution=tfd.Normal(loc=0., scale=tf.ones([2, 2, 2])),
          bijector=chain,
          validate_args=True)
      nd = tfd.Normal(loc=1., scale=3., validate_args=True)
      self.assertAllEqual(tf.ones(td.batch_shape, dtype=tf.bool),
                          td.cdf(nd.quantile(.4)) < td.cdf(nd.quantile(.6)),
                          msg=chain.name)

  def testSfDescending(self):
    td = self._cls()(
        distribution=tfd.Normal(loc=0., scale=[1., 1.]),
        bijector=tfb.Shift(shift=1.)(tfb.Scale(scale=[2., -2.])),
        validate_args=True)
    nd = tfd.Normal(loc=1., scale=2., validate_args=True)
    self.assertAllEqual(tf.ones(td.batch_shape, dtype=tf.bool),
                        td.survival_function(nd.quantile(.8)) >
                        td.survival_function(nd.quantile(.9)))

  def testQuantileDescending(self):
    td = self._cls()(
        distribution=tfd.Normal(loc=0., scale=[1., 1.]),
        bijector=tfb.Shift(shift=1.)(tfb.Scale(scale=[2., -2.])),
        validate_args=True)
    self.assertAllEqual(tf.ones(td.batch_shape, dtype=tf.bool),
                        td.quantile(.8) < td.quantile(.9))

  def testCachedSamples(self):
    class ExpForwardOnly(tfb.Bijector):

      def __init__(self):
        super(ExpForwardOnly, self).__init__(forward_min_event_ndims=0)

      def _forward(self, x):
        return tf.exp(x)

      def _forward_log_det_jacobian(self, x):
        return tf.convert_to_tensor(x)

    exp_forward_only = ExpForwardOnly()

    mu = 3.0
    sigma = 0.02
    log_normal = self._cls()(
        distribution=tfd.Normal(loc=mu, scale=sigma),
        bijector=exp_forward_only,
        validate_args=True)

    sample = log_normal.sample([2, 3], seed=test_util.test_seed())
    sample_val, log_pdf_val = self.evaluate(
        [sample, log_normal.log_prob(sample)])
    expected_log_pdf = stats.lognorm.logpdf(
        sample_val, s=sigma, scale=np.exp(mu))
    self.assertAllClose(expected_log_pdf, log_pdf_val, rtol=1e-4, atol=0.)

  def testCachedSamplesInvert(self):
    class ExpInverseOnly(tfb.Bijector):

      def __init__(self):
        super(ExpInverseOnly, self).__init__(inverse_min_event_ndims=0)

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
    exp_normal = self._cls()(
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
    standard_normal = tfd.Normal(loc=0., scale=1.)
    multi_logit_normal = self._cls()(
        distribution=standard_normal,
        bijector=softmax,
        event_shape=[1],
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
    normal = self._cls()(
        distribution=tfd.Normal(loc=0., scale=1.),
        bijector=int_identity,
        validate_args=True)

    y = normal.sample(seed=test_util.test_seed())
    self.evaluate(normal.log_prob(y))
    self.evaluate(normal.prob(y))
    self.evaluate(normal.mean())
    self.evaluate(normal.entropy())

  def testMode(self):
    dist = self._cls()(
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
    fake_mvn = self._cls()(
        tfd.MultivariateNormalDiag(
            loc=tf.zeros_like(shift),
            scale_diag=tf.ones_like(diag),
            validate_args=True),
        tfb.AffineLinearOperator(
            shift,
            scale=tf.linalg.LinearOperatorDiag(diag, is_non_singular=True),
            validate_args=True),
        validate_args=True)
    self.assertAllClose(shift, self.evaluate(fake_mvn.mean()))

  def testMeanShapeOverride(self):
    shift = np.array([[-1, 0, 1], [-1, -2, -3]], dtype=np.float32)
    diag = np.array([[1, 2, 3], [2, 3, 2]], dtype=np.float32)
    fake_mvn = self._cls()(
        tfd.Normal(loc=0.0, scale=1.0),
        tfb.AffineLinearOperator(
            shift,
            scale=tf.linalg.LinearOperatorDiag(diag, is_non_singular=True),
            validate_args=True),
        batch_shape=[2],
        event_shape=[3],
        validate_args=True)
    self.assertAllClose(shift, self.evaluate(fake_mvn.mean()))

  def testEntropy(self):
    shift = np.array([[-1, 0, 1], [-1, -2, -3]], dtype=np.float32)
    diag = np.array([[1, 2, 3], [2, 3, 2]], dtype=np.float32)
    actual_mvn_entropy = np.concatenate(
        [[stats.multivariate_normal(shift[i], np.diag(diag[i]**2)).entropy()]
         for i in range(len(diag))])
    fake_mvn = self._cls()(
        tfd.MultivariateNormalDiag(
            loc=tf.zeros_like(shift),
            scale_diag=tf.ones_like(diag),
            validate_args=True),
        tfb.AffineLinearOperator(
            shift,
            scale=tf.linalg.LinearOperatorDiag(diag, is_non_singular=True),
            validate_args=True),
        validate_args=True)
    self.assertAllClose(actual_mvn_entropy, self.evaluate(fake_mvn.entropy()))

  def testScalarBatchScalarEventIdentityScale(self):
    exp2 = self._cls()(
        tfd.Exponential(rate=0.25),
        bijector=tfb.Scale(scale=2.),
        validate_args=True)
    log_prob = exp2.log_prob(1.)
    log_prob_ = self.evaluate(log_prob)
    base_log_prob = -0.5 * 0.25 + np.log(0.25)
    ildj = np.log(2.)
    self.assertAllClose(base_log_prob - ildj, log_prob_, rtol=1e-6, atol=0.)


@test_util.test_all_tf_execution_regimes
class ScalarToMultiTest(test_util.TestCase):

  def _cls(self):
    return tfd.TransformedDistribution

  def setUp(self):
    self._shift = np.array([-1, 0, 1], dtype=np.float32)
    self._tril = np.array([[[1., 0, 0],
                            [2, 1, 0],
                            [3, 2, 1]],
                           [[2, 0, 0],
                            [3, 2, 0],
                            [4, 3, 2]]],
                          dtype=np.float32)
    super(ScalarToMultiTest, self).setUp()

  def _testMVN(self,
               base_distribution_class,
               base_distribution_kwargs,
               batch_shape=(),
               event_shape=(),
               not_implemented_message=None):
    # Overriding shapes must be compatible w/bijector; most bijectors are
    # batch_shape agnostic and only care about event_ndims.
    # In the case of `Affine`, if we got it wrong then it would fire an
    # exception due to incompatible dimensions.
    batch_shape_var = tf.Variable(
        np.int32(batch_shape),
        shape=tf.TensorShape(None),
        name='dynamic_batch_shape')
    event_shape_var = tf.Variable(
        np.int32(event_shape),
        shape=tf.TensorShape(None),
        name='dynamic_event_shape')

    fake_mvn_dynamic = self._cls()(
        distribution=base_distribution_class(
            validate_args=True, **base_distribution_kwargs),
        bijector=tfb.Affine(shift=self._shift, scale_tril=self._tril),
        batch_shape=batch_shape_var,
        event_shape=event_shape_var,
        validate_args=True)

    fake_mvn_static = self._cls()(
        distribution=base_distribution_class(
            validate_args=True, **base_distribution_kwargs),
        bijector=tfb.Affine(shift=self._shift, scale_tril=self._tril),
        batch_shape=batch_shape,
        event_shape=event_shape,
        validate_args=True)

    actual_mean = np.tile(self._shift, [2, 1])  # Affine elided this tile.
    actual_cov = np.matmul(self._tril, np.transpose(self._tril, [0, 2, 1]))

    def actual_mvn_log_prob(x):
      return np.concatenate([[  # pylint: disable=g-complex-comprehension
          stats.multivariate_normal(actual_mean[i],
                                    actual_cov[i]).logpdf(x[:, i, :])
      ] for i in range(len(actual_cov))]).T

    actual_mvn_entropy = np.concatenate(
        [[stats.multivariate_normal(actual_mean[i], actual_cov[i]).entropy()]
         for i in range(len(actual_cov))])

    self.assertAllEqual([3], fake_mvn_static.event_shape)
    self.assertAllEqual([2], fake_mvn_static.batch_shape)

    if not tf.executing_eagerly():
      self.assertAllEqual(tf.TensorShape(None), fake_mvn_dynamic.event_shape)
      self.assertAllEqual(tf.TensorShape(None), fake_mvn_dynamic.batch_shape)

    x = self.evaluate(fake_mvn_static.sample(5, seed=test_util.test_seed()))
    for unsupported_fn in (fake_mvn_static.log_cdf, fake_mvn_static.cdf,
                           fake_mvn_static.survival_function,
                           fake_mvn_static.log_survival_function):
      with self.assertRaisesRegexp(NotImplementedError,
                                   not_implemented_message):
        unsupported_fn(x)

    num_samples = 7e3
    for fake_mvn in [fake_mvn_static, fake_mvn_dynamic]:
      # Ensure sample works by checking first, second moments.
      y = fake_mvn.sample(int(num_samples), seed=test_util.test_seed())
      x = y[0:5, ...]
      sample_mean = tf.reduce_mean(y, axis=0)
      centered_y = tf.transpose(a=y - sample_mean, perm=[1, 2, 0])
      sample_cov = tf.matmul(
          centered_y, centered_y, transpose_b=True) / num_samples
      self.evaluate([batch_shape_var.initializer, event_shape_var.initializer])
      [
          sample_mean_,
          sample_cov_,
          x_,
          fake_event_shape_,
          fake_batch_shape_,
          fake_log_prob_,
          fake_prob_,
          fake_mean_,
          fake_entropy_,
      ] = self.evaluate([
          sample_mean,
          sample_cov,
          x,
          fake_mvn.event_shape_tensor(),
          fake_mvn.batch_shape_tensor(),
          fake_mvn.log_prob(x),
          fake_mvn.prob(x),
          fake_mvn.mean(),
          fake_mvn.entropy(),
      ])

      self.assertAllClose(actual_mean, sample_mean_, atol=0.1, rtol=0.1)
      self.assertAllClose(actual_cov, sample_cov_, atol=0., rtol=0.1)

      # Ensure all other functions work as intended.
      self.assertAllEqual([5, 2, 3], x_.shape)
      self.assertAllEqual([3], fake_event_shape_)
      self.assertAllEqual([2], fake_batch_shape_)
      self.assertAllClose(
          actual_mvn_log_prob(x_), fake_log_prob_, atol=0., rtol=1e-6)
      self.assertAllClose(
          np.exp(actual_mvn_log_prob(x_)), fake_prob_, atol=0., rtol=1e-5)
      self.assertAllClose(actual_mean, fake_mean_, atol=0., rtol=1e-6)
      self.assertAllClose(actual_mvn_entropy, fake_entropy_, atol=0., rtol=1e-6)

  def testScalarBatchScalarEvent(self):
    self._testMVN(
        base_distribution_class=tfd.Normal,
        base_distribution_kwargs={
            'loc': 0.,
            'scale': 1.
        },
        batch_shape=[2],
        event_shape=[3],
        not_implemented_message='not implemented when overriding `event_shape`')

  def testScalarBatchNonScalarEvent(self):
    self._testMVN(
        base_distribution_class=tfd.MultivariateNormalDiag,
        base_distribution_kwargs={
            'loc': [0., 0., 0.],
            'scale_diag': [1., 1, 1]
        },
        batch_shape=[2],
        not_implemented_message='not implemented')

    # Can't override event_shape for scalar batch, non-scalar event.
    with self.assertRaisesWithPredicateMatch(
        Exception, 'Base distribution is not scalar.'):

      self._cls()(
          distribution=tfd.MultivariateNormalDiag(loc=[0.], scale_diag=[1.]),
          bijector=tfb.Affine(shift=self._shift, scale_tril=self._tril),
          batch_shape=[2],
          event_shape=[3],
          validate_args=True)

  def testNonScalarBatchScalarEvent(self):

    self._testMVN(
        base_distribution_class=tfd.Normal,
        base_distribution_kwargs={
            'loc': [0., 0],
            'scale': [1., 1]
        },
        event_shape=[3],
        not_implemented_message='not implemented when overriding'
        ' `event_shape`')

    # Can't override batch_shape for non-scalar batch, scalar event.
    with self.assertRaisesWithPredicateMatch(
        Exception, 'Base distribution is not scalar.'):
      self._cls()(
          distribution=tfd.Normal(loc=[0.], scale=[1.]),
          bijector=tfb.Affine(shift=self._shift, scale_tril=self._tril),
          batch_shape=[2],
          event_shape=[3],
          validate_args=True)

  def testNonScalarBatchNonScalarEvent(self):
    # Can't override event_shape and/or batch_shape for non_scalar batch,
    # non-scalar event.
    with self.assertRaisesRegexp(ValueError, 'Base distribution is not scalar'):
      self._cls()(
          distribution=tfd.MultivariateNormalDiag(
              loc=[[0.]], scale_diag=[[1.]]),
          bijector=tfb.Affine(shift=self._shift, scale_tril=self._tril),
          batch_shape=[2],
          event_shape=[3],
          validate_args=True)

  def testMatrixEvent(self):
    batch_shape = [2]
    event_shape = [2, 3, 3]
    batch_shape_var = tf.Variable(
        np.int32(batch_shape),
        shape=tf.TensorShape(None),
        name='dynamic_batch_shape')
    event_shape_var = tf.Variable(
        np.int32(event_shape),
        shape=tf.TensorShape(None),
        name='dynamic_event_shape')

    scale = 2.
    loc = 0.
    fake_mvn_dynamic = self._cls()(
        distribution=tfd.Normal(loc=loc, scale=scale),
        bijector=DummyMatrixTransform(),
        batch_shape=batch_shape_var,
        event_shape=event_shape_var,
        validate_args=True)

    fake_mvn_static = self._cls()(
        distribution=tfd.Normal(loc=loc, scale=scale),
        bijector=DummyMatrixTransform(),
        batch_shape=batch_shape,
        event_shape=event_shape,
        validate_args=True)

    def actual_mvn_log_prob(x):
      # This distribution is the normal PDF, reduced over the
      # last 3 dimensions + a jacobian term which corresponds
      # to the determinant of x.
      return (np.sum(stats.norm(loc, scale).logpdf(x), axis=(-1, -2, -3)) +
              np.sum(np.linalg.det(x), axis=-1))

    self.assertAllEqual([2, 3, 3], fake_mvn_static.event_shape)
    self.assertAllEqual([2], fake_mvn_static.batch_shape)

    if not tf.executing_eagerly():
      self.assertAllEqual(tf.TensorShape(None), fake_mvn_dynamic.event_shape)
      self.assertAllEqual(tf.TensorShape(None), fake_mvn_dynamic.batch_shape)

    num_samples = 5e3
    self.evaluate([event_shape_var.initializer, batch_shape_var.initializer])
    for fake_mvn in [fake_mvn_static, fake_mvn_dynamic]:
      # Ensure sample works by checking first, second moments.
      y = fake_mvn.sample(int(num_samples), seed=test_util.test_seed())
      x = y[0:5, ...]
      [
          x_,
          fake_event_shape_,
          fake_batch_shape_,
          fake_log_prob_,
          fake_prob_,
      ] = self.evaluate([
          x,
          fake_mvn.event_shape_tensor(),
          fake_mvn.batch_shape_tensor(),
          fake_mvn.log_prob(x),
          fake_mvn.prob(x),
      ])

      # Ensure all other functions work as intended.
      self.assertAllEqual([5, 2, 2, 3, 3], x_.shape)
      self.assertAllEqual([2, 3, 3], fake_event_shape_)
      self.assertAllEqual([2], fake_batch_shape_)
      self.assertAllClose(
          actual_mvn_log_prob(x_), fake_log_prob_, atol=0., rtol=1e-6)
      # With this many dimensions and samples, the direct space probability
      # may underflow.
      self.assertAllClose(
          np.exp(actual_mvn_log_prob(x_)), fake_prob_, atol=1e-12, rtol=1e-5)

  def testEmptyEvent(self):
    # Verify that zero-dimensional multivariate Normal distributions still
    # return reasonable shapes and a log-prob of 0.0.

    event_shape = [0]
    for batch_shape in ([2], []):
      for shapes_are_dynamic in (True, False):
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

  @test_util.jax_disable_test_missing_functionality(
      'JAX only has static shapes.')
  def testVectorDynamicShapeOverrideWithMutation(self):
    batch_shape = tf.Variable([4], shape=tf.TensorShape(None), dtype=tf.int32)
    d = tfd.TransformedDistribution(
        distribution=tfd.Normal(loc=2., scale=1.),
        bijector=tfb.Exp(),
        batch_shape=batch_shape,
        validate_args=True)
    self.evaluate(batch_shape.initializer)
    self.evaluate(d.sample(seed=test_util.test_seed()))
    with tf.control_dependencies(
        [batch_shape.assign([[4, 2]])]):
      with self.assertRaisesOpError('must be a vector'):
        self.evaluate(d.sample(seed=test_util.test_seed()))

  def testNonNegativeDynamicShapeOverrideWithMutation(self):
    batch_shape = tf.Variable([4], shape=tf.TensorShape(None), dtype=tf.int32)
    d = tfd.TransformedDistribution(
        distribution=tfd.Normal(loc=-1., scale=1.),
        bijector=tfb.Exp(),
        batch_shape=batch_shape,
        validate_args=True)
    self.evaluate(batch_shape.initializer)
    self.evaluate(d.sample(seed=test_util.test_seed()))
    with tf.control_dependencies([batch_shape.assign([-4])]):
      with self.assertRaisesOpError('must have non-negative elements'):
        self.evaluate(d.sample(seed=test_util.test_seed()))

  @test_util.jax_disable_test_missing_functionality(
      'JAX only has static shapes.')
  def testNonScalarDynamicShapeOverrideWithMutation(self):
    loc = tf.Variable(3., shape=tf.TensorShape(None))
    base_dist = tfd.Normal(loc=loc, scale=1.)
    d = tfd.TransformedDistribution(
        distribution=base_dist,
        bijector=tfb.Exp(),
        batch_shape=tf.convert_to_tensor([3], dtype=tf.int32),
        validate_args=True)
    self.evaluate(loc.initializer)
    self.evaluate(d.sample(seed=test_util.test_seed()))
    with tf.control_dependencies([loc.assign([4., 2.])]):
      with self.assertRaisesWithPredicateMatch(
          Exception, 'Base distribution is not scalar'):
        self.evaluate(d.sample(seed=test_util.test_seed()))

  def testSupportBijectorOutsideRange(self):
    log_normal = tfd.TransformedDistribution(
        distribution=tfd.Normal(loc=1., scale=2.),
        bijector=tfb.Exp(),
        validate_args=True)
    x = np.array([-4.2, -1e-6, -1.3])
    bijector_inverse_x = (
        log_normal._experimental_default_event_space_bijector().inverse(x))
    self.assertAllNan(self.evaluate(bijector_inverse_x))


@test_util.test_all_tf_execution_regimes
class ExcessiveConcretizationTest(test_util.TestCase):

  def setUp(self):
    super(ExcessiveConcretizationTest, self).setUp()

    self.max_permissible = {

        # extra concretizations primarily of base distribution parameters
        'mean': 4,
        'sample': 3,
        'log_cdf': 4,
        'cdf': 4,
        'survival_function': 4,
        'log_survival_function': 4,

        # extra concretizations primarily of bijector parameters
        'entropy': 6,
        'log_prob': 7,
        'prob': 7,
        'quantile': 4,

        'event_shape_tensor': 2,
        'batch_shape_tensor': 2,
    }

    self.shape = None

  def testExcessiveConcretizationOfParams(self):
    loc = tfp_hps.defer_and_count_usage(
        tf.Variable(0., name='loc', dtype=tf.float32, shape=self.shape))
    scale = tfp_hps.defer_and_count_usage(
        tf.Variable(2., name='scale', dtype=tf.float32, shape=self.shape))
    bij_scale = tfp_hps.defer_and_count_usage(
        tf.Variable(2., name='bij_scale', dtype=tf.float32, shape=self.shape))
    event_shape = tfp_hps.defer_and_count_usage(
        tf.Variable([2, 2], name='input_event_shape', dtype=tf.int32,
                    shape=self.shape))
    batch_shape = tfp_hps.defer_and_count_usage(
        tf.Variable([4, 3, 5], name='input_batch_shape', dtype=tf.int32,
                    shape=self.shape))

    dist = tfd.TransformedDistribution(
        distribution=tfd.Normal(loc=loc, scale=scale, validate_args=True),
        bijector=tfb.Scale(scale=bij_scale, validate_args=True),
        event_shape=event_shape,
        batch_shape=batch_shape,
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

  def testExcessiveConcretizationOfParamsBatchShapeOverride(self):
    # Test methods that are not implemented if event_shape is overriden.
    loc = tfp_hps.defer_and_count_usage(
        tf.Variable(0., name='loc', dtype=tf.float32, shape=self.shape))
    scale = tfp_hps.defer_and_count_usage(
        tf.Variable(2., name='scale', dtype=tf.float32, shape=self.shape))
    bij_scale = tfp_hps.defer_and_count_usage(
        tf.Variable(2., name='bij_scale', dtype=tf.float32, shape=self.shape))
    batch_shape = tfp_hps.defer_and_count_usage(
        tf.Variable([4, 3, 5], name='input_batch_shape', dtype=tf.int32,
                    shape=self.shape))
    dist = tfd.TransformedDistribution(
        distribution=tfd.Normal(loc=loc, scale=scale, validate_args=True),
        bijector=tfb.Scale(scale=bij_scale, validate_args=True),
        batch_shape=batch_shape,
        validate_args=True)

    for method in (
        'log_cdf', 'cdf', 'survival_function', 'log_survival_function'):
      with tfp_hps.assert_no_excessive_var_usage(
          method, max_permissible=self.max_permissible[method]):
        getattr(dist, method)(np.ones((4, 3, 2)) / 3.)

    with tfp_hps.assert_no_excessive_var_usage(
        'quantile', max_permissible=self.max_permissible['quantile']):
      dist.quantile(.1)


@test_util.test_all_tf_execution_regimes
class ExcessiveConcretizationTestUnknownShape(ExcessiveConcretizationTest):

  def setUp(self):
    super(ExcessiveConcretizationTestUnknownShape, self).setUp()

    self.max_permissible = {

        # extra concretizations primarily of base distribution parameters
        'mean': 9,
        'sample': 9,
        'log_cdf': 7,
        'cdf': 7,
        'survival_function': 7,
        'log_survival_function': 7,
        'entropy': 9,
        'quantile': 5,
        'event_shape_tensor': 5,
        'batch_shape_tensor': 6,
        'log_prob': 10,
        'prob': 13,
    }

    self.shape = tf.TensorShape(None)


# TODO(emilyaf): Check in `Split` bijector and remove this.
class ToySplit(tfb.Bijector):

  def __init__(self, size_splits):
    self._size_splits = size_splits
    self._flat_size_splits = tf.nest.flatten(size_splits)
    super(ToySplit, self).__init__(
        forward_min_event_ndims=1,
        inverse_min_event_ndims=0,
        is_constant_jacobian=True)

  @property
  def size_splits(self):
    return self._size_splits

  def forward(self, x):
    return tf.nest.pack_sequence_as(
        self.size_splits, tf.split(x, self._flat_size_splits, axis=-1))

  def inverse(self, y):
    return tf.concat(tf.nest.flatten(y), axis=-1)

  def forward_dtype(self, dtype):
    return tf.nest.map_structure(lambda _: dtype, self.size_splits)

  def inverse_dtype(self, dtype):
    flat_dtype = tf.nest.flatten(dtype)
    if any(d != flat_dtype[0] for d in flat_dtype):
      raise ValueError('All dtypes must be equivalent.')
    return flat_dtype[0]

  def forward_event_shape(self, x):
    return tf.nest.map_structure(lambda k: tf.TensorShape([k]),
                                 self.size_splits)

  def inverse_event_shape(self, y):
    return tf.TensorShape(sum(self._flat_size_splits))

  def forward_event_shape_tensor(self, x):
    return tf.nest.map_structure(lambda k: tf.convert_to_tensor([k]),
                                 self.size_splits)

  def inverse_event_shape_tensor(self, y):
    return tf.reduce_sum(self._flat_size_splits)[..., tf.newaxis]

  def forward_log_det_jacobian(self, x, event_ndims):
    return tf.constant(0., dtype=tf.float32)

  def inverse_log_det_jacobian(self, y, event_ndims):
    return tf.constant(0., dtype=tf.float32)


# TODO(emilyaf): Check in `ZipMap` bijector and remove this.
class ToyZipMap(tfb.Bijector):

  def __init__(self, bijectors):
    self._bijectors = bijectors
    super(ToyZipMap, self).__init__(
        forward_min_event_ndims=0,
        inverse_min_event_ndims=0,
        is_constant_jacobian=all([
            b.is_constant_jacobian for b in tf.nest.flatten(bijectors)]))

  @property
  def bijectors(self):
    return self._bijectors

  def forward(self, x):
    return tf.nest.map_structure(lambda b_i, x_i: b_i.forward(x_i),
                                 self.bijectors, x)

  def inverse(self, y):
    return tf.nest.map_structure(lambda b_i, y_i: b_i.inverse(y_i),
                                 self.bijectors, y)

  def forward_dtype(self, dtype):
    return tf.nest.map_structure(lambda b_i, d_i: b_i.forward_dtype(d_i),
                                 self.bijectors, dtype)

  def inverse_dtype(self, dtype):
    return tf.nest.map_structure(lambda b_i, d_i: b_i.inverse_dtype(d_i),
                                 self.bijectors, dtype)

  def forward_event_shape(self, x_shape):
    return tf.nest.map_structure(
        lambda b_i, x_i: b_i.forward_event_shape(x_i),
        self.bijectors, x_shape)

  def inverse_event_shape(self, y_shape):
    return tf.nest.map_structure(
        lambda b_i, y_i: b_i.inverse_event_shape(y_i),
        self.bijectors, y_shape)

  def forward_event_shape_tensor(self, x_shape_tensor):
    return tf.nest.map_structure(
        lambda b_i, x_i: b_i.forward_event_shape_tensor(x_i),
        self.bijectors, x_shape_tensor)

  def inverse_event_shape_tensor(self, y_shape_tensor):
    return tf.nest.map_structure(
        lambda b_i, y_i: b_i.inverse_event_shape_tensor(y_i),
        self.bijectors, y_shape_tensor)

  def _forward_log_det_jacobian(self, x, event_ndims):
    fldj_parts = tf.nest.map_structure(
        lambda b, y, n: b.forward_log_det_jacobian(x, event_ndims=n),
        self.bijectors, x, event_ndims)
    return sum(tf.nest.flatten(fldj_parts))

  def inverse_log_det_jacobian(self, y, event_ndims):
    ildj_parts = tf.nest.map_structure(
        lambda b, y, n: b.inverse_log_det_jacobian(y, event_ndims=n),
        self.bijectors, y, event_ndims)
    return sum(tf.nest.flatten(ildj_parts))


@test_util.test_all_tf_execution_regimes
class JointBijectorsTest(test_util.TestCase):

  @parameterized.named_parameters(
      {'testcase_name': 'list',
       'split_sizes': [1, 3, 2]},
      {'testcase_name': 'dict',
       'split_sizes': {'a': 1, 'b': 3, 'c': 2}})
  def test_transform_parts_to_vector(self, split_sizes):
    batch_shape = [4, 2]

    # Create a joint distribution with parts of the specified sizes.
    seed = test_util.test_seed_stream()
    component_dists = tf.nest.map_structure(
        lambda size: tfd.MultivariateNormalDiag(  # pylint: disable=g-long-lambda
            loc=tf.random.normal(batch_shape + [size], seed=seed()),
            scale_diag=tf.exp(
                tf.random.normal(batch_shape + [size], seed=seed()))),
        split_sizes)
    if isinstance(split_sizes, dict):
      base_dist = tfd.JointDistributionNamed(component_dists)
    else:
      base_dist = tfd.JointDistributionSequential(component_dists)

    # Transform to a vector-valued distribution by concatenating the parts.
    bijector = tfb.Invert(ToySplit(split_sizes))

    with self.assertRaisesRegexp(ValueError, 'Overriding the batch shape'):
      tfd.TransformedDistribution(base_dist, bijector, batch_shape=[3])

    with self.assertRaisesRegexp(ValueError, 'Overriding the event shape'):
      tfd.TransformedDistribution(base_dist, bijector, event_shape=[3])

    concat_dist = tfd.TransformedDistribution(base_dist, bijector)

    concat_event_size = self.evaluate(
        tf.reduce_sum(tf.nest.flatten(split_sizes)))
    self.assertAllEqual(concat_dist.event_shape, [concat_event_size])
    self.assertAllEqual(self.evaluate(concat_dist.event_shape_tensor()),
                        [concat_event_size])
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
      {'testcase_name': 'list',
       'split_sizes': [1, 3, 2]},
      {'testcase_name': 'dict',
       'split_sizes': {'a': 1, 'bc': {'b': 3, 'c': 2}}},)
  def test_transform_vector_to_parts(self, split_sizes):
    batch_shape = [4, 2]
    base_event_size = tf.reduce_sum(tf.nest.flatten(split_sizes))
    base_dist = tfd.MultivariateNormalDiag(
        loc=tf.random.normal(
            batch_shape + [base_event_size], seed=test_util.test_seed()),
        scale_diag=tf.exp(tf.random.normal(
            batch_shape + [base_event_size], seed=test_util.test_seed())))

    bijector = ToySplit(split_sizes)
    split_dist = tfd.TransformedDistribution(base_dist, bijector)

    expected_event_shape = tf.nest.map_structure(
        lambda s: np.array([s]), split_sizes)
    output_event_shape = nest.map_structure_up_to(
        split_dist.dtype, np.array, split_dist.event_shape)
    self.assertAllEqualNested(output_event_shape, expected_event_shape)
    self.assertAllEqualNested(self.evaluate(split_dist.event_shape_tensor()),
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
    self.assertAllEqualNested(
        tf.nest.map_structure(lambda x: x.shape, y),
        tf.nest.map_structure(lambda x: x.shape, y_sampled))

    # Test that `batch_shape` override works and does not affect the event shape
    base_dist = tfd.Independent(
        tfd.Normal(loc=list(range(6)), scale=1.),
        reinterpreted_batch_ndims=1, validate_args=True)
    override_batch_shape = [5, 2]
    split_dist_batch_override = tfd.TransformedDistribution(
        base_dist, bijector, batch_shape=override_batch_shape)
    self.assertAllEqualNested(
        split_dist_batch_override.event_shape, expected_event_shape)
    self.assertAllEqualNested(
        self.evaluate(split_dist_batch_override.event_shape_tensor()),
        expected_event_shape)
    self.assertAllEqual(split_dist_batch_override.batch_shape,
                        override_batch_shape)
    self.assertAllEqual(
        self.evaluate(split_dist_batch_override.batch_shape_tensor()),
        override_batch_shape)

    # Test that `event_shape` override works as expected with `Split`
    override_event_shape = [6]
    base_dist = tfd.Normal(0., [2., 1.])
    split_dist_event_override = tfd.TransformedDistribution(
        base_dist, bijector, event_shape=override_event_shape)
    self.assertAllEqualNested(
        split_dist_event_override.event_shape, expected_event_shape)
    self.assertAllEqualNested(
        self.evaluate(split_dist_event_override.event_shape_tensor()),
        expected_event_shape)
    self.assertAllEqual(
        split_dist_event_override.batch_shape, base_dist.batch_shape)
    self.assertAllEqual(
        self.evaluate(split_dist_event_override.batch_shape_tensor()),
        self.evaluate(base_dist.batch_shape_tensor()))

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
            scale_diag=tf.exp(
                tf.random.normal(batch_shape + [size], seed=seed()))),
        split_sizes, dist_batch_shape)
    if isinstance(split_sizes, dict):
      base_dist = tfd.JointDistributionNamed(component_dists)
    else:
      base_dist = tfd.JointDistributionSequential(component_dists)

    # Transform the distribution by applying a separate bijector to each part.
    bijectors = [tfb.Exp(),
                 tfb.Scale(tf.random.normal(bijector_batch_shape, seed=seed())),
                 tfb.Reshape([2, 1])]
    bijector = ToyZipMap(tf.nest.pack_sequence_as(split_sizes, bijectors))

    with self.assertRaisesRegexp(ValueError, 'Overriding the batch shape'):
      tfd.TransformedDistribution(base_dist, bijector, batch_shape=[3])

    with self.assertRaisesRegexp(ValueError, 'Overriding the event shape'):
      tfd.TransformedDistribution(base_dist, bijector, event_shape=[3])

    # Transform a joint distribution that has different batch shape components
    transformed_dist = tfd.TransformedDistribution(base_dist, bijector)

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


if __name__ == '__main__':
  tf.test.main()
