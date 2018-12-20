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
import numpy as np
from scipy import stats
import tensorflow as tf
import tensorflow_probability as tfp

tfb = tfp.bijectors
tfd = tfp.distributions
tfe = tf.contrib.eager


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
        name="dummy")

  def _forward(self, x):
    return x

  def _inverse(self, y):
    return y

  # Note: These jacobians don't make sense.
  def _forward_log_det_jacobian(self, x):
    return -tf.matrix_determinant(x)

  def _inverse_log_det_jacobian(self, x):
    return tf.matrix_determinant(x)


@tfe.run_all_tests_in_graph_and_eager_modes
class TransformedDistributionTest(tf.test.TestCase):

  def _cls(self):
    return tfd.TransformedDistribution

  def _make_unimplemented(self, name):
    def _unimplemented(self, *args):  # pylint: disable=unused-argument
      raise NotImplementedError("{} not implemented".format(name))
    return _unimplemented

  def testTransformedDistribution(self):
    mu = 3.0
    sigma = 2.0
    # Note: the Jacobian callable only works for this example; more generally
    # you may or may not need a reduce_sum.
    log_normal = self._cls()(
        distribution=tfd.Normal(loc=mu, scale=sigma), bijector=tfb.Exp())
    sp_dist = stats.lognorm(s=sigma, scale=np.exp(mu))

    # sample
    sample = log_normal.sample(100000, seed=235)
    self.assertAllEqual([], log_normal.event_shape)
    with self.cached_session():
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
      with self.cached_session():
        self.assertAllClose(
            expected, self.evaluate(actual), atol=0, rtol=0.01)

  def testNonInjectiveTransformedDistribution(self):
    mu = 1.
    sigma = 2.0
    abs_normal = self._cls()(
        distribution=tfd.Normal(loc=mu, scale=sigma),
        bijector=tfb.AbsoluteValue())
    sp_normal = stats.norm(mu, sigma)

    # sample
    sample = abs_normal.sample(100000, seed=235)
    self.assertAllEqual([], abs_normal.event_shape)
    with self.cached_session():
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
        distribution=tfd.Normal(loc=mu, scale=sigma), bijector=exp_forward_only)

    sample = log_normal.sample([2, 3], seed=42)
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
        return tf.log(y)

      def _inverse_log_det_jacobian(self, y):
        return -tf.log(y)

    exp_inverse_only = ExpInverseOnly()

    log_forward_only = tfb.Invert(exp_inverse_only)

    # The log bijector isn't defined over the whole real line, so we make
    # sigma sufficiently small so that the draws are positive.
    mu = 2.
    sigma = 1e-2
    exp_normal = self._cls()(
        distribution=tfd.Normal(loc=mu, scale=sigma), bijector=log_forward_only)

    sample = exp_normal.sample([2, 3], seed=42)
    sample_val, log_pdf_val = self.evaluate(
        [sample, exp_normal.log_prob(sample)])
    expected_log_pdf = sample_val + stats.norm.logpdf(
        np.exp(sample_val), loc=mu, scale=sigma)
    self.assertAllClose(expected_log_pdf, log_pdf_val, atol=0.)

  def testShapeChangingBijector(self):
    softmax = tfb.SoftmaxCentered()
    standard_normal = tfd.Normal(loc=0., scale=1.)
    multi_logit_normal = self._cls()(
        distribution=standard_normal, bijector=softmax, event_shape=[1])
    x = [[[-np.log(3.)], [0.]], [[np.log(3)], [np.log(5)]]]
    x = np.float32(x)
    y = self.evaluate(softmax.forward(x))
    expected_log_pdf = (
        np.squeeze(stats.norm(loc=0., scale=1.).logpdf(x)) - np.sum(
            np.log(y), axis=-1))
    self.assertAllClose(expected_log_pdf,
                        self.evaluate(multi_logit_normal.log_prob(y)))
    self.assertAllClose([1, 2, 3, 2],
                        self.evaluate(
                            tf.shape(multi_logit_normal.sample([1, 2, 3]))))
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

    y = normal.sample()
    self.evaluate(normal.log_prob(y))
    self.evaluate(normal.prob(y))
    self.evaluate(normal.entropy())

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
        tfd.Exponential(rate=0.25), bijector=tfb.AffineScalar(scale=2.))
    log_prob = exp2.log_prob(1.)
    log_prob_ = self.evaluate(log_prob)
    base_log_prob = -0.5 * 0.25 + np.log(0.25)
    ildj = np.log(2.)
    self.assertAllClose(base_log_prob - ildj, log_prob_, rtol=1e-6, atol=0.)


class ScalarToMultiTest(tf.test.TestCase):

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
    batch_shape_pl = tf.placeholder_with_default(
        input=np.int32(batch_shape), shape=None, name="dynamic_batch_shape")
    event_shape_pl = tf.placeholder_with_default(
        input=np.int32(event_shape), shape=None, name="dynamic_event_shape")
    fake_mvn_dynamic = self._cls()(
        distribution=base_distribution_class(
            validate_args=True, **base_distribution_kwargs),
        bijector=tfb.Affine(shift=self._shift, scale_tril=self._tril),
        batch_shape=batch_shape_pl,
        event_shape=event_shape_pl,
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
      return np.concatenate([[
          stats.multivariate_normal(actual_mean[i],
                                    actual_cov[i]).logpdf(x[:, i, :])
      ] for i in range(len(actual_cov))]).T

    actual_mvn_entropy = np.concatenate(
        [[stats.multivariate_normal(actual_mean[i], actual_cov[i]).entropy()]
         for i in range(len(actual_cov))])

    self.assertAllEqual([3], fake_mvn_static.event_shape)
    self.assertAllEqual([2], fake_mvn_static.batch_shape)

    self.assertAllEqual(tf.TensorShape(None), fake_mvn_dynamic.event_shape)
    self.assertAllEqual(tf.TensorShape(None), fake_mvn_dynamic.batch_shape)

    x = self.evaluate(fake_mvn_static.sample(5, seed=0))
    for unsupported_fn in (fake_mvn_static.log_cdf, fake_mvn_static.cdf,
                           fake_mvn_static.survival_function,
                           fake_mvn_static.log_survival_function):
      with self.assertRaisesRegexp(NotImplementedError,
                                   not_implemented_message):
        unsupported_fn(x)

    num_samples = 5e3
    for fake_mvn in [fake_mvn_static, fake_mvn_dynamic]:
      # Ensure sample works by checking first, second moments.
      y = fake_mvn.sample(int(num_samples), seed=0)
      x = y[0:5, ...]
      sample_mean = tf.reduce_mean(y, 0)
      centered_y = tf.transpose(y - sample_mean, [1, 2, 0])
      sample_cov = tf.matmul(
          centered_y, centered_y, transpose_b=True) / num_samples
      [
          sample_mean_,
          sample_cov_,
          x_,
          fake_event_shape_,
          fake_batch_shape_,
          fake_log_prob_,
          fake_prob_,
          fake_entropy_,
      ] = self.evaluate([
          sample_mean,
          sample_cov,
          x,
          fake_mvn.event_shape_tensor(),
          fake_mvn.batch_shape_tensor(),
          fake_mvn.log_prob(x),
          fake_mvn.prob(x),
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
      self.assertAllClose(actual_mvn_entropy, fake_entropy_, atol=0., rtol=1e-6)

  def testScalarBatchScalarEvent(self):
    self._testMVN(
        base_distribution_class=tfd.Normal,
        base_distribution_kwargs={
            "loc": 0.,
            "scale": 1.
        },
        batch_shape=[2],
        event_shape=[3],
        not_implemented_message="not implemented when overriding event_shape")

  def testScalarBatchNonScalarEvent(self):
    self._testMVN(
        base_distribution_class=tfd.MultivariateNormalDiag,
        base_distribution_kwargs={
            "loc": [0., 0., 0.],
            "scale_diag": [1., 1, 1]
        },
        batch_shape=[2],
        not_implemented_message="not implemented")

    # Can't override event_shape for scalar batch, non-scalar event.
    with self.assertRaisesRegexp(ValueError, "base distribution not scalar"):
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
            "loc": [0., 0],
            "scale": [1., 1]
        },
        event_shape=[3],
        not_implemented_message="not implemented when overriding event_shape")

    # Can't override batch_shape for non-scalar batch, scalar event.
    with self.assertRaisesRegexp(ValueError, "base distribution not scalar"):
      self._cls()(
          distribution=tfd.Normal(loc=[0.], scale=[1.]),
          bijector=tfb.Affine(shift=self._shift, scale_tril=self._tril),
          batch_shape=[2],
          event_shape=[3],
          validate_args=True)

  def testNonScalarBatchNonScalarEvent(self):
    # Can't override event_shape and/or batch_shape for non_scalar batch,
    # non-scalar event.
    with self.assertRaisesRegexp(ValueError, "base distribution not scalar"):
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
    batch_shape_pl = tf.placeholder_with_default(
        input=np.int32(batch_shape), shape=None, name="dynamic_batch_shape")
    event_shape_pl = tf.placeholder_with_default(
        input=np.int32(event_shape), shape=None, name="dynamic_event_shape")

    scale = 2.
    loc = 0.
    fake_mvn_dynamic = self._cls()(
        distribution=tfd.Normal(loc=loc, scale=scale),
        bijector=DummyMatrixTransform(),
        batch_shape=batch_shape_pl,
        event_shape=event_shape_pl,
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

    self.assertAllEqual(tf.TensorShape(None), fake_mvn_dynamic.event_shape)
    self.assertAllEqual(tf.TensorShape(None), fake_mvn_dynamic.batch_shape)

    num_samples = 5e3
    for fake_mvn in [fake_mvn_static, fake_mvn_dynamic]:
      # Ensure sample works by checking first, second moments.
      y = fake_mvn.sample(int(num_samples), seed=0)
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
      self.assertAllClose(
          np.exp(actual_mvn_log_prob(x_)), fake_prob_, atol=0., rtol=1e-5)

  def testEmptyEvent(self):
    # Verify that zero-dimensional multivariate Normal distributions still
    # return reasonable shapes and a log-prob of 0.0.
    event_shape = [0]
    for batch_shape in ([2], []):
      for shapes_are_dynamic in (True, False):
        loc = tf.zeros(batch_shape + event_shape)
        scale_diag = tf.ones(batch_shape + event_shape)
        if shapes_are_dynamic:
          loc = tf.placeholder_with_default(loc, shape=None,
                                            name="dynamic_loc")
          scale_diag = tf.placeholder_with_default(scale_diag, shape=None,
                                                   name="dynamic_scale_diag")

        mvn = tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale_diag)

        self.assertAllEqual(self.evaluate(mvn.event_shape_tensor()),
                            event_shape)
        self.assertAllEqual(self.evaluate(mvn.batch_shape_tensor()),
                            batch_shape)
        if not shapes_are_dynamic:
          self.assertAllEqual(mvn.event_shape.as_list(),
                              event_shape)
          self.assertAllEqual(mvn.batch_shape.as_list(),
                              batch_shape)

        for sample_shape in ([3], []):
          sample_ = self.evaluate(mvn.sample(sample_shape))
          self.assertAllEqual(sample_.shape,
                              sample_shape + batch_shape + event_shape)
          self.assertAllEqual(
              self.evaluate(mvn.log_prob(sample_)),
              np.zeros(sample_shape + batch_shape))


if __name__ == "__main__":
  tf.test.main()
