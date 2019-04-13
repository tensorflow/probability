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

import importlib
# Dependency imports
import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.internal import test_util as tfp_test_util
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top


def try_import(name):  # pylint: disable=invalid-name
  module = None
  try:
    module = importlib.import_module(name)
  except ImportError as e:
    tf.compat.v1.logging.warning("Could not import %s: %s" % (name, str(e)))
  return module

stats = try_import("scipy.stats")

tfd = tfp.distributions


@test_util.run_all_in_graph_and_eager_modes
class IndependentDistributionTest(tf.test.TestCase):

  def setUp(self):
    self._rng = np.random.RandomState(42)

  def testSampleAndLogProbUnivariate(self):
    loc = np.float32([-1., 1])
    scale = np.float32([0.1, 0.5])
    ind = tfd.Independent(
        distribution=tfd.Normal(loc=loc, scale=scale),
        reinterpreted_batch_ndims=1)

    x = ind.sample([4, 5], seed=tfp_test_util.test_seed(hardcoded_seed=42))
    log_prob_x = ind.log_prob(x)
    x_, actual_log_prob_x = self.evaluate([x, log_prob_x])

    self.assertEqual([], ind.batch_shape)
    self.assertEqual([2], ind.event_shape)
    self.assertEqual([4, 5, 2], x.shape)
    self.assertEqual([4, 5], log_prob_x.shape)

    expected_log_prob_x = stats.norm(loc, scale).logpdf(x_).sum(-1)
    self.assertAllClose(
        expected_log_prob_x, actual_log_prob_x, rtol=1e-5, atol=0.)

  def testSampleAndLogProbMultivariate(self):
    loc = np.float32([[-1., 1], [1, -1]])
    scale = np.float32([1., 0.5])
    ind = tfd.Independent(
        distribution=tfd.MultivariateNormalDiag(
            loc=loc, scale_identity_multiplier=scale),
        reinterpreted_batch_ndims=1)

    x = ind.sample([4, 5], seed=tfp_test_util.test_seed())
    log_prob_x = ind.log_prob(x)
    x_, actual_log_prob_x = self.evaluate([x, log_prob_x])

    self.assertEqual([], ind.batch_shape)
    self.assertEqual([2, 2], ind.event_shape)
    self.assertEqual([4, 5, 2, 2], x.shape)
    self.assertEqual([4, 5], log_prob_x.shape)

    expected_log_prob_x = stats.norm(loc,
                                     scale[:, None]).logpdf(x_).sum(-1).sum(-1)
    self.assertAllClose(
        expected_log_prob_x, actual_log_prob_x, rtol=1e-6, atol=0.)

  def testCdfMultivariate(self):
    ind = tfd.Independent(
        distribution=tfd.Normal(loc=tf.zeros([3]), scale=1.),
        reinterpreted_batch_ndims=1)

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
        reinterpreted_batch_ndims=1)

    x = ind.sample(int(n_samp), seed=tfp_test_util.test_seed(hardcoded_seed=42))
    sample_mean = tf.reduce_mean(input_tensor=x, axis=0)
    sample_var = tf.reduce_mean(
        input_tensor=tf.math.squared_difference(x, sample_mean), axis=0)
    sample_std = tf.sqrt(sample_var)
    sample_entropy = -tf.reduce_mean(input_tensor=ind.log_prob(x), axis=0)

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

    self.assertAllClose(sample_mean_, actual_mean_, rtol=0.02, atol=0.)
    self.assertAllClose(sample_var_, actual_var_, rtol=0.04, atol=0.)
    self.assertAllClose(sample_std_, actual_std_, rtol=0.02, atol=0.)
    self.assertAllClose(sample_entropy_, actual_entropy_, rtol=0.01, atol=0.)
    self.assertAllClose(loc, actual_mode_, rtol=1e-6, atol=0.)

  def test_event_ndims_is_static_when_possible(self):
    ind = tfd.Independent(
        distribution=tfd.Normal(
            loc=tf.compat.v1.placeholder_with_default(input=[2.], shape=None),
            scale=tf.compat.v1.placeholder_with_default(input=1., shape=None)),
        reinterpreted_batch_ndims=1)
    # Even though `event_shape` is not static, event_ndims must equal
    # `reinterpreted_batch_ndims + rank(distribution.event_shape)`.
    self.assertEqual(tensorshape_util.rank(ind.event_shape), 1)

  def testKLRaises(self):
    ind1 = tfd.Independent(
        distribution=tfd.Normal(
            loc=np.float32([-1., 1]), scale=np.float32([0.1, 0.5])),
        reinterpreted_batch_ndims=1)
    ind2 = tfd.Independent(
        distribution=tfd.Normal(
            loc=np.float32(-1), scale=np.float32(0.5)),
        reinterpreted_batch_ndims=0)

    with self.assertRaisesRegexp(
        ValueError, "Event shapes do not match"):
      tfd.kl_divergence(ind1, ind2)

    ind1 = tfd.Independent(
        distribution=tfd.Normal(
            loc=np.float32([-1., 1]), scale=np.float32([0.1, 0.5])),
        reinterpreted_batch_ndims=1)
    ind2 = tfd.Independent(
        distribution=tfd.MultivariateNormalDiag(
            loc=np.float32([-1., 1]), scale_diag=np.float32([0.1, 0.5])),
        reinterpreted_batch_ndims=0)

    with self.assertRaisesRegexp(
        NotImplementedError, "different event shapes"):
      tfd.kl_divergence(ind1, ind2)

  def testKLScalarToMultivariate(self):
    normal1 = tfd.Normal(
        loc=np.float32([-1., 1]), scale=np.float32([0.1, 0.5]))
    ind1 = tfd.Independent(distribution=normal1, reinterpreted_batch_ndims=1)

    normal2 = tfd.Normal(
        loc=np.float32([-3., 3]), scale=np.float32([0.3, 0.3]))
    ind2 = tfd.Independent(distribution=normal2, reinterpreted_batch_ndims=1)

    normal_kl = tfd.kl_divergence(normal1, normal2)
    ind_kl = tfd.kl_divergence(ind1, ind2)
    self.assertAllClose(
        self.evaluate(tf.reduce_sum(input_tensor=normal_kl, axis=-1)),
        self.evaluate(ind_kl))

  def testKLIdentity(self):
    normal1 = tfd.Normal(
        loc=np.float32([-1., 1]), scale=np.float32([0.1, 0.5]))
    # This is functionally just a wrapper around normal1,
    # and doesn't change any outputs.
    ind1 = tfd.Independent(distribution=normal1, reinterpreted_batch_ndims=0)

    normal2 = tfd.Normal(
        loc=np.float32([-3., 3]), scale=np.float32([0.3, 0.3]))
    # This is functionally just a wrapper around normal2,
    # and doesn't change any outputs.
    ind2 = tfd.Independent(distribution=normal2, reinterpreted_batch_ndims=0)

    normal_kl = tfd.kl_divergence(normal1, normal2)
    ind_kl = tfd.kl_divergence(ind1, ind2)
    self.assertAllClose(
        self.evaluate(normal_kl), self.evaluate(ind_kl))

  def testKLMultivariateToMultivariate(self):
    # (1, 1, 2) batch of MVNDiag
    mvn1 = tfd.MultivariateNormalDiag(
        loc=np.float32([[[[-1., 1, 3.], [2., 4., 3.]]]]),
        scale_diag=np.float32([[[0.2, 0.1, 5.], [2., 3., 4.]]]))
    ind1 = tfd.Independent(distribution=mvn1, reinterpreted_batch_ndims=2)

    # (1, 1, 2) batch of MVNDiag
    mvn2 = tfd.MultivariateNormalDiag(
        loc=np.float32([[[[-2., 3, 2.], [1., 3., 2.]]]]),
        scale_diag=np.float32([[[0.1, 0.5, 3.], [1., 2., 1.]]]))

    ind2 = tfd.Independent(distribution=mvn2, reinterpreted_batch_ndims=2)

    mvn_kl = tfd.kl_divergence(mvn1, mvn2)
    ind_kl = tfd.kl_divergence(ind1, ind2)
    self.assertAllClose(
        self.evaluate(tf.reduce_sum(input_tensor=mvn_kl, axis=[-1, -2])),
        self.evaluate(ind_kl))

  def _testMnistLike(self, static_shape):
    sample_shape = [4, 5]
    batch_shape = [10]
    image_shape = [28, 28, 1]
    logits = 3 * self._rng.random_sample(
        batch_shape + image_shape).astype(np.float32) - 1

    def expected_log_prob(x, logits):
      return (x * logits - np.log1p(np.exp(logits))).sum(-1).sum(-1).sum(-1)

    logits_ph = tf.compat.v1.placeholder_with_default(
        input=logits, shape=logits.shape if static_shape else None)
    ind = tfd.Independent(
        distribution=tfd.Bernoulli(logits=logits_ph))
    x = ind.sample(sample_shape, seed=tfp_test_util.test_seed())
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
        tf.shape(input=x),
        tf.shape(input=log_prob_x),
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
        expected_log_prob(x_, logits), actual_log_prob_x, rtol=1e-6, atol=0.)

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
    d = tfd.Independent(tfd.Bernoulli(logits=0))
    self.assertAllEqual([], d[...].batch_shape)
    self.assertAllEqual([], d[...].event_shape)
    self.assertAllEqual([1], d[tf.newaxis].batch_shape)
    self.assertAllEqual([], d[tf.newaxis].event_shape)
    self.assertAllEqual([1], d[..., tf.newaxis].batch_shape)
    self.assertAllEqual([], d[..., tf.newaxis].event_shape)
    self.assertAllEqual([1, 1], d[tf.newaxis, ..., tf.newaxis].batch_shape)
    self.assertAllEqual([], d[tf.newaxis, ..., tf.newaxis].event_shape)

  def testSlicingGeneral(self):
    d = tfd.Independent(tfd.Bernoulli(logits=tf.zeros([5, 6])))
    self.assertAllEqual([5], d.batch_shape)
    self.assertAllEqual([6], d.event_shape)
    self.assertAllEqual([1, 5], d[tf.newaxis].batch_shape)
    self.assertAllEqual([6], d[tf.newaxis].event_shape)

    d = tfd.Independent(tfd.Bernoulli(logits=tf.zeros([4, 5, 6])))
    self.assertAllEqual([4], d.batch_shape)
    self.assertAllEqual([5, 6], d.event_shape)
    self.assertAllEqual([1, 3], d[tf.newaxis, ..., :3].batch_shape)
    self.assertAllEqual([5, 6], d[tf.newaxis, ..., :3].event_shape)

    d = tfd.Independent(tfd.Bernoulli(logits=tf.zeros([4, 5, 6])),
                        reinterpreted_batch_ndims=1)
    self.assertAllEqual([4, 5], d.batch_shape)
    self.assertAllEqual([6], d.event_shape)
    self.assertAllEqual([1, 4, 3], d[tf.newaxis, ..., :3].batch_shape)
    self.assertAllEqual([6], d[tf.newaxis, ..., :3].event_shape)

  def testSlicingSubclasses(self):
    class IndepBern1d(tfd.Independent):

      def __init__(self, logits):
        super(IndepBern1d, self).__init__(tfd.Bernoulli(logits=logits,
                                                        dtype=tf.float32),
                                          reinterpreted_batch_ndims=1)
        self._parameters = {"logits": logits}

    d = IndepBern1d(tf.zeros([4, 5, 6]))
    with self.assertRaisesRegexp(NotImplementedError,
                                 "does not support batch slicing"):
      d[:3]  # pylint: disable=pointless-statement

    class IndepBern1dSliceable(IndepBern1d):

      def _params_event_ndims(self):
        return dict(logits=1)

    d_sliceable = IndepBern1dSliceable(tf.zeros([4, 5, 6]))
    self.assertAllEqual([3, 5], d_sliceable[:3].batch_shape)
    self.assertAllEqual([6], d_sliceable[:3].event_shape)


if __name__ == "__main__":
  tf.test.main()
