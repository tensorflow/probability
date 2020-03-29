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
"""Tests for the continuous Bernoulli distribution."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import numpy as np
from scipy import special as sp_special
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.internal import test_util


def make_continuous_bernoulli(batch_shape, dtype=tf.float32):
  prob = np.random.uniform(size=list(batch_shape))
  prob = tf.constant(prob, dtype=tf.float32)
  return tfd.ContinuousBernoulli(probs=prob, dtype=dtype, validate_args=True)


def log_norm_const(probs):
  # do not evaluate close to 0.5
  return np.log(np.abs(2.0 * np.arctanh(1.0 - 2.0 * probs))) - np.log(
      np.abs(1.0 - 2.0 * probs))


def pdf(x, probs):
  # do not evaluate close to 0.5
  x = np.array(x, dtype=np.float32)
  probs = np.array(probs, dtype=np.float32)
  return (
      np.power(probs, x)
      * np.power(1.0 - probs, 1.0 - x)
      * np.exp(log_norm_const(probs)))


def mean(probs):
  # do not evaluate close to 0.5
  return probs / (2.0 * probs - 1.0) + 1.0 / (
      2.0 * np.arctanh(1.0 - 2.0 * probs))


def var(probs):
  # do not evaluate close to 0.5
  return (
      probs * (probs - 1.0) / (1.0 - 2.0 * probs) ** 2
      + 1.0 / (2.0 * np.arctanh(1.0 - 2.0 * probs)) ** 2)


def entropy(probs):
  # do not evaluate close to 0.5
  return (
      mean(probs) * (np.log1p(-probs) - np.log(probs))
      - log_norm_const(probs)
      - np.log1p(-probs))


def cdf(x, probs):
  # do not evaluate close to 0.5
  tentative_cdf = (probs ** x * (1.0 - probs) ** (1.0 - x) + probs - 1.0) / (
      2. * probs - 1.)
  correct_above = np.where(x > 1., 1., tentative_cdf)
  return np.where(x < 0, 0., correct_above)


def quantile(p, probs):
  # do not evaluate close to 0.5
  return (
      np.log1p(p * (2.0 * probs - 1.0) - probs) - np.log1p(-probs)
  ) / (np.log(probs) - np.log1p(-probs))


@test_util.test_all_tf_execution_regimes
class ContinuousBernoulliTest(test_util.TestCase):

  def testProbs(self):
    prob = [0.2, 0.4]
    dist = tfd.ContinuousBernoulli(probs=prob, validate_args=True)
    self.assertAllClose(prob, self.evaluate(dist.probs))

  def testLogits(self):
    logits = [-42.0, 42.0]
    dist = tfd.ContinuousBernoulli(logits=logits, validate_args=True)
    self.assertAllClose(logits, self.evaluate(dist.logits))
    self.assertAllClose(
        sp_special.expit(logits), self.evaluate(dist.probs_parameter()))

    prob = [0.01, 0.99, 0.42]
    dist = tfd.ContinuousBernoulli(probs=prob, validate_args=True)
    self.assertAllClose(
        sp_special.logit(prob), self.evaluate(dist.logits_parameter())
    )

  def testInvalidProb(self):
    invalid_probs = [1.01, 2.0]
    for prob in invalid_probs:
      with self.assertRaisesOpError(
          "probs has components greater than 1"):
        dist = tfd.ContinuousBernoulli(probs=prob, validate_args=True)
        self.evaluate(dist.probs_parameter())

    invalid_probs = [-0.01, -3.0]
    for prob in invalid_probs:
      with self.assertRaisesOpError("probs has components less than 0"):
        dist = tfd.ContinuousBernoulli(probs=prob, validate_args=True)
        self.evaluate(dist.probs_parameter())

    valid_probs = [0.1, 0.5, 0.9]
    for prob in valid_probs:
      dist = tfd.ContinuousBernoulli(probs=prob, validate_args=True)
      self.assertEqual(np.float32(prob), self.evaluate(dist.probs))

  def testShapes(self):
    for batch_shape in ([], [1], [2, 3, 4]):
      dist = make_continuous_bernoulli(batch_shape)
      self.assertAllEqual(
          batch_shape, tensorshape_util.as_list(dist.batch_shape))
      self.assertAllEqual(
          batch_shape, self.evaluate(dist.batch_shape_tensor()))
      self.assertAllEqual([], tensorshape_util.as_list(dist.event_shape))
      self.assertAllEqual([], self.evaluate(dist.event_shape_tensor()))

  def testDtype(self):
    dist = make_continuous_bernoulli([])
    self.assertEqual(dist.dtype, tf.float32)
    self.assertEqual(
        dist.dtype, dist.sample(5, seed=test_util.test_seed()).dtype)
    self.assertEqual(dist.dtype, dist.mode().dtype)
    self.assertEqual(dist.probs.dtype, dist.mean().dtype)
    self.assertEqual(dist.probs.dtype, dist.variance().dtype)
    self.assertEqual(dist.probs.dtype, dist.stddev().dtype)
    self.assertEqual(dist.probs.dtype, dist.entropy().dtype)
    self.assertEqual(dist.probs.dtype, dist.prob(0.1).dtype)
    self.assertEqual(dist.probs.dtype, dist.prob(0.9).dtype)
    self.assertEqual(dist.probs.dtype, dist.log_prob(0.1).dtype)
    self.assertEqual(dist.probs.dtype, dist.log_prob(0.9).dtype)
    self.assertEqual(dist.probs.dtype, dist.cdf(0.1).dtype)
    self.assertEqual(dist.probs.dtype, dist.cdf(0.9).dtype)
    self.assertEqual(dist.probs.dtype, dist.log_cdf(0.1).dtype)
    self.assertEqual(dist.probs.dtype, dist.log_cdf(0.9).dtype)
    self.assertEqual(dist.probs.dtype, dist.survival_function(0.1).dtype)
    self.assertEqual(dist.probs.dtype, dist.survival_function(0.9).dtype)
    self.assertEqual(dist.probs.dtype, dist.log_survival_function(0.1).dtype)
    self.assertEqual(dist.probs.dtype, dist.log_survival_function(0.9).dtype)
    self.assertEqual(dist.probs.dtype, dist.quantile(0.1).dtype)
    self.assertEqual(dist.probs.dtype, dist.quantile(0.9).dtype)

    dist64 = make_continuous_bernoulli([], tf.float64)
    self.assertEqual(dist64.dtype, tf.float64)
    self.assertEqual(
        dist64.dtype, dist64.sample(5, seed=test_util.test_seed()).dtype)
    self.assertEqual(dist64.dtype, dist64.mode().dtype)

  def testFloatMode(self):
    dist = tfd.ContinuousBernoulli(
        probs=0.6, dtype=tf.float32, validate_args=True)
    self.assertEqual(np.float32(1), self.evaluate(dist.mode()))

  def _testPdf(self, **kwargs):
    dist = tfd.ContinuousBernoulli(validate_args=True, **kwargs)
    # pylint: disable=bad-continuation
    xs = [
        0.1,
        [0.9],
        [0.9, 0.1],
        [[0.9, 0.1]],
        [[0.9, 0.1], [0.9, 0.9]],
    ]
    expected_pmfs = [
        [[pdf(0.1, 0.2), pdf(0.1, 0.4)], [pdf(0.1, 0.3), pdf(0.1, 0.6)]],
        [[pdf(0.9, 0.2), pdf(0.9, 0.4)], [pdf(0.9, 0.3), pdf(0.9, 0.6)]],
        [[pdf(0.9, 0.2), pdf(0.1, 0.4)], [pdf(0.9, 0.3), pdf(0.1, 0.6)]],
        [[pdf(0.9, 0.2), pdf(0.1, 0.4)], [pdf(0.9, 0.3), pdf(0.1, 0.6)]],
        [[pdf(0.9, 0.2), pdf(0.1, 0.4)], [pdf(0.9, 0.3), pdf(0.9, 0.6)]],
    ]
    # pylint: enable=bad-continuation

    for x, expected_pmf in zip(xs, expected_pmfs):
      self.assertAllClose(self.evaluate(dist.prob(x)), expected_pmf)
      self.assertAllClose(
          self.evaluate(dist.log_prob(x)), np.log(expected_pmf))

  def testPdfCorrectBroadcastDynamicShape(self):
    prob = tf1.placeholder_with_default([0.2, 0.3, 0.4], shape=None)
    dist = tfd.ContinuousBernoulli(probs=prob, validate_args=True)
    event1 = [0.9, 0.1, 1.0]
    event2 = [[0.9, 0.1, 1.0]]
    self.assertAllClose(
        [pdf(0.9, 0.2), pdf(0.1, 0.3), pdf(1.0, 0.4)],
        self.evaluate(dist.prob(event1)))
    self.assertAllClose(
        [[pdf(0.9, 0.2), pdf(0.1, 0.3), pdf(1.0, 0.4)]],
        self.evaluate(dist.prob(event2)))

  def testPdfWithprob(self):
    prob = [[0.2, 0.4], [0.3, 0.6]]
    self._testPdf(probs=prob)
    self._testPdf(logits=sp_special.logit(prob))

  def testPdfWithFloatArg(self):
    prob = [[0.2], [0.4], [0.3], [0.6]]
    samps = [0.1, 0.2, 0.8]
    self.assertAllClose(
        np.log(pdf(samps, prob)),
        self.evaluate(
            tfd.ContinuousBernoulli(probs=prob, validate_args=False).log_prob(
                samps
            )
        ),
    )

  def testBroadcasting(self):
    probs = lambda prob: tf1.placeholder_with_default(prob, shape=None)
    def dist(p):
      return tfd.ContinuousBernoulli(probs=probs(p), validate_args=True)
    self.assertAllClose(0.0, self.evaluate(dist(0.5).log_prob(0.9)))
    self.assertAllClose(
        np.log([1.0, 1.0, 1.0]),
        self.evaluate(dist(0.5).log_prob([0.9, 0.9, 0.9])))
    self.assertAllClose(
        np.log([1.0, 1.0, 1.0]),
        self.evaluate(dist([0.5, 0.5, 0.5]).log_prob(0.9)))

  def testPdfShapes(self):
    probs = lambda prob: tf1.placeholder_with_default(prob, shape=None)
    def dist(p):
      return tfd.ContinuousBernoulli(probs=probs(p), validate_args=True)
    self.assertEqual(
        2, len(self.evaluate(dist([[0.5], [0.5]]).log_prob(0.9)).shape))

    dist = tfd.ContinuousBernoulli(probs=0.5, validate_args=True)
    self.assertEqual(
        2, len(self.evaluate(dist.log_prob([[0.9], [0.9]])).shape))

    dist = tfd.ContinuousBernoulli(probs=0.5, validate_args=True)
    self.assertAllEqual([], dist.log_prob(0.9).shape)
    self.assertAllEqual([1], dist.log_prob([0.9]).shape)
    self.assertAllEqual([2, 1], dist.log_prob([[0.9], [0.9]]).shape)

    dist = tfd.ContinuousBernoulli(probs=[[0.5], [0.5]], validate_args=True)
    self.assertAllEqual([2, 1], dist.log_prob(0.9).shape)

  def testEntropyNoBatch(self):
    prob = 0.2
    dist = tfd.ContinuousBernoulli(probs=prob, validate_args=True)
    self.assertAllClose(self.evaluate(dist.entropy()), entropy(prob))

  def testEntropyWithBatch(self):
    prob = [[0.1, 0.7], [0.2, 0.6]]
    dist = tfd.ContinuousBernoulli(probs=prob, validate_args=False)
    self.assertAllClose(
        self.evaluate(dist.entropy()),
        [[entropy(0.1), entropy(0.7)], [entropy(0.2), entropy(0.6)]])

  def testSampleN(self):
    prob = [0.2, 0.6]
    dist = tfd.ContinuousBernoulli(probs=prob, validate_args=True)
    n = 100000
    samples = dist.sample(n, seed=test_util.test_seed())
    tensorshape_util.set_shape(samples, [n, 2])
    self.assertEqual(samples.dtype, tf.float32)
    sample_values = self.evaluate(samples)
    self.assertTrue(np.all(sample_values > 0))
    self.assertTrue(np.all(sample_values < 1))
    # This  tolerance is very sensitive to the value of prob as well as n.
    self.assertAllClose(
        mean(np.array(prob, dtype=np.float32)),
        np.mean(sample_values, axis=0),
        atol=1e-2)
    # In this test we're just interested in verifying there isn't a crash
    # owing to mismatched types. b/30940152
    dist = tfd.ContinuousBernoulli(np.log([0.2, 0.4]), validate_args=True)
    x = dist.sample(1, seed=test_util.test_seed())
    self.assertAllEqual((1, 2), tensorshape_util.as_list(x.shape))

  def testReparameterized(self):
    prob = tf.constant([0.2, 0.6])
    _, grad_prob = tfp.math.value_and_gradient(
        lambda x: tfd.ContinuousBernoulli(probs=x, validate_args=True).sample(  # pylint: disable=g-long-lambda
            100, seed=test_util.test_seed()), prob)
    self.assertIsNotNone(grad_prob)

  def testSampleDeterministicScalarVsVector(self):
    prob = [0.2, 0.6]
    dist = tfd.ContinuousBernoulli(probs=prob, validate_args=True)
    n = 1000

    def _seed(seed=None):
      seed = test_util.test_seed() if seed is None else seed
      if tf.executing_eagerly():
        tf1.set_random_seed(seed)
      return seed

    seed = _seed()
    self.assertAllClose(
        self.evaluate(dist.sample(n, _seed(seed=seed))),
        self.evaluate(dist.sample([n], _seed(seed=seed))))
    n = tf1.placeholder_with_default(np.int32(1000), shape=None)
    seed = _seed()
    sample1 = dist.sample(n, _seed(seed=seed))
    sample2 = dist.sample([n], _seed(seed=seed))
    sample1, sample2 = self.evaluate([sample1, sample2])
    self.assertAllClose(sample1, sample2)

  def testMean(self):
    prob = np.array([[0.2, 0.7], [0.8, 0.4]], dtype=np.float32)
    dist = tfd.ContinuousBernoulli(probs=prob, validate_args=True)
    self.assertAllClose(self.evaluate(dist.mean()), mean(prob))

  def testVarianceAndStd(self):
    prob = [[0.2, 0.7], [0.8, 0.4]]
    dist = tfd.ContinuousBernoulli(probs=prob, validate_args=True)
    self.assertAllClose(
        self.evaluate(dist.variance()),
        np.array(
            [[var(0.2), var(0.7)], [var(0.8), var(0.4)]], dtype=np.float32
        ), rtol=1e-5)
    self.assertAllClose(
        self.evaluate(dist.stddev()),
        np.array(
            [
                [np.sqrt(var(0.2)), np.sqrt(var(0.7))],
                [np.sqrt(var(0.8)), np.sqrt(var(0.4))],
            ],
            dtype=np.float32), rtol=1e-5)

  def testCdfAndLogCdf(self):
    prob = 0.2
    dist = tfd.ContinuousBernoulli(probs=prob)
    self.assertAllClose(
        self.evaluate(
            dist.cdf(np.array([-2.0, 0.3, 1.1], dtype=np.float32))
        ),
        np.array(
            [cdf(-2.0, 0.2), cdf(0.3, 0.2), cdf(1.1, 0.2)], dtype=np.float32))
    self.assertAllClose(
        self.evaluate(
            dist.log_cdf(np.array([-2.0, 0.3, 1.1], dtype=np.float32))
        ),
        np.array(
            [
                np.log(cdf(-2.0, 0.2)),
                np.log(cdf(0.3, 0.2)),
                np.log(cdf(1.1, 0.2)),
            ],
            dtype=np.float32))

  def testSurvivalAndLogSurvival(self):
    prob = 0.2
    dist = tfd.ContinuousBernoulli(probs=prob)
    self.assertAllClose(
        self.evaluate(
            dist.survival_function(
                np.array([-2.0, 0.3, 1.1], dtype=np.float32)
            )
        ),
        1.0 - np.array(
            [cdf(-2.0, 0.2), cdf(0.3, 0.2), cdf(1.1, 0.2)], dtype=np.float32))
    self.assertAllClose(
        self.evaluate(
            dist.log_survival_function(
                np.array([-2.0, 0.3, 1.1], dtype=np.float32))),
        np.array(
            [
                np.log(1.0 - cdf(-2.0, 0.2)),
                np.log(1.0 - cdf(0.3, 0.2)),
                np.log(1.0 - cdf(1.1, 0.2)),
            ],
            dtype=np.float32))

  def testQuantile(self):
    prob = 0.2
    dist = tfd.ContinuousBernoulli(probs=prob, validate_args=True)
    self.assertAllClose(
        self.evaluate(
            dist.quantile(np.array([0.1, 0.3, 0.9], dtype=np.float32))
        ),
        np.array(
            [quantile(0.1, 0.2), quantile(0.3, 0.2), quantile(0.9, 0.2)],
            dtype=np.float32))

  def testContinuousBernoulliContinuousBernoulliKL(self):
    batch_size = 6
    a_p = np.array([0.6] * batch_size, dtype=np.float32)
    b_p = np.array([0.4] * batch_size, dtype=np.float32)

    a = tfd.ContinuousBernoulli(probs=a_p, validate_args=True)
    b = tfd.ContinuousBernoulli(probs=b_p, validate_args=True)

    kl = tfd.kl_divergence(a, b)
    kl_val = self.evaluate(kl)

    kl_expected = (
        mean(a_p)
        * (
            np.log(a_p)
            + np.log1p(-b_p)
            - np.log(b_p)
            - np.log1p(-a_p)
        )
        + log_norm_const(a_p)
        - log_norm_const(b_p)
        + np.log1p(-a_p)
        - np.log1p(-b_p)
    )

    self.assertEqual(kl.shape, (batch_size,))
    self.assertAllClose(kl_val, kl_expected)


class _MakeSlicer(object):

  def __getitem__(self, slices):
    return lambda x: x[slices]


make_slicer = _MakeSlicer()


@test_util.test_all_tf_execution_regimes
class ContinuousBernoulliSlicingTest(test_util.TestCase):

  def testScalarSlice(self):
    logits = self.evaluate(tf.random.normal([], seed=test_util.test_seed()))
    dist = tfd.ContinuousBernoulli(logits=logits, validate_args=True)
    self.assertAllEqual([], dist.batch_shape)
    self.assertAllEqual([1], dist[tf.newaxis].batch_shape)
    self.assertAllEqual([], dist[...].batch_shape)
    self.assertAllEqual(
        [1, 1], dist[tf.newaxis, ..., tf.newaxis].batch_shape)

  def testSlice(self):
    logits = self.evaluate(
        tf.random.normal([20, 3, 1, 5], seed=test_util.test_seed()))
    dist = tfd.ContinuousBernoulli(logits=logits, validate_args=True)
    batch_shape = tensorshape_util.as_list(dist.batch_shape)
    dist_noshape = tfd.ContinuousBernoulli(
        logits=tf1.placeholder_with_default(logits, shape=None),
        validate_args=True)

    def check(*slicers):
      for ds, assert_static_shape in (dist, True), (dist_noshape, False):
        bs = batch_shape
        prob = self.evaluate(dist.prob(0))
        for slicer in slicers:
          ds = slicer(ds)
          bs = slicer(np.zeros(bs)).shape
          prob = slicer(prob)
          if assert_static_shape or tf.executing_eagerly():
            self.assertAllEqual(bs, ds.batch_shape)
          else:
            self.assertIsNone(tensorshape_util.rank(ds.batch_shape))
          self.assertAllEqual(
              bs, self.evaluate(ds.batch_shape_tensor()))
          self.assertAllClose(prob, self.evaluate(ds.prob(0)))

    check(make_slicer[3])
    check(make_slicer[tf.newaxis])
    check(make_slicer[3::7])
    check(make_slicer[:, :2])
    check(make_slicer[tf.newaxis, :, ..., 0, :2])
    check(make_slicer[tf.newaxis, :, ..., 3:, tf.newaxis])
    check(make_slicer[..., tf.newaxis, 3:, tf.newaxis])
    check(make_slicer[..., tf.newaxis, -3:, tf.newaxis])
    check(make_slicer[tf.newaxis, :-3, tf.newaxis, ...])

    def halfway(x):
      if isinstance(x, tfd.ContinuousBernoulli):
        return x.batch_shape_tensor()[0] // 2
      return x.shape[0] // 2

    check(lambda x: x[halfway(x)])
    check(lambda x: x[: halfway(x)])
    check(lambda x: x[halfway(x) :])
    check(
        make_slicer[:-3, tf.newaxis],
        make_slicer[..., 0, :2],
        make_slicer[::2],
    )
    if tf.executing_eagerly():
      return
    with self.assertRaisesRegexp(
        (ValueError, tf.errors.InvalidArgumentError),
        "Index out of range.*input has only 4 dims"):
      check(make_slicer[19, tf.newaxis, 2, ..., :, 0, 4])
    with self.assertRaisesRegexp(
        (ValueError, tf.errors.InvalidArgumentError),
        "slice index.*out of bounds",
    ):
      check(make_slicer[..., 2, :])  # ...,1,5 -> 2 is oob.

  def testSliceSequencePreservesOrigVarGradLinkage(self):
    logits = tf.Variable(
        tf.random.normal([20, 3, 1, 5], seed=test_util.test_seed()))
    self.evaluate(logits.initializer)
    dist = tfd.ContinuousBernoulli(logits=logits, validate_args=True)
    for slicer in [make_slicer[:5], make_slicer[..., -1], make_slicer[:, 1::2]]:
      with tf.GradientTape() as tape:
        dist = slicer(dist)
        lp = dist.log_prob(0.1)
        dlpdlogits = tape.gradient(lp, logits)
        self.assertIsNotNone(dlpdlogits)
        self.assertGreater(
            self.evaluate(tf.reduce_sum(tf.abs(dlpdlogits))), 0)

  def testSliceThenCopyPreservesOrigVarGradLinkage(self):
    logits = tf.Variable(
        tf.random.normal([20, 3, 1, 5], seed=test_util.test_seed()))
    self.evaluate(logits.initializer)
    dist = tfd.ContinuousBernoulli(logits=logits, validate_args=True)
    dist = dist[:5]
    with tf.GradientTape() as tape:
      dist = dist.copy(name="contbern2")
      lp = dist.log_prob(0.1)
    dlpdlogits = tape.gradient(lp, logits)
    self.assertIn("contbern2", dist.name)
    self.assertIsNotNone(dlpdlogits)
    self.assertGreater(self.evaluate(tf.reduce_sum(tf.abs(dlpdlogits))), 0)
    with tf.GradientTape() as tape:
      dist = dist[:3]
      lp = dist.log_prob(0)
    dlpdlogits = tape.gradient(lp, logits)
    self.assertIn("contbern2", dist.name)
    self.assertIsNotNone(dlpdlogits)
    self.assertGreater(self.evaluate(tf.reduce_sum(tf.abs(dlpdlogits))), 0)

  def testCopyUnknownRank(self):
    logits = tf1.placeholder_with_default(
        tf.random.normal([20, 3, 1, 5], seed=test_util.test_seed()),
        shape=None)
    dist = tfd.ContinuousBernoulli(
        logits=logits, name="cb1", validate_args=True)
    self.assertIn("cb1", dist.name)
    dist = dist.copy(name="cb2")
    self.assertIn("cb2", dist.name)

  def testSliceCopyOverrideNameSliceAgainCopyOverrideLogitsSliceAgain(self):
    seed_stream = test_util.test_seed_stream("slice_continuous_bernoulli")
    logits = tf.random.normal([20, 3, 2, 5], seed=seed_stream())
    dist = tfd.ContinuousBernoulli(
        logits=logits, name="cb1", validate_args=True)
    self.assertIn("cb1", dist.name)
    dist = dist[:10].copy(name="cb2")
    self.assertAllEqual((10, 3, 2, 5), dist.batch_shape)
    self.assertIn("cb2", dist.name)
    dist = dist.copy(name="cb3")[..., 1]
    self.assertAllEqual((10, 3, 2), dist.batch_shape)
    self.assertIn("cb3", dist.name)
    dist = dist.copy(logits=tf.random.normal([2], seed=seed_stream()))
    self.assertAllEqual((2,), dist.batch_shape)
    self.assertIn("cb3", dist.name)

  def testDocstrSliceExample(self):
    # batch shape [3, 5, 7, 9]
    cb = tfd.ContinuousBernoulli(
        logits=tf.zeros([3, 5, 7, 9]), validate_args=True
    )
    self.assertAllEqual((3, 5, 7, 9), cb.batch_shape)
    cb2 = cb[:, tf.newaxis, ..., -2:, 1::2]  # batch shape [3, 1, 5, 2, 4]
    self.assertAllEqual((3, 1, 5, 2, 4), cb2.batch_shape)


@test_util.test_all_tf_execution_regimes
class ContinuousBernoulliFromVariableTest(test_util.TestCase):

  @test_util.tf_tape_safety_test
  def testGradientLogits(self):
    x = tf.Variable([-1.0, 1])
    self.evaluate(x.initializer)
    d = tfd.ContinuousBernoulli(logits=x, validate_args=True)
    with tf.GradientTape() as tape:
      loss = -d.log_prob([0.1, 0.9])
    g = tape.gradient(loss, d.trainable_variables)
    self.assertLen(g, 1)
    self.assertAllNotNone(g)

  @test_util.tf_tape_safety_test
  def testGradientProbs(self):
    x = tf.Variable([0.1, 0.7])
    self.evaluate(x.initializer)
    d = tfd.ContinuousBernoulli(probs=x, validate_args=True)
    with tf.GradientTape() as tape:
      loss = -d.log_prob([0.1, 0.9])
    g = tape.gradient(loss, d.trainable_variables)
    self.assertLen(g, 1)
    self.assertAllNotNone(g)


if __name__ == "__main__":
  tf.test.main()
