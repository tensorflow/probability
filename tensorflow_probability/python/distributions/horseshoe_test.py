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
"""Tests for Horseshoe Distribution."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import test_case

tfe = tf.contrib.eager
tfd = tfp.distributions


@tfe.run_all_tests_in_graph_and_eager_modes
class _HorseshoeTest(object):

  def _test_param_shapes(self, sample_shape, expected):
    param_shapes = tfd.Horseshoe.param_shapes(sample_shape)
    scale_shape = param_shapes["scale"]
    self.assertAllEqual(expected, self.evaluate(scale_shape))
    scale = self._test_param(np.ones(self.evaluate(scale_shape)))
    self.assertAllEqual(
        expected,
        self.evaluate(tf.shape(tfd.Horseshoe(scale).sample())))

  def _test_param_static_shapes(self, sample_shape, expected):
    param_shapes = tfd.Horseshoe.param_static_shapes(sample_shape)
    scale_shape = param_shapes["scale"]
    self.assertEqual(expected, scale_shape)

  def _test_batch_shapes(self, dist, tensor):
    self.assertAllEqual(dist.batch_shape, tensor.shape)
    self.assertAllEqual(
        self.evaluate(dist.batch_shape_tensor()),
        self.evaluate(tensor).shape)

  def testParamShapes(self):
    sample_shape = [10, 3, 4]
    self._test_param_static_shapes(sample_shape, sample_shape)
    self._test_param_static_shapes(tf.TensorShape(sample_shape), sample_shape)
    self._test_param_shapes(sample_shape, sample_shape)
    self._test_param_shapes(tf.constant(sample_shape), sample_shape)

  def testHorseshoeMeanAndMode(self):
    scale = self._test_param([11., 12., 13.])

    dist = tfd.Horseshoe(scale=scale)

    self.assertAllEqual((3,), self.evaluate(dist.mean()).shape)
    self.assertAllEqual([0., 0., 0.], self.evaluate(dist.mean()))

    self.assertAllEqual((3,), self.evaluate(dist.mode()).shape)
    self.assertAllEqual([0., 0., 0.], self.evaluate(dist.mode()))

  def testHorseshoeSample(self):
    scale = self.dtype(2.6)
    n = 100000
    dist = tfd.Horseshoe(scale=scale)

    sample = dist.sample(n, seed=1)
    self.assertEqual(self.evaluate(sample).shape, (n,))

    scale_mle = self._scale_mle(
        sample,
        scale_candidates=self._test_param(np.linspace(2, 3, 11)))
    self.assertAllClose(scale, self.evaluate(scale_mle), atol=1e-2)

    expected_shape = tf.TensorShape([n]).concatenate(
        tf.TensorShape(self.evaluate(dist.batch_shape_tensor())))
    self.assertAllEqual(expected_shape, sample.shape)
    self.assertAllEqual(expected_shape, self.evaluate(sample).shape)

    expected_shape_static = (
        tf.TensorShape([n]).concatenate(dist.batch_shape))
    self.assertAllEqual(expected_shape_static, sample.shape)
    self.assertAllEqual(expected_shape_static, self.evaluate(sample).shape)

  def testHorseshoeSampleMultiDimensional(self):
    batch_size = 2
    scale = self._test_param([[2.8, 3.1]] * batch_size)
    n = 100000
    dist = tfd.Horseshoe(scale=scale)

    sample = dist.sample(n, seed=2)
    self.assertEqual(self.evaluate(sample).shape, (n, batch_size, 2))
    template = tf.ones_like(scale)
    scale_candidates = tf.stack(
        [template * s for s in np.linspace(2.5, 3.5, 11, dtype=self.dtype)],
        axis=-1)
    scale_mle = self._scale_mle(sample, scale_candidates)
    self.assertAllClose(
        scale, self.evaluate(scale_mle), atol=1e-2)

    expected_shape = tf.TensorShape([n]).concatenate(
        tf.TensorShape(self.evaluate(dist.batch_shape_tensor())))
    self.assertAllEqual(expected_shape, self.evaluate(sample).shape)

    expected_shape_static = (
        tf.TensorShape([n]).concatenate(dist.batch_shape))
    self.assertAllEqual(expected_shape_static, sample.shape)

  def testNegativeScaleFails(self):
    with self.assertRaisesOpError("Condition x > 0 did not hold"):
      dist = tfd.Horseshoe(scale=[self.dtype(-5)], validate_args=True, name="G")
      self.evaluate(dist.sample(1))

  def testHorseshoeShape(self):
    scale = self._test_param([6.0] * 5)
    dist = tfd.Horseshoe(scale=scale)

    self.assertEqual(self.evaluate(dist.batch_shape_tensor()), [5])
    if self.use_static_shape or tf.executing_eagerly():
      expected_batch_shape = [5]
    else:
      expected_batch_shape = None
    self.assertEqual(dist.batch_shape, tf.TensorShape(expected_batch_shape))
    self.assertAllEqual(self.evaluate(dist.event_shape_tensor()), [])
    self.assertEqual(dist.event_shape, tf.TensorShape([]))

  def testHorseshoeLogPDFWithBounds(self):
    """Test our numerical approximation of horseshoe log_prob is within bounds.

    Upper bounds and lower bounds derived in Appendix A of
    of Carvalho, Polson, Scott (2008)
    http://faculty.chicagobooth.edu/nicholas.polson/research/papers/Horse.pdf

    """
    scale = self._test_param([.5, .8, 1.0, 2.0, 3.0])
    x = self._test_param(np.logspace(-8, 8, 9).reshape((-1, 1)))
    horseshoe = tfd.Horseshoe(scale=scale)

    log_pdf = horseshoe.log_prob(x)
    self._test_batch_shapes(horseshoe, log_pdf[0])

    k = 1 / np.sqrt(2 * np.pi**3)
    upper_bound = tf.log(k * tf.log1p(
        2 / (x / scale)**2)) - tf.log(scale)
    lower_bound = tf.log(k / 2 * tf.log1p(
        4 / (x / scale)**2)) - tf.log(scale)

    tolerance = 1e-5
    self.assertAllInRange(
        self.evaluate(log_pdf), self.evaluate(lower_bound - tolerance),
        self.evaluate(upper_bound + tolerance))

  def testHorseshoeLogPDFWithMonteCarlo(self):
    scale = self._test_param([.5, .8, 1.0, 2.0, 3.0])
    horseshoe = tfd.Horseshoe(scale=scale)
    x = self._test_param(np.linspace(.1, 10.1, 11).reshape((-1, 1)))
    horseshoe_log_pdf = self.evaluate(horseshoe.log_prob(x))
    num_mc_samples = 1000000
    sigmas = tf.reshape(scale, [-1, 1]) * tfd.HalfCauchy(
        self.dtype(0.), self.dtype(1.)).sample(num_mc_samples)
    monte_carlo_horseshoe = tfd.MixtureSameFamily(
        tfd.Categorical(logits=self._test_param(np.zeros(num_mc_samples))),
        tfd.Normal(self.dtype(0.), sigmas))
    mc_log_pdf = self.evaluate(monte_carlo_horseshoe.log_prob(x))
    print("horseshoe_log_pdf:\n{}".format(horseshoe_log_pdf))
    print("MC_log_pdf:\n{}".format(mc_log_pdf))
    self.assertAllClose(mc_log_pdf, horseshoe_log_pdf, atol=0.01)

  def testHorseshoeLogPDFGradient(self):
    scale = self.dtype(2.3)
    horseshoe = tfd.Horseshoe(scale=scale)
    x = self._test_param(np.linspace(.1, 10.1, 11))
    horseshoe_log_prob_tf_gradient = self._tf_gradient(horseshoe.log_prob, x)
    # The expected derivative of log_prob can be explicitly derived from
    # PDF formula as shown in Horseshoe class docstring; it will have a
    # relatively simple form assuming PDF is known.
    k = 1 / np.sqrt(2 * np.pi**3)
    horseshoe_log_prob_derivatives_expected = x / scale**2 - 2 * k * tf.exp(
        -horseshoe.log_prob(x) - tf.log(x * scale))
    horseshoe_log_prob_gradient_expected = tf.reshape(
        horseshoe_log_prob_derivatives_expected,
        tf.shape(horseshoe_log_prob_tf_gradient))
    self.assertAllClose(
        self.evaluate(horseshoe_log_prob_gradient_expected),
        self.evaluate(horseshoe_log_prob_tf_gradient),
        # atol is not set to very tight and the max difference is observed
        # to be around 1e-3.
        atol=1.5e-3)

  def _tf_gradient(self, func, x):
    if tf.executing_eagerly():
      with tf.GradientTape() as grad_tape:
        grad_tape.watch(x)
        y = func(x)
    else:
      y = func(x)

    if tf.executing_eagerly():
      return grad_tape.gradient(y, x)
    else:
      return tf.gradients(y, x)

  def _scale_mle(self, samples, scale_candidates):
    """Max log-likelihood estimate for scale.

    Args:
      samples: Observed data points.
      scale_candidates: A simple grid of candiates for
        scale, with shape original_batch_shape + [num_candidates],
        where different candidates for a single scalar parameter are at the
        inner most dimension (axis -1).
    """
    dist = tfd.Horseshoe(scale=scale_candidates)
    dims = tf.shape(scale_candidates)
    num_candidates = dims[-1]
    original_batch_shape = dims[:-1]
    # log_likelihood has same shape as scale_candidates
    # i.e. original_batch_shape + [num_candidates]
    log_likelihood = tf.reduce_sum(
        # dist.log_prob here returns a tensor with shape
        # [num_samples] + original_batch_shape + [num_candidates]
        dist.log_prob(
            tf.reshape(samples,
                       tf.concat([[-1], original_batch_shape, [1]], axis=0))),
        axis=0)
    # max log-likelihood candidate location mask
    mask = tf.one_hot(
        tf.argmax(log_likelihood, axis=-1),
        depth=num_candidates,
        dtype=self.dtype)
    return tf.reduce_sum(scale_candidates * mask, axis=-1)

  def _test_param(self, param):
    if isinstance(param, np.ndarray):
      param_ = param.astype(self.dtype)
    else:
      param_ = np.array(param, dtype=self.dtype)
    return tf.placeholder_with_default(
        input=param_, shape=param_.shape if self.use_static_shape else None)


class HorseshoeTestStaticShapeFloat32(test_case.TestCase, _HorseshoeTest):
  dtype = np.float32
  use_static_shape = True


class HorseshoeTestDynamicShapeFloat32(test_case.TestCase, _HorseshoeTest):
  dtype = np.float32
  use_static_shape = False


class HorseshoeTestStaticShapeFloat64(test_case.TestCase, _HorseshoeTest):
  dtype = np.float64
  use_static_shape = True


class HorseshoeTestDynamicShapeFloat64(test_case.TestCase, _HorseshoeTest):
  dtype = np.float64
  use_static_shape = False


if __name__ == "__main__":
  tf.test.main()
