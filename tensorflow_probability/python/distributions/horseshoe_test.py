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

# Dependency imports
import numpy as np
import scipy.stats
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.distributions import horseshoe
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.math import gradient


@test_util.test_all_tf_execution_regimes
class _HorseshoeTest(object):

  def _test_param_shapes(self, sample_shape, expected):
    param_shapes = horseshoe.Horseshoe.param_shapes(sample_shape)
    scale_shape = param_shapes['scale']
    self.assertAllEqual(expected, self.evaluate(scale_shape))
    scale = self._test_param(np.ones(self.evaluate(scale_shape)))
    self.assertAllEqual(
        expected,
        self.evaluate(
            tf.shape(
                horseshoe.Horseshoe(
                    scale,
                    validate_args=True).sample(seed=test_util.test_seed()))))

  def _test_param_static_shapes(self, sample_shape, expected):
    param_shapes = horseshoe.Horseshoe.param_static_shapes(sample_shape)
    scale_shape = param_shapes['scale']
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

    dist = horseshoe.Horseshoe(scale=scale, validate_args=True)

    self.assertAllEqual((3,), self.evaluate(dist.mean()).shape)
    self.assertAllEqual([0., 0., 0.], self.evaluate(dist.mean()))

    self.assertAllEqual((3,), self.evaluate(dist.mode()).shape)
    self.assertAllEqual([0., 0., 0.], self.evaluate(dist.mode()))

  def testHorseshoeSample(self):
    scale = self.dtype(2.6)
    n = 100000
    dist = horseshoe.Horseshoe(scale=scale, validate_args=True)

    sample = dist.sample(n, seed=test_util.test_seed())
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
    dist = horseshoe.Horseshoe(scale=scale, validate_args=True)

    sample = dist.sample(n, seed=test_util.test_seed())
    self.assertEqual(self.evaluate(sample).shape, (n, batch_size, 2))
    template = tf.ones_like(scale)
    scale_candidates = tf.stack(
        [template * s for s in np.linspace(2.5, 3.5, 11, dtype=self.dtype)],
        axis=-1)
    scale_mle = self._scale_mle(sample, scale_candidates)
    self.assertAllClose(scale, self.evaluate(scale_mle), rtol=.15)

    expected_shape = tf.TensorShape([n]).concatenate(
        tf.TensorShape(self.evaluate(dist.batch_shape_tensor())))
    self.assertAllEqual(expected_shape, self.evaluate(sample).shape)

    expected_shape_static = (
        tf.TensorShape([n]).concatenate(dist.batch_shape))
    self.assertAllEqual(expected_shape_static, sample.shape)

  def testNegativeScaleFails(self):
    with self.assertRaisesOpError('Condition x > 0 did not hold'):
      dist = horseshoe.Horseshoe(
          scale=[self.dtype(-5)], validate_args=True, name='G')
      self.evaluate(dist.sample(1, seed=test_util.test_seed()))

  def testHorseshoeShape(self):
    scale = self._test_param([6.0] * 5)
    dist = horseshoe.Horseshoe(scale=scale, validate_args=True)

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
    scale_np = np.array([.5, .8, 1.0, 2.0, 3.0])
    x_np = np.logspace(-8, 8, 9).reshape((-1, 1))
    scale = self._test_param(scale_np)
    x = self._test_param(x_np)
    dist = horseshoe.Horseshoe(scale=scale, validate_args=True)

    log_pdf = dist.log_prob(x)
    self._test_batch_shapes(dist, log_pdf[0])

    k = 1 / np.sqrt(2 * np.pi**3)
    upper_bound = np.log(
        k * np.log1p(2. / (x_np / scale_np) ** 2.)) - np.log(scale_np)
    lower_bound = np.log(
        k / 2. * np.log1p(4. / (x_np / scale_np) ** 2.)) - np.log(scale_np)

    tolerance = 1e-5
    self.assertAllInRange(
        self.evaluate(log_pdf),
        lower_bound - tolerance,
        upper_bound + tolerance)

  def testHorseshoeLogPDFWithQuadrature(self):
    scale_np = np.array([.5, .8, 1.0, 2.0, 3.0])
    scale = self._test_param(scale_np)
    dist = horseshoe.Horseshoe(scale=scale, validate_args=True)
    x = np.linspace(.1, 10.1, 11)
    dist_log_pdf = self.evaluate(
        dist.log_prob(self._test_param(x.reshape((-1, 1)))))

    # Now use quadrature to estimate the log_prob. This is
    def log_prob_at_x(x, global_shrinkage):
      def integrand(z):
        return (np.exp(scipy.stats.halfcauchy.logpdf(z, loc=0, scale=1) +
                       scipy.stats.norm.logpdf(
                           x, loc=0, scale=global_shrinkage * z)))
      return scipy.integrate.quad(integrand, 0., np.inf)[0]

    log_probs_quad = []
    for p in x:
      log_probs_quad.append([])
      for s in scale_np:
        log_probs_quad[-1].append(log_prob_at_x(p, s))
    log_probs_quad = np.log(log_probs_quad)
    self.assertAllClose(log_probs_quad, dist_log_pdf, atol=0.01)

  @test_util.numpy_disable_gradient_test
  def testHorseshoeLogPDFGradient(self):
    scale = self.dtype(2.3)
    x = self._test_param(np.linspace(0.1, 10.1, 11))
    [
        dist_log_prob,
        dist_log_prob_gradient,
    ] = gradient.value_and_gradient(
        lambda x_: horseshoe.Horseshoe(scale=scale, validate_args=True).  # pylint: disable=g-long-lambda
        log_prob(x_), x)
    # The expected derivative of log_prob can be explicitly derived from
    # PDF formula as shown in Horseshoe class docstring; it will have a
    # relatively simple form assuming PDF is known.
    k = 1 / np.sqrt(2 * np.pi**3)
    dist_log_prob_derivatives_expected = x / scale**2 - 2 * k * tf.exp(
        -dist_log_prob - tf.math.log(x * scale))
    dist_log_prob_gradient_expected = tf.reshape(
        dist_log_prob_derivatives_expected, tf.shape(dist_log_prob_gradient))
    self.assertAllClose(
        self.evaluate(dist_log_prob_gradient_expected),
        self.evaluate(dist_log_prob_gradient),
        # atol is not set to very tight and the max difference is observed
        # to be around 1e-3.
        atol=1.5e-3)

  def _scale_mle(self, samples, scale_candidates):
    """Max log-likelihood estimate for scale.

    Args:
      samples: Observed data points.
      scale_candidates: A simple grid of candiates for
        scale, with shape original_batch_shape + [num_candidates],
        where different candidates for a single scalar parameter are at the
        inner most dimension (axis -1).
    Returns:
      scale_mle: max log-likelihood estimate for scale.
    """
    dist = horseshoe.Horseshoe(scale=scale_candidates, validate_args=True)
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
    return tf1.placeholder_with_default(
        param_, shape=param_.shape if self.use_static_shape else None)


class HorseshoeTestStaticShapeFloat32(test_util.TestCase, _HorseshoeTest):
  dtype = np.float32
  use_static_shape = True


class HorseshoeTestDynamicShapeFloat32(test_util.TestCase, _HorseshoeTest):
  dtype = np.float32
  use_static_shape = False


class HorseshoeTestStaticShapeFloat64(test_util.TestCase, _HorseshoeTest):
  dtype = np.float64
  use_static_shape = True


class HorseshoeTestDynamicShapeFloat64(test_util.TestCase, _HorseshoeTest):
  dtype = np.float64
  use_static_shape = False


if __name__ == '__main__':
  test_util.main()
