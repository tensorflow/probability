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
"""Tests for GeneralizedGamma distribution."""
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
from scipy import stats
from scipy import special

import tensorflow.compat.v2 as tf
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.distributions.internal import statistical_testing as st
from tensorflow_probability.python.internal import test_util


@test_util.test_all_tf_execution_regimes
class _GeneralizedGammaTest(object):

  def testGeneralizedGammaShape(self):
    shape = np.array([1.] * 5, dtype=self.dtype)
    scale = np.array([2.] * 5, dtype=self.dtype)
    exponent = np.array([1.] * 5, dtype=self.dtype)
    GeneralizedGamma = tfd.GeneralizedGamma(
        scale=self.make_input(scale),
        shape=self.make_input(shape),
        exponent=self.make_input(exponent),
        validate_args=True)
    if self.use_static_shape:
      self.assertEqual((5,), self.evaluate(GeneralizedGamma.batch_shape_tensor()))
      self.assertEqual(tf.TensorShape([5]), GeneralizedGamma.batch_shape)
      self.assertAllEqual([], self.evaluate(GeneralizedGamma.event_shape_tensor()))
      self.assertEqual(tf.TensorShape([]), GeneralizedGamma.event_shape)

  def testInvalidScale(self):
    scale = self.make_input(np.array([-0.01, 0., 2.], dtype=self.dtype))
    with self.assertRaisesOpError('Argument `scale` must be positive.'):
      GeneralizedGamma = tfd.GeneralizedGamma(shape=1., scale=scale, exponent=1., validate_args=True)
      self.evaluate(GeneralizedGamma.mean())

    scale = tf.Variable([0.01])
    self.evaluate(scale.initializer)
    GeneralizedGamma = tfd.GeneralizedGamma(shape=1., scale=scale, exponent=1., validate_args=True)
    self.assertIs(scale, GeneralizedGamma.scale)
    self.evaluate(GeneralizedGamma.mean())
    with tf.control_dependencies([scale.assign([-0.01])]):
      with self.assertRaisesOpError('Argument `scale` must be positive.'):
        self.evaluate(GeneralizedGamma.mean())

  def testInvalidShape(self):
    shape = [-0.01, 0., 2.]
    with self.assertRaisesOpError('Argument `shape` must be positive.'):
      GeneralizedGamma = tfd.GeneralizedGamma(
          shape=shape, scale=1., exponent=1., validate_args=True)
      self.evaluate(GeneralizedGamma.mean())

    shape = tf.Variable([0.01])
    self.evaluate(shape.initializer)
    GeneralizedGamma = tfd.GeneralizedGamma(
        shape=shape, scale=1., exponent=1., validate_args=True)
    self.assertIs(shape, GeneralizedGamma.shape)
    self.evaluate(GeneralizedGamma.mean())
    with tf.control_dependencies([shape.assign([-0.01])]):
      with self.assertRaisesOpError(
          'Argument `shape` must be positive.'):
        self.evaluate(GeneralizedGamma.mean())
        
  def testInvalidExponent(self):
    exponent = [-0.01, 0., 2.]
    with self.assertRaisesOpError('Argument `exponent` must be positive.'):
      GeneralizedGamma = tfd.GeneralizedGamma(
          shape=1., scale=1., exponent=exponent, validate_args=True)
      self.evaluate(GeneralizedGamma.mean())

    exponent = tf.Variable([0.01])
    self.evaluate(exponent.initializer)
    GeneralizedGamma = tfd.GeneralizedGamma(
        shape=1., scale=1., exponent=exponent, validate_args=True)
    self.assertIs(exponent, GeneralizedGamma.exponent)
    self.evaluate(GeneralizedGamma.mean())
    with tf.control_dependencies([exponent.assign([-0.01])]):
      with self.assertRaisesOpError(
          'Argument `exponent` must be positive.'):
        self.evaluate(GeneralizedGamma.mean())

  def testGeneralizedGammaEntropy(self):
    shape = np.array([7.8], dtype=self.dtype)
    scale = np.array([1.1], dtype=self.dtype)
    exponent = np.array([1.0], dtype=self.dtype)

    GeneralizedGamma = tfd.GeneralizedGamma(
        shape=self.make_input(shape),
        scale=self.make_input(scale),
        exponent=self.make_input(exponent),
        validate_args=True)

    entropy = GeneralizedGamma.entropy()
    expected_entropy = (
      np.log(scale) + np.special.gammaln(shape/exponent)
      - np.log(exponent) + shape/exponent
      + (1.0 - shape)/exponent*special.digamma(shape/exponent)
    )

    self.assertAllClose(
        self.evaluate(entropy), expected_entropy, atol=1e-5,
        rtol=1e-5)  # relaxed tol for fp32 in JAX
    self.assertEqual(self.evaluate(GeneralizedGamma.batch_shape_tensor()), entropy.shape)


  def testGeneralizedGammaSample(self):
    shape = self.dtype(4.)
    scale = self.dtype(1.)
    exponent = self.dtype(1.)
    n = int(100e3)

    GeneralizedGamma = tfd.GeneralizedGamma(
        shape=self.make_input(shape),
        scale=self.make_input(scale),
        exponent=self.make_input(exponent),
        validate_args=True)

    samples = GeneralizedGamma.sample(n, seed=test_util.test_seed())
    sample_values = self.evaluate(samples)

    low = self.dtype(0.)
    high = self.dtype(np.inf)

    self.evaluate(
        st.assert_true_mean_equal_by_dkwm(
            samples,
            low=low,
            high=high,
            expected=GeneralizedGamma.mean(),
            false_fail_rate=self.dtype(1e-6)))

  def testSampleLikeArgsGetDistDType(self):
    if self.dtype is np.float32:
      # Raw Python literals should always be interpreted as fp32.
      dist = tfd.GeneralizedGamma(1., 2.)
    elif self.dtype is np.float64:
      # The make_input function will cast them to self.dtype
      dist = tfd.GeneralizedGamma(self.make_input(1.), self.make_input(2.))
    self.assertEqual(self.dtype, dist.dtype)
    for method in ('log_prob', 'prob', 'log_cdf', 'cdf'):
      self.assertEqual(self.dtype, getattr(dist, method)(1.).dtype)
    for method in ('entropy', 'mean', 'variance', 'stddev', 'mode'):
      self.assertEqual(self.dtype, getattr(dist, method)().dtype)

  def testSupportBijectorOutsideRange(self):
    shape = np.array([2., 4., 5.], dtype=self.dtype)
    scale = np.array([2., 4., 5.], dtype=self.dtype)
    exponent = np.array([2., 4., 5.], dtype=self.dtype)

    dist = tfd.GeneralizedGamma(
        shape=shape, scale=scale, exponent=exponent, validate_args=True)
    x = np.array([-8.3, -0.4, -1e-6])
    bijector_inverse_x = dist.experimental_default_event_space_bijector(
    ).inverse(x)
    self.assertAllNan(self.evaluate(bijector_inverse_x))


@test_util.test_all_tf_execution_regimes
class GeneralizedGammaTestStaticShapeFloat32(test_util.TestCase, _GeneralizedGammaTest):
  dtype = np.float32
  use_static_shape = True


@test_util.test_all_tf_execution_regimes
class GeneralizedGammaTestStaticShapeFloat64(test_util.TestCase, _GeneralizedGammaTest):
  dtype = np.float64
  use_static_shape = True


@test_util.test_all_tf_execution_regimes
class GeneralizedGammaTestDynamicShapeFloat32(test_util.TestCase, _GeneralizedGammaTest):
  dtype = np.float32
  use_static_shape = False


@test_util.test_all_tf_execution_regimes
class GeneralizedGammaTestDynamicShapeFloat64(test_util.TestCase, _GeneralizedGammaTest):
  dtype = np.float64
  use_static_shape = True


if __name__ == '__main__':
  tf.test.main()
