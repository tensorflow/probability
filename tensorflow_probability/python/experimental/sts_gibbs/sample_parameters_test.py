# Copyright 2022 The TensorFlow Probability Authors.
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
"""Tests for sample_parameters."""

import tensorflow as tf
from tensorflow_probability.python.bijectors import invert
from tensorflow_probability.python.bijectors import square
from tensorflow_probability.python.distributions import inverse_gamma
from tensorflow_probability.python.distributions import transformed_distribution
from tensorflow_probability.python.experimental.sts_gibbs import sample_parameters
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import test_util


@test_util.test_all_tf_execution_regimes
class NormalScalePosteriorInverseGammaConjugate(test_util.TestCase):

  def testNoObservations(self):
    distribution = inverse_gamma.InverseGamma(16., 4.)
    posterior_distribution = sample_parameters.normal_scale_posterior_inverse_gamma_conjugate(
        distribution, observations=tf.constant([], dtype=tf.float32))
    self.assertIsInstance(posterior_distribution,
                          transformed_distribution.TransformedDistribution)
    self.assertIsInstance(posterior_distribution.bijector, invert.Invert)
    self.assertIsInstance(posterior_distribution.bijector.bijector,
                          square.Square)
    self.assertIsInstance(posterior_distribution.distribution,
                          inverse_gamma.InverseGamma)
    self.assertAllEqual(distribution.concentration,
                        posterior_distribution.distribution.concentration)
    self.assertAllEqual(distribution.scale,
                        posterior_distribution.distribution.scale)

  def testSingleObservation(self):
    concentration = 16.
    scale = 4.
    distribution = inverse_gamma.InverseGamma(
        concentration=concentration, scale=scale)
    posterior_distribution = sample_parameters.normal_scale_posterior_inverse_gamma_conjugate(
        distribution, observations=tf.constant([10.]))
    self.assertAllEqual(
        concentration + 0.5,  # Add half the number of observations
        posterior_distribution.distribution.concentration)
    self.assertAllEqual(
        scale + 50,  # Add half the square of observations sum.
        posterior_distribution.distribution.scale)

  def testTwoObservations(self):
    concentration = 16.
    scale = 4.
    distribution = inverse_gamma.InverseGamma(
        concentration=concentration, scale=scale)
    posterior_distribution = sample_parameters.normal_scale_posterior_inverse_gamma_conjugate(
        distribution, observations=tf.constant([10., 4.]))
    self.assertAllEqual(
        concentration + 1,  # Add half the number of observations
        posterior_distribution.distribution.concentration)
    self.assertAllEqual(
        scale + 58,  # Add half the square of observations sum.
        posterior_distribution.distribution.scale)

  def testUpperBoundPropagatedFromPrior(self):
    # If no upper bound is provided, expect there to be none.
    distribution = inverse_gamma.InverseGamma(16., 4.)
    posterior_distribution = sample_parameters.normal_scale_posterior_inverse_gamma_conjugate(
        distribution, observations=tf.constant([], dtype=tf.float32))
    self.assertFalse(hasattr(posterior_distribution, 'upper_bound'))
    self.assertFalse(
        hasattr(posterior_distribution.distribution, 'upper_bound'))

    distribution = inverse_gamma.InverseGamma(16., 4.)
    distribution.upper_bound = 16.
    posterior_distribution = sample_parameters.normal_scale_posterior_inverse_gamma_conjugate(
        distribution, observations=tf.constant([], dtype=tf.float32))
    self.assertAllEqual(posterior_distribution.distribution.upper_bound, 16.)
    self.assertAllEqual(
        posterior_distribution.upper_bound,
        # TODO(kloveless): This should have sqrt applied, but it is not for
        # temporary backwards compatibility.
        16.)


@test_util.test_all_tf_execution_regimes
class SampleWithOptionalUpperBoundTest(test_util.TestCase):

  def testBasic(self):
    distribution = inverse_gamma.InverseGamma(16., 4.)
    seed = samplers.sanitize_seed((0, 1))
    unbounded_result = sample_parameters.sample_with_optional_upper_bound(
        distribution, seed=seed)
    # Whatever the result, we want to get that value again minus a fixed offset.
    # Since we fix the seed, we know it is equal just because of the upper
    # bound.
    new_target_result = unbounded_result - 0.5
    distribution.upper_bound = new_target_result
    self.assertAllEqual(
        new_target_result,
        sample_parameters.sample_with_optional_upper_bound(
            distribution, seed=seed))


if __name__ == '__main__':
  test_util.main()
