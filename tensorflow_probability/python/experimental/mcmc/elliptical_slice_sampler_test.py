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
"""Tests for slice_sampler_utils.py and slice_sampler_kernel.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.internal import test_util


tfd = tfp.distributions


def normal_normal_posterior(
    prior_mean,
    prior_stddev,
    likelihood_stddev,
    data):
  r"""Returns the posterior mean and variance of a normal-normal model.

  Given a normal likelihood and a normal prior on the mean, return the
  posterior mean and variance. In this model, the posterior distribution is
  normally distributed and can be computed easily from a formula.

  ```
  p(mu | mu0, sigma0) ~ N(mu0, sigma0)
  p(x | mu, sigma) ~ N(mu, sigma)
  p(mu | x, mu0, sigma0) \proportional p(x | mu, sigma) p(mu | mu0, sigma0)
  = N(mu1, sigma1)
  ```

  where:
    * `sigma1 = sqrt( 1 / (1 / sigma0**2 + n / sigma**2))`
    * `mu1 = sigma1 ** 2 * (mu0 / sigma0**2 + n * xbar / sigma**2)
    * `n` is the number of samples.
    * `xbar` is the mean of the data.

  For a derivation, see https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf.

  Args:
    prior_mean: `numpy.ndarray` representing mean of the normal prior.
    prior_stddev: `numpy.ndarray` representing standard deviation of the
      normal prior.
    likelihood_stddev: `numpy.ndarray` representing standard deviation of the
      normal likelihood.
    data: `numpy.ndarray` whose first dimension represents the number
      of data points.
  Returns:
    posterior_mean, posterior_variance: `Tensors` representing the
      posterior_mean and posterior_variance under this conjugate model.
  """

  num_samples = int(data.shape[0])
  posterior_variance = 1 / (
      1 / prior_stddev ** 2 + num_samples / likelihood_stddev ** 2)

  data_mean = np.mean(data, axis=0)

  posterior_mean = posterior_variance * (
      prior_mean / prior_stddev ** 2 +
      num_samples * data_mean / likelihood_stddev ** 2)

  return posterior_mean, posterior_variance


class _EllipticalSliceSamplerTest(test_util.TestCase):

  def testSampleChainSeedReproducible(self):
    normal_prior = tfd.Normal(5 * [[0., 0.]], 1.)

    def normal_sampler(seed):
      return normal_prior.sample(seed=seed)

    def normal_log_likelihood(state):
      return tf.math.reduce_sum(
          tfd.Normal(state, 2.).log_prob(0.),
          axis=-1)

    num_results = 10
    seed = test_util.test_seed()

    current_state = np.float32(np.random.rand(5, 2))
    samples0 = tf.function(lambda: tfp.mcmc.sample_chain(  # pylint: disable=g-long-lambda
        num_results=2 * num_results,
        num_steps_between_results=0,
        current_state=current_state,
        kernel=tfp.experimental.mcmc.EllipticalSliceSampler(
            normal_sampler_fn=normal_sampler,
            log_likelihood_fn=normal_log_likelihood,
            seed=seed),
        num_burnin_steps=150,
        trace_fn=None,
        parallel_iterations=1))()

    samples1 = tf.function(lambda: tfp.mcmc.sample_chain(  # pylint: disable=g-long-lambda
        num_results=num_results,
        num_steps_between_results=1,
        current_state=current_state,
        kernel=tfp.experimental.mcmc.EllipticalSliceSampler(
            normal_sampler_fn=normal_sampler,
            log_likelihood_fn=normal_log_likelihood,
            seed=seed),
        trace_fn=None,
        num_burnin_steps=150,
        parallel_iterations=1))()
    samples0_, samples1_ = self.evaluate([samples0, samples1])

    self.assertAllClose(samples0_[::2], samples1_, atol=1e-5, rtol=1e-5)

  # In order to test Elliptical Slice Sampling, we take
  # the case where we have a Normal prior and Normal posterior.
  def testNormalNormalSample(self):
    # Standard normal prior.
    # Samples are shape [2].
    normal_prior = tfd.Normal(self.dtype([0., 0.]), self.dtype(1.))

    def normal_sampler(seed):
      return normal_prior.sample(seed=seed)

    # A single data point at the mode.
    # The state is expected to be 2 dimensional, so
    # we reduce sum on the last axis.
    def normal_log_likelihood(state):
      return tf.math.reduce_sum(
          tfd.Normal(state, self.dtype(2.)).log_prob(self.dtype(0.)),
          axis=-1)

    kernel = tfp.experimental.mcmc.EllipticalSliceSampler(
        normal_sampler_fn=normal_sampler,
        log_likelihood_fn=normal_log_likelihood,
        seed=test_util.test_seed(),
    )

    samples = tf.function(lambda: tfp.mcmc.sample_chain(  # pylint: disable=g-long-lambda
        num_results=int(3e5),
        current_state=self.dtype(np.random.randn(2)),
        kernel=kernel,
        num_burnin_steps=int(1e4),
        parallel_iterations=1,
        trace_fn=None
    ))()

    mean, variance = self.evaluate(
        tf.nn.moments(samples, axes=[0]))
    # Computed exactly from the formula in normal-normal posterior.
    self.assertAllClose([0., 0.], mean, rtol=5e-2, atol=6e-3)
    self.assertAllClose([4./5, 4./5], variance, rtol=5e-2)

  def testNormalNormalSampleMultipleDatapoints(self):
    # Two independent chains, of states of shape [3].
    prior_stddev = self.dtype(np.exp(np.random.rand(2, 3)))

    likelihood_stddev = self.dtype(np.exp(np.random.rand(2, 3)))
    # 10 data points.
    data = self.dtype(np.random.randn(10, 2, 3))

    # Standard normal prior.
    normal_prior = tfd.Normal(self.dtype(0.), prior_stddev)

    def normal_sampler(seed):
      return normal_prior.sample(seed=seed)

    # 10 samples at 2 chains.
    def normal_log_likelihood(state):
      return tf.math.reduce_sum(
          tfd.Normal(state, likelihood_stddev).log_prob(data),
          axis=[0, -1],
      )

    kernel = tfp.experimental.mcmc.EllipticalSliceSampler(
        normal_sampler_fn=normal_sampler,
        log_likelihood_fn=normal_log_likelihood,
        seed=test_util.test_seed(),
    )

    samples = tf.function(lambda: tfp.mcmc.sample_chain(  # pylint: disable=g-long-lambda
        num_results=int(3e5),
        current_state=self.dtype(np.random.randn(2, 3)),
        kernel=kernel,
        num_burnin_steps=int(1e4),
        parallel_iterations=1,
        trace_fn=None))()

    mean, variance = self.evaluate(
        tf.nn.moments(samples, axes=[0]))
    posterior_mean, posterior_variance = normal_normal_posterior(
        prior_mean=0.,
        prior_stddev=prior_stddev,
        likelihood_stddev=likelihood_stddev,
        data=data)
    # Computed exactly from the formula in normal-normal posterior.
    self.assertAllClose(posterior_mean, mean, rtol=2e-2, atol=6e-3)
    self.assertAllClose(posterior_variance, variance, rtol=5e-2)


@test_util.test_all_tf_execution_regimes
class EllipticalSliceSamplerTestFloat32(_EllipticalSliceSamplerTest):
  dtype = np.float32


@test_util.test_all_tf_execution_regimes
class EllipticalSliceSamplerTestFloat64(_EllipticalSliceSamplerTest):
  dtype = np.float64


del _EllipticalSliceSamplerTest


if __name__ == '__main__':
  tf.test.main()
