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
"""Tests for MCMC driver, `sample_annealed_importance_chain`."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
# Dependency imports
import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import

tfd = tfp.distributions


def _compute_sample_variance(x, axis=None, keepdims=False):
  sample_mean = tf.reduce_mean(input_tensor=x, axis=axis, keepdims=True)
  return tf.reduce_mean(
      input_tensor=tf.math.squared_difference(x, sample_mean),
      axis=axis,
      keepdims=keepdims)


_maybe_seed = lambda s: tf.compat.v1.set_random_seed(s) if tf.executing_eagerly(
) else s


@test_util.run_all_in_graph_and_eager_modes
class SampleAnnealedImportanceTest(tf.test.TestCase):

  def setUp(self):
    self._shape_param = 5.
    self._rate_param = 10.

    tf.compat.v1.random.set_random_seed(10003)
    np.random.seed(10003)

  def _log_gamma_log_prob(self, x, event_dims=()):
    """Computes unnormalized log-pdf of a log-gamma random variable.

    Args:
      x: Value of the random variable.
      event_dims: Dimensions not to treat as independent.

    Returns:
      log_prob: The log-pdf up to a normalizing constant.
    """
    return tf.reduce_sum(
        input_tensor=self._shape_param * x - self._rate_param * tf.exp(x),
        axis=event_dims)

  # TODO(b/74154679): Create Fake TransitionKernel and not rely on HMC.

  def _ais_gets_correct_log_normalizer(self, init, independent_chain_ndims):
    counter = collections.Counter()

    def proposal_log_prob(x):
      counter['proposal_calls'] += 1
      event_dims = tf.range(independent_chain_ndims, tf.rank(x))
      return tf.reduce_sum(
          input_tensor=tfd.Normal(loc=0., scale=1.).log_prob(x),
          axis=event_dims)

    def target_log_prob(x):
      counter['target_calls'] += 1
      event_dims = tf.range(independent_chain_ndims, tf.rank(x))
      return self._log_gamma_log_prob(x, event_dims)

    num_steps = 200

    def make_kernel(tlp_fn):
      return tfp.mcmc.HamiltonianMonteCarlo(
          target_log_prob_fn=tlp_fn,
          step_size=0.5,
          num_leapfrog_steps=2,
          seed=_maybe_seed(make_kernel.seed()))
    make_kernel.seed = tfp.distributions.SeedStream('make_kernel', 45)

    _, ais_weights, _ = tfp.mcmc.sample_annealed_importance_chain(
        num_steps=num_steps,
        proposal_log_prob_fn=proposal_log_prob,
        target_log_prob_fn=target_log_prob,
        current_state=init,
        make_kernel_fn=make_kernel,
        parallel_iterations=1)

    # We have three calls because the calculation of `ais_weights` entails
    # another call to the `convex_combined_log_prob_fn`. We could refactor
    # things to avoid this, if needed (eg, b/72994218).
    if not tf.executing_eagerly():
      self.assertAllEqual(dict(target_calls=5, proposal_calls=5), counter)

    event_shape = tf.shape(input=init)[independent_chain_ndims:]
    event_size = tf.reduce_prod(input_tensor=event_shape)

    log_true_normalizer = (-self._shape_param * tf.math.log(self._rate_param) +
                           tf.math.lgamma(self._shape_param))
    log_true_normalizer *= tf.cast(event_size, log_true_normalizer.dtype)

    ais_weights_size = tf.cast(tf.size(input=ais_weights), ais_weights.dtype)
    log_estimated_normalizer = (
        tf.reduce_logsumexp(input_tensor=ais_weights) -
        tf.math.log(ais_weights_size))

    ratio_estimate_true = tf.exp(ais_weights - log_true_normalizer)
    standard_error = tf.sqrt(
        _compute_sample_variance(ratio_estimate_true)
        / ais_weights_size)

    [
        ratio_estimate_true_,
        log_true_normalizer_,
        log_estimated_normalizer_,
        standard_error_,
        ais_weights_size_,
        event_size_,
    ] = self.evaluate([
        ratio_estimate_true,
        log_true_normalizer,
        log_estimated_normalizer,
        standard_error,
        ais_weights_size,
        event_size,
    ])

    tf.compat.v1.logging.vlog(
        1, '        log_true_normalizer: {}\n'
        '   log_estimated_normalizer: {}\n'
        '           ais_weights_size: {}\n'
        '                 event_size: {}\n'.format(
            log_true_normalizer_, log_estimated_normalizer_, ais_weights_size_,
            event_size_))
    self.assertNear(ratio_estimate_true_.mean(), 1., 4. * standard_error_)

  def _ais_gets_correct_log_normalizer_wrapper(self, independent_chain_ndims):
    """Tests that AIS yields reasonable estimates of normalizers."""
    initial_draws = np.random.normal(size=[30, 2, 1])
    x_ph = tf.compat.v1.placeholder_with_default(
        np.float32(initial_draws), shape=initial_draws.shape, name='x_ph')
    self._ais_gets_correct_log_normalizer(x_ph, independent_chain_ndims)

  def testAIS1(self):
    self._ais_gets_correct_log_normalizer_wrapper(1)

  def testAIS2(self):
    self._ais_gets_correct_log_normalizer_wrapper(2)

  def testAIS3(self):
    self._ais_gets_correct_log_normalizer_wrapper(3)

  def testSampleAIChainSeedReproducibleWorksCorrectly(self):
    independent_chain_ndims = 1
    x = np.random.rand(4, 3, 2)

    def proposal_log_prob(x):
      event_dims = tf.range(independent_chain_ndims, tf.rank(x))
      return -0.5 * tf.reduce_sum(
          input_tensor=x**2. + np.log(2 * np.pi), axis=event_dims)

    def target_log_prob(x):
      event_dims = tf.range(independent_chain_ndims, tf.rank(x))
      return self._log_gamma_log_prob(x, event_dims)

    def make_kernel(tlp_fn):
      return tfp.mcmc.HamiltonianMonteCarlo(
          target_log_prob_fn=tlp_fn,
          step_size=0.5,
          num_leapfrog_steps=2,
          seed=_maybe_seed(53))

    ais_kwargs = dict(
        num_steps=200,
        proposal_log_prob_fn=proposal_log_prob,
        target_log_prob_fn=target_log_prob,
        current_state=x,
        make_kernel_fn=make_kernel,
        parallel_iterations=1)

    _, ais_weights0, _ = tfp.mcmc.sample_annealed_importance_chain(
        **ais_kwargs)

    _, ais_weights1, _ = tfp.mcmc.sample_annealed_importance_chain(
        **ais_kwargs)

    ais_weights0_, ais_weights1_ = self.evaluate([
        ais_weights0, ais_weights1])

    self.assertAllClose(ais_weights0_, ais_weights1_,
                        atol=1e-5, rtol=1e-5)


if __name__ == '__main__':
  tf.test.main()
