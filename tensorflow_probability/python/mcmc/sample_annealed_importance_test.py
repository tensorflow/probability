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

import collections

# Dependency imports

from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
from tensorflow_probability.python.bijectors import identity
from tensorflow_probability.python.distributions import normal
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.mcmc import hmc
from tensorflow_probability.python.mcmc import sample_annealed_importance as sai
from tensorflow_probability.python.mcmc import transformed_kernel


JAX_MODE = False


def _maybe_seed(s, sampler_type='stateful'):
  if tf.executing_eagerly() and not JAX_MODE and sampler_type != 'stateless':
    tf.random.set_seed(s)
    return None
  return s


@test_util.test_all_tf_execution_regimes
class SampleAnnealedImportanceTest(test_util.TestCase):

  def setUp(self):  # pylint: disable=g-missing-super-call
    self._shape_param = 5.
    self._rate_param = 10.

  def _log_gamma_log_prob(self, x, event_dims=()):
    """Computes unnormalized log-pdf of a log-gamma random variable.

    Args:
      x: Value of the random variable.
      event_dims: Dimensions not to treat as independent.

    Returns:
      log_prob: The log-pdf up to a normalizing constant.
    """
    return tf.reduce_sum(
        self._shape_param * x - self._rate_param * tf.math.exp(x),
        axis=event_dims)

  # TODO(b/74154679): Create Fake TransitionKernel and not rely on HMC.

  def _ais_gets_correct_log_normalizer(self, init, independent_chain_ndims,
                                       use_transformed_kernel=False):
    counter = collections.Counter()

    def proposal_log_prob(x):
      counter['proposal_calls'] += 1
      event_dims = ps.range(independent_chain_ndims, ps.rank(x))
      return tf.reduce_sum(
          normal.Normal(loc=0., scale=1.).log_prob(x), axis=event_dims)

    def target_log_prob(x):
      counter['target_calls'] += 1
      event_dims = ps.range(independent_chain_ndims, ps.rank(x))
      return self._log_gamma_log_prob(x, event_dims)

    num_steps = 200

    if use_transformed_kernel:
      def make_kernel(tlp_fn):
        return transformed_kernel.TransformedTransitionKernel(
            inner_kernel=hmc.HamiltonianMonteCarlo(
                target_log_prob_fn=tlp_fn, step_size=0.5, num_leapfrog_steps=2),
            bijector=identity.Identity())
    else:
      def make_kernel(tlp_fn):
        return hmc.HamiltonianMonteCarlo(
            target_log_prob_fn=tlp_fn, step_size=0.5, num_leapfrog_steps=2)

    _, ais_weights, _ = sai.sample_annealed_importance_chain(
        num_steps=num_steps,
        proposal_log_prob_fn=proposal_log_prob,
        target_log_prob_fn=target_log_prob,
        current_state=init,
        make_kernel_fn=make_kernel,
        seed=test_util.test_seed())

    # We have three calls because the calculation of `ais_weights` entails
    # another call to the `convex_combined_log_prob_fn`. We could refactor
    # things to avoid this, if needed (eg, b/72994218).
    if not tf.executing_eagerly():
      self.assertAllEqual(dict(target_calls=3, proposal_calls=3), counter)

    event_shape = tf.shape(init)[independent_chain_ndims:]
    event_size = tf.reduce_prod(event_shape)

    log_true_normalizer = (-self._shape_param * tf.math.log(self._rate_param) +
                           tf.math.lgamma(self._shape_param))
    log_true_normalizer *= tf.cast(event_size, log_true_normalizer.dtype)

    ais_weights_size = tf.cast(tf.size(ais_weights), ais_weights.dtype)
    log_estimated_normalizer = (
        tf.reduce_logsumexp(ais_weights) -
        tf.math.log(ais_weights_size))

    ratio_estimate_true = tf.math.exp(ais_weights - log_true_normalizer)
    standard_error = tf.sqrt(
        tf.math.reduce_variance(ratio_estimate_true) / ais_weights_size)

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

    tf1.logging.vlog(
        1, '        log_true_normalizer: {}\n'
        '   log_estimated_normalizer: {}\n'
        '           ais_weights_size: {}\n'
        '                 event_size: {}\n'.format(
            log_true_normalizer_, log_estimated_normalizer_, ais_weights_size_,
            event_size_))
    self.assertNear(ratio_estimate_true_.mean(), 1., 4. * standard_error_)

  def _ais_gets_correct_log_normalizer_wrapper(self, independent_chain_ndims,
                                               use_transformed_kernel=False):
    """Tests that AIS yields reasonable estimates of normalizers."""
    initial_draws = np.random.normal(size=[30, 2, 1])
    x_ph = tf1.placeholder_with_default(
        np.float32(initial_draws), shape=initial_draws.shape, name='x_ph')
    self._ais_gets_correct_log_normalizer(
        x_ph, independent_chain_ndims,
        use_transformed_kernel=use_transformed_kernel)

  def testAIS1(self):
    self._ais_gets_correct_log_normalizer_wrapper(1)

  def testAIS2(self):
    self._ais_gets_correct_log_normalizer_wrapper(2)

  def testAIS3(self):
    self._ais_gets_correct_log_normalizer_wrapper(3)

  def testAISWithTransformedKernel(self):
    self._ais_gets_correct_log_normalizer_wrapper(
        1, use_transformed_kernel=True)

  @parameterized.named_parameters(
      dict(testcase_name='_stateless', sampler_type='stateless'),
      dict(testcase_name='_stateful', sampler_type='stateful'))
  def testSampleAIChainSeedReproducibleWorksCorrectly(self, sampler_type):
    independent_chain_ndims = 1
    x = np.random.rand(4, 3, 2)

    def proposal_log_prob(x):
      event_dims = ps.range(independent_chain_ndims, ps.rank(x))
      return -0.5 * tf.reduce_sum(x**2. + np.log(2 * np.pi), axis=event_dims)

    def target_log_prob(x):
      event_dims = ps.range(independent_chain_ndims, ps.rank(x))
      return self._log_gamma_log_prob(x, event_dims)

    seed = test_util.test_seed()
    def make_kernel(tlp_fn):
      return hmc.HamiltonianMonteCarlo(
          target_log_prob_fn=tlp_fn, step_size=0.5, num_leapfrog_steps=2)

    ais_kwargs = dict(
        num_steps=200,
        proposal_log_prob_fn=proposal_log_prob,
        target_log_prob_fn=target_log_prob,
        current_state=x,
        make_kernel_fn=make_kernel)

    seed = test_util.test_seed(sampler_type=sampler_type)
    _, ais_weights0, _ = sai.sample_annealed_importance_chain(
        seed=_maybe_seed(seed, sampler_type=sampler_type), **ais_kwargs)

    _, ais_weights1, _ = sai.sample_annealed_importance_chain(
        seed=_maybe_seed(seed, sampler_type=sampler_type), **ais_kwargs)

    ais_weights0_, ais_weights1_ = self.evaluate([
        ais_weights0, ais_weights1])

    self.assertAllClose(ais_weights0_, ais_weights1_,
                        atol=1e-5, rtol=1e-5)


if __name__ == '__main__':
  test_util.main()
