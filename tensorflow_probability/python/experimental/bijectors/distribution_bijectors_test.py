# Copyright 2019 The TensorFlow Probability Authors.
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
"""Tests for distribution bijectors."""

from absl.testing import parameterized
import hypothesis as hp
from hypothesis import strategies as hps

import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.distributions import hypothesis_testlib as dhps
from tensorflow_probability.python.internal import hypothesis_testlib as tfp_hps
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import test_util

JAX_MODE = False

tfb = tfp.bijectors
tfd = tfp.distributions

PRECONDITIONING_FAILS_DISTS = (
    'Bates',  # CDF seems pretty crazy.
    'Deterministic',  # Inverse image is the empty set.
    'ExpRelaxedOneHotCategorical',  #  Bijector ldj error (maybe b/175354524).
    'GammaGamma',  #  Incorrect log prob (maybe b/175354524).
    'GeneralizedNormal',  # CDF gradient incorrect at 0.
    'HalfStudentT',  # Uses StudentT CDF.
    'Laplace',  # CDF gradient incorrect at 0.
    'LambertWNormal',  # CDF gradient incorrect at 0.
    'SigmoidBeta',  # inverse CDF numerical precision issues for large x
    'StudentT',  # CDF gradient incorrect at 0 (and unstable near zero).
    )

if JAX_MODE:
  PRECONDITIONING_FAILS_DISTS = (
      'VonMises',  # Abstract eval for 'von_mises_cdf_jvp' not implemented.
      ) + PRECONDITIONING_FAILS_DISTS


def _constrained_zeros_fn(shape, dtype, constraint_fn):
  """Generates dummy parameters initialized to a valid default value."""
  return hps.just(constraint_fn(tf.fill(shape, tf.cast(0., dtype))))


@test_util.test_graph_and_eager_modes
class DistributionBijectorsTest(test_util.TestCase):

  def assertDistributionIsApproximatelyStandardNormal(self,
                                                      dist,
                                                      logprob_atol=1e-2,
                                                      grad_atol=1e-2):
    """Verifies that dist's lps and gradients match those of Normal(0., 1.)."""
    event_ndims = ps.rank_from_shape(dist.event_shape_tensor, dist.event_shape)
    batch_ndims = ps.rank_from_shape(dist.batch_shape_tensor, dist.batch_shape)
    dist_shape = ps.concat([dist.batch_shape_tensor(),
                            dist.event_shape_tensor()], axis=0)
    reference_dist = tfd.Independent(
        tfd.Normal(loc=tf.zeros(dist_shape, dtype=dist.dtype), scale=1.),
        reinterpreted_batch_ndims=event_ndims)
    zs = tf.reshape([-4., -2., 0., 2., 4.],
                    ps.concat([[5],
                               ps.ones([batch_ndims + event_ndims],
                                       dtype=np.int32)],
                              axis=0))
    zs = tf.broadcast_to(zs, ps.concat([[5], dist_shape], axis=0))
    lp_dist, grad_dist = tfp.math.value_and_gradient(dist.log_prob, zs)
    lp_reference, grad_reference = tfp.math.value_and_gradient(
        reference_dist.log_prob, zs)
    self.assertAllClose(lp_reference, lp_dist, atol=logprob_atol)
    self.assertAllClose(grad_reference, grad_dist, atol=grad_atol)

  @parameterized.named_parameters(
      {'testcase_name': dname, 'dist_name': dname}
      for dname in sorted(dhps.INSTANTIABLE_BASE_DISTS.keys()))
  @test_util.numpy_disable_gradient_test
  @hp.given(hps.data())
  @tfp_hps.tfp_hp_settings(default_max_examples=5)
  def test_all_distributions_either_work_or_raise_error(self, dist_name, data):
    if not tf.executing_eagerly():
      self.skipTest('No need to test every distribution in graph mode.')
    if dist_name in PRECONDITIONING_FAILS_DISTS:
      self.skipTest('Known failure.')

    dist = data.draw(dhps.base_distributions(
        dist_name=dist_name,
        # TODO(b/175354524) fix autodiff for batch LDJs and enable batch tests.
        batch_shape=[],
        enable_vars=False,
        param_strategy_fn=_constrained_zeros_fn))
    try:
      b = tfp.experimental.bijectors.make_distribution_bijector(dist)
    except NotImplementedError:
      # Okay to fail as long as explicit error is raised.
      self.skipTest('Bijector not implemented.')
    self.assertDistributionIsApproximatelyStandardNormal(tfb.Invert(b)(dist))

  @test_util.numpy_disable_gradient_test
  def test_multivariate_normal(self):
    d = tfd.MultivariateNormalFullCovariance(loc=[4., 8.],
                                             covariance_matrix=[[11., 0.099],
                                                                [0.099, 0.1]])
    b = tfp.experimental.bijectors.make_distribution_bijector(d)
    self.assertDistributionIsApproximatelyStandardNormal(
        tfb.Invert(b)(d))

  @test_util.numpy_disable_gradient_test
  def test_nested_joint_distribution(self):

    def model():
      x = yield tfd.Normal(loc=-2., scale=1.)
      yield tfd.JointDistributionSequentialAutoBatched([
          tfd.Uniform(low=1. + tf.exp(x),
                      high=1 + tf.exp(x) + tf.nn.softplus(x)),
          lambda v: tfd.Exponential(v)])  # pylint: disable=unnecessary-lambda
    dist = tfd.JointDistributionCoroutineAutoBatched(model)
    samples = self.evaluate(dist.sample(10000, seed=test_util.test_seed()))

    b = tfp.experimental.bijectors.make_distribution_bijector(dist)
    whitened_samples = b.inverse(samples)
    x, (v, y) = whitened_samples
    whitened_vectors = tf.stack([x, v, y], axis=-1)
    whitened_mean = tf.reduce_mean(whitened_vectors, axis=0)
    self.assertAllClose(whitened_mean, tf.zeros_like(whitened_mean), atol=1e-1)
    whitened_cov = tfp.stats.covariance(
        whitened_vectors, sample_axis=0, event_axis=-1)
    self.assertAllClose(whitened_cov, tf.eye(3), atol=1e-1)

  @test_util.numpy_disable_gradient_test
  @test_util.jax_disable_test_missing_functionality(
      'build_asvi_surrogate_posterior')
  def test_mcmc_funnel_docstring_example_runs(self):

    # TODO(b/170865194): Use JDC here once sample_chain can take non-list state.
    model_with_funnel = tfd.JointDistributionSequentialAutoBatched([
        tfd.Normal(loc=-1., scale=2., name='z'),
        lambda z: tfd.Normal(loc=[0.], scale=tf.exp(z), name='x'),
        lambda x: tfd.Poisson(log_rate=x, name='y')])
    pinned_model = tfp.experimental.distributions.JointDistributionPinned(
        model_with_funnel, y=[1])
    surrogate_posterior = tfp.experimental.vi.build_asvi_surrogate_posterior(
        pinned_model)

    tfp.vi.fit_surrogate_posterior(
        pinned_model.unnormalized_log_prob,
        surrogate_posterior=surrogate_posterior,
        optimizer=tf.optimizers.Adam(0.01),
        sample_size=10,
        num_steps=1)
    bijector = (
        tfp.experimental.bijectors.make_distribution_bijector(
            surrogate_posterior))

    @tf.function(autograph=False)
    def do_sample():
      return tfp.mcmc.sample_chain(
          kernel=tfp.mcmc.DualAveragingStepSizeAdaptation(
              tfp.mcmc.TransformedTransitionKernel(
                  tfp.mcmc.NoUTurnSampler(
                      pinned_model.unnormalized_log_prob,
                      step_size=0.1),
                  bijector=bijector),
              num_adaptation_steps=5),
          current_state=surrogate_posterior.sample(),
          num_burnin_steps=5,
          trace_fn=lambda _0, _1: [],
          num_results=10)
    do_sample()

if __name__ == '__main__':
  tf.test.main()
