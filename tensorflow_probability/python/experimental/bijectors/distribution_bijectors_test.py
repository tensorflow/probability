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
      'PERT',  # Testing triggers second derivative path in JAX mode.
      'VonMises',  # Abstract eval for 'von_mises_cdf_jvp' not implemented.
  ) + PRECONDITIONING_FAILS_DISTS


def _constrained_zeros_fn(shape, dtype, constraint_fn):
  """Generates dummy parameters initialized to a valid default value."""
  return hps.just(constraint_fn(tf.fill(shape, tf.cast(0., dtype))))


@test_util.test_graph_and_eager_modes
class DistributionBijectorsTest(test_util.TestCase):

  def assertDistributionIsApproximatelyStandardNormal(self,
                                                      dist,
                                                      rtol=1e-6,
                                                      logprob_atol=1e-2,
                                                      grad_atol=1e-2):
    """Verifies that dist's lps and gradients match those of Normal(0., 1.)."""
    batch_shape = dist.batch_shape_tensor()

    def make_reference_values(event_shape):
      dist_shape = ps.concat([batch_shape, event_shape], axis=0)
      x = tf.reshape([-4., -2., 0., 2., 4.],
                     ps.concat([[5], ps.ones_like(dist_shape)], axis=0))
      return tf.broadcast_to(x, ps.concat([[5], dist_shape], axis=0))

    flat_event_shape = tf.nest.flatten(dist.event_shape_tensor())
    zs = [make_reference_values(s) for s in flat_event_shape]
    lp_dist, grad_dist = tfp.math.value_and_gradient(
        lambda *xs: dist.log_prob(tf.nest.pack_sequence_as(dist.dtype, xs)), zs)

    def reference_value_and_gradient(z, event_shape):
      reference_dist = tfd.Independent(
          tfd.Normal(loc=tf.zeros_like(z), scale=1.),
          reinterpreted_batch_ndims=ps.rank_from_shape(event_shape))
      return tfp.math.value_and_gradient(reference_dist.log_prob, z)

    reference_vals_and_grads = [
        reference_value_and_gradient(z, event_shape)
        for (z, event_shape) in zip(zs, flat_event_shape)]

    lps_reference = [lp for lp, grad in reference_vals_and_grads]
    self.assertAllClose(
        sum(lps_reference), lp_dist, rtol=rtol, atol=logprob_atol)

    grads_reference = [grad for lp, grad in reference_vals_and_grads]
    self.assertAllCloseNested(
        grads_reference, grad_dist, rtol=rtol, atol=grad_atol)

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

    dist = data.draw(
        dhps.base_distributions(
            dist_name=dist_name,
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
    d = tfd.MultivariateNormalTriL(
        loc=[4., 8.],
        scale_tril=tf.linalg.cholesky([[11., 0.099], [0.099, 0.1]]))
    b = tfp.experimental.bijectors.make_distribution_bijector(d)
    self.assertDistributionIsApproximatelyStandardNormal(tfb.Invert(b)(d))

  @test_util.numpy_disable_gradient_test
  def test_markov_chain(self):
    d = tfd.MarkovChain(
        initial_state_prior=tfd.Uniform(low=0., high=1.),
        transition_fn=lambda _, x: tfd.Uniform(low=0., high=tf.nn.softplus(x)),
        num_steps=3)
    b = tfp.experimental.bijectors.make_distribution_bijector(d)
    self.assertDistributionIsApproximatelyStandardNormal(
        tfb.Invert(b)(d), rtol=1e-4)

  @test_util.numpy_disable_gradient_test
  def test_markov_chain_joint(self):
    d = tfd.MarkovChain(
        initial_state_prior=tfd.JointDistributionSequentialAutoBatched(
            [tfd.Normal(0., 1.),
             lambda x: tfd.Uniform(low=0., high=tf.exp(x))]),
        transition_fn=(
            lambda _, state: tfd.JointDistributionSequentialAutoBatched(  # pylint: disable=g-long-lambda
                batch_ndims=ps.rank(state[1]),
                model=[tfd.Normal(state[1], 1.),
                       lambda x: tfd.Uniform(low=0., high=tf.nn.softplus(x))])),
        num_steps=10)
    b = tfp.experimental.bijectors.make_distribution_bijector(d)
    self.assertDistributionIsApproximatelyStandardNormal(
        tfb.Invert(b)(d), rtol=1e-4)

  @test_util.numpy_disable_gradient_test
  def test_nested_joint_distribution(self):

    def model():
      x = yield tfd.Normal(loc=-2., scale=1.)
      yield tfd.JointDistributionSequentialAutoBatched([
          tfd.Uniform(low=1. - tf.exp(x),
                      high=2. + tf.exp(x) + tf.nn.softplus(x)),
          lambda v: tfd.Exponential(v)])  # pylint: disable=unnecessary-lambda

    dist = tfd.JointDistributionCoroutineAutoBatched(model)
    b = tfp.experimental.bijectors.make_distribution_bijector(dist)
    self.assertDistributionIsApproximatelyStandardNormal(
        tfb.Invert(b)(dist), rtol=1e-4)

  @test_util.numpy_disable_gradient_test
  @test_util.jax_disable_test_missing_functionality(
      'build_asvi_surrogate_posterior')
  def test_mcmc_funnel_docstring_example_runs(self):

    @tfd.JointDistributionCoroutineAutoBatched
    def model_with_funnel():
      z = yield tfd.Normal(loc=-1., scale=2., name='z')
      x = yield tfd.Normal(loc=[0.], scale=tf.exp(z), name='x')
      yield tfd.Poisson(log_rate=x, name='y')

    pinned_model = model_with_funnel.experimental_pin(y=[1])
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
                      pinned_model.unnormalized_log_prob, step_size=0.1),
                  bijector=bijector),
              num_adaptation_steps=5),
          current_state=surrogate_posterior.sample(),
          num_burnin_steps=5,
          trace_fn=lambda _0, _1: [],
          num_results=10)

    do_sample()


if __name__ == '__main__':
  test_util.main()
