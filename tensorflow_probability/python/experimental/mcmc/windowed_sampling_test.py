# Copyright 2021 The TensorFlow Probability Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the _License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Tests for windowed sampling."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.experimental.mcmc import windowed_sampling
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.internal import unnest


tfb = tfp.bijectors
tfd = tfp.distributions
Root = tfd.JointDistributionCoroutine.Root

NUM_SCHOOLS = 8  # number of schools
TREATMENT_EFFECTS = [28., 8, -3, 7, -1, 1, 18, 12]
TREATMENT_STDDEVS = [15., 10, 16, 11, 9, 11, 10, 18]


def eight_schools_coroutine():

  @tfd.JointDistributionCoroutine
  def model():
    avg_effect = yield Root(tfd.Normal(0., 5., name='avg_effect'))
    avg_stddev = yield Root(tfd.HalfNormal(5., name='avg_stddev'))
    school_effects_std = yield Root(
        tfd.Sample(tfd.Normal(0., 1.), NUM_SCHOOLS, name='school_effects_std'))
    yield tfd.Independent(
        tfd.Normal(loc=(avg_effect[..., tf.newaxis] +
                        avg_stddev[..., tf.newaxis] * school_effects_std),
                   scale=tf.constant(TREATMENT_STDDEVS)),
        reinterpreted_batch_ndims=1,
        name='treatment_effects')
  return model


def eight_schools_sequential():
  model = tfd.JointDistributionSequential([
      tfd.Normal(0., 5., name='avg_effect'),
      tfd.HalfNormal(5., name='avg_stddev'),
      tfd.Sample(tfd.Normal(0., 1.), NUM_SCHOOLS, name='school_effects_std'),
      # pylint: disable=g-long-lambda
      lambda school_effects_std, avg_stddev, avg_effect: tfd.Independent(
          tfd.Normal(loc=(avg_effect[..., tf.newaxis] +
                          avg_stddev[..., tf.newaxis] * school_effects_std),
                     scale=tf.constant(TREATMENT_STDDEVS)),
          reinterpreted_batch_ndims=1,
          name='treatment_effects')])
  # pylint: enable=g-long-lambda
  return model


def eight_schools_named():
  model = tfd.JointDistributionNamed(
      dict(
          avg_effect=tfd.Normal(0., 5., name='avg_effect'),
          avg_stddev=tfd.HalfNormal(5., name='avg_stddev'),
          school_effects_std=tfd.Sample(
              tfd.Normal(0., 1.), NUM_SCHOOLS, name='school_effects_std'),
          # pylint: disable=g-long-lambda
          treatment_effects=lambda school_effects_std, avg_stddev, avg_effect:
          tfd.Independent(
              tfd.Normal(loc=(avg_effect[..., tf.newaxis] +
                              avg_stddev[..., tf.newaxis] * school_effects_std),
                         scale=tf.constant(TREATMENT_STDDEVS)),
              reinterpreted_batch_ndims=1,
              name='treatment_effects')))
  # pylint: enable=g-long-lambda
  return model


def eight_schools_nested():
  model = tfd.JointDistributionNamed(
      dict(
          effect_and_stddev=tfd.JointDistributionSequential([
              tfd.Normal(0., 5., name='avg_effect'),
              tfd.HalfNormal(5., name='avg_stddev')], name='effect_and_stddev'),
          school_effects_std=tfd.Sample(
              tfd.Normal(0., 1.), NUM_SCHOOLS, name='school_effects_std'),
          # pylint: disable=g-long-lambda
          treatment_effects=lambda school_effects_std, effect_and_stddev:
          tfd.Independent(
              tfd.Normal(loc=(effect_and_stddev[0][..., tf.newaxis] +
                              effect_and_stddev[1][..., tf.newaxis] *
                              school_effects_std),
                         scale=tf.constant(TREATMENT_STDDEVS)),
              reinterpreted_batch_ndims=1,
              name='treatment_effects')))
  # pylint: enable=g-long-lambda
  return model


def _gen_gaussian_updating_example(x_dim, y_dim, seed):
  """An implementation of section 2.3.3 from [1].

  We initialize a joint distribution

  x ~ N(mu, Lambda^{-1})
  y ~ N(Ax, L^{-1})

  Then condition the model on an observation for y. We can test to confirm that
  Cov(p(x | y_obs)) is near to

  Sigma = (Lambda + A^T L A)^{-1}

  This test can actually check whether the posterior samples have the proper
  covariance, and whether the windowed tuning recovers 1 / diag(Sigma) as the
  diagonal scaling factor.

  References:
  [1] Bishop, Christopher M. Pattern Recognition and Machine Learning.
      Springer, 2006.

  Args:
    x_dim: int
    y_dim: int
    seed: PRNG seed; see `tfp.random.sanitize_seed` for details.
  Returns:
    (tfd.JointDistribution, tf.Tensor), representing the joint distribution
    above, and the posterior variance.
  """
  seeds = samplers.split_seed(seed, 6)
  x_mean = samplers.normal((x_dim,), seed=seeds[0])
  x_scale_diag = samplers.normal((x_dim,), seed=seeds[1])
  y_scale_diag = samplers.normal((y_dim,), seed=seeds[2])
  scale_mat = samplers.normal((y_dim, x_dim), seed=seeds[3])
  y_shift = samplers.normal((y_dim,), seed=seeds[4])

  @tfd.JointDistributionCoroutine
  def model():
    x = yield Root(tfd.MultivariateNormalDiag(
        x_mean, scale_diag=x_scale_diag, name='x'))
    yield tfd.MultivariateNormalDiag(
        tf.linalg.matvec(scale_mat, x) + y_shift,
        scale_diag=y_scale_diag,
        name='y')

  dists, _ = model.sample_distributions(seed=seeds[5])
  precision_x = tf.linalg.inv(dists.x.covariance())
  precision_y = tf.linalg.inv(dists.y.covariance())
  true_cov = tf.linalg.inv(precision_x  +
                           tf.linalg.matmul(
                               tf.linalg.matmul(scale_mat, precision_y,
                                                transpose_a=True),
                               scale_mat))
  return model, tf.linalg.diag_part(true_cov)


@test_util.test_graph_and_eager_modes
class WindowedSamplingTest(test_util.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name='_' + fn.__name__, model_fn=fn) for fn in
      [eight_schools_coroutine, eight_schools_named, eight_schools_sequential,
       eight_schools_nested])
  def test_hmc_type_checks(self, model_fn):
    model = model_fn()
    pins = {'treatment_effects': tf.constant(TREATMENT_EFFECTS)}

    @tf.function(autograph=False)
    def do_sample(seed):
      return tfp.experimental.mcmc.windowed_adaptive_hmc(
          3, model, num_leapfrog_steps=2, num_adaptation_steps=21,
          seed=seed, **pins)

    draws, _ = do_sample(test_util.test_seed())
    self.evaluate(draws)

  @parameterized.named_parameters(
      dict(testcase_name='_' + fn.__name__, model_fn=fn) for fn in
      [eight_schools_coroutine, eight_schools_named, eight_schools_sequential,
       eight_schools_nested])
  def test_nuts_type_checks(self, model_fn):
    model = model_fn()
    pins = {'treatment_effects': tf.constant(TREATMENT_EFFECTS)}

    @tf.function
    def do_sample(seed):
      return tfp.experimental.mcmc.windowed_adaptive_nuts(
          3, model, max_tree_depth=2, num_adaptation_steps=50,
          seed=seed, **pins)

    draws, _ = do_sample(test_util.test_seed())
    self.evaluate(draws)

  # TODO(b/186878587) Figure out what's wrong and re-enable.
  def disabled_test_hmc_samples_well(self):
    model = eight_schools_named()
    pins = {'treatment_effects': tf.constant(TREATMENT_EFFECTS)}

    @tf.function
    def do_sample(seed):
      return tfp.experimental.mcmc.windowed_adaptive_hmc(
          400, model, num_leapfrog_steps=12, seed=seed,
          **pins)

    draws, _ = do_sample(test_util.test_seed())
    flat_draws = tf.nest.flatten(
        model.experimental_pin(**pins)._model_flatten(draws))
    max_scale_reduction = tf.reduce_max(
        tf.nest.map_structure(tf.reduce_max,
                              tfp.mcmc.potential_scale_reduction(flat_draws)))
    self.assertLess(self.evaluate(max_scale_reduction), 1.5)

  def test_nuts_samples_well(self):
    model = eight_schools_named()
    pins = {'treatment_effects': tf.constant(TREATMENT_EFFECTS)}

    @tf.function
    def do_sample():
      return tfp.experimental.mcmc.windowed_adaptive_nuts(
          200, model, max_tree_depth=5, seed=test_util.test_seed(),
          **pins)

    draws, _ = do_sample()
    flat_draws = tf.nest.flatten(
        model.experimental_pin(**pins)._model_flatten(draws))
    max_scale_reduction = tf.reduce_max(
        tf.nest.map_structure(tf.reduce_max,
                              tfp.mcmc.potential_scale_reduction(flat_draws)))
    self.assertLess(self.evaluate(max_scale_reduction), 1.05)

  @parameterized.named_parameters(
      dict(testcase_name=f'_{num_draws}', num_draws=num_draws) for num_draws in
      [0, 1, 525, 524, 100, 10000])
  def test_get_window_sizes(self, num_draws):
    [first_window,
     slow_window,
     last_window] = windowed_sampling._get_window_sizes(num_draws)
    self.assertEqual(first_window +
                     slow_window +
                     2 * slow_window +
                     4 * slow_window +
                     8 * slow_window +
                     last_window, num_draws)
    if num_draws == 525:
      self.assertEqual(slow_window, 25)
      self.assertEqual(first_window, 75)
      self.assertEqual(last_window, 75)

  def test_explicit_init(self):
    sample_dist = tfd.JointDistributionSequential(
        [tfd.HalfNormal(1., name=f'dist_{idx}') for idx in range(4)])

    explicit_init = [tf.ones(20) for _ in range(3)]
    _, init, bijector, _, _ = windowed_sampling._setup_mcmc(
        model=sample_dist,
        n_chains=20,
        init_position=explicit_init,
        seed=test_util.test_seed(),
        dist_3=1.)

    self.assertAllEqual(self.evaluate(init),
                        tf.convert_to_tensor(bijector(explicit_init)))

  def test_explicit_init_samples(self):
    stream = test_util.test_seed_stream()

    # Compute everything in a function so it is consistent in graph mode
    @tf.function
    def do_sample():
      jd_model = tfd.JointDistributionNamed({
          'x': tfd.HalfNormal(1.),
          'y': lambda x: tfd.Normal(0., x)})
      init = {'x': tf.ones(64)}
      return tfp.experimental.mcmc.windowed_adaptive_hmc(
          10,
          jd_model,
          num_adaptation_steps=200,
          current_state=init,
          num_leapfrog_steps=5,
          discard_tuning=False,
          y=tf.constant(1.),
          seed=stream(),
          trace_fn=None)

    self.evaluate(do_sample())

  def test_valid_init(self):

    class _HalfNormal(tfd.HalfNormal):

      def _default_event_space_bijector(self):
        # This bijector is intentionally mis-specified so that ~50% of
        # initialiations will fail.
        return tfb.Identity(validate_args=self.validate_args)

    tough_dist = tfd.JointDistributionSequential(
        [_HalfNormal(scale=1., name=f'dist_{idx}') for idx in range(4)])

    # Twenty chains with three parameters gives a 1 / 2^60 chance of
    # initializing with a finite log probability by chance.
    _, init, _, _, _ = windowed_sampling._setup_mcmc(
        model=tough_dist,
        n_chains=20,
        seed=test_util.test_seed(),
        dist_3=1.)

    self.assertAllGreater(self.evaluate(init), 0.)

  def test_extra_pins_not_required(self):
    model = tfd.JointDistributionSequential([
        tfd.Normal(0., 1., name='x'),
        lambda x: tfd.Normal(x, 1., name='y')
    ])
    pinned = model.experimental_pin(y=4.2)

    # No explicit pins are passed, since the model is already pinned.
    _, init, _, _, _ = windowed_sampling._setup_mcmc(
        model=pinned, n_chains=20,
        seed=test_util.test_seed())
    self.assertLen(init, 1)

  def test_hmc_fitting_gaussian(self):
    # See docstring to _gen_gaussian_updating_example
    x_dim = 3
    y_dim = 12

    stream = test_util.test_seed_stream()

    # Compute everything in a function so it is consistent in graph mode
    @tf.function
    def do_sample():
      jd_model, true_var = _gen_gaussian_updating_example(
          x_dim, y_dim, stream())
      y_val = jd_model.sample(seed=stream()).y
      _, trace = tfp.experimental.mcmc.windowed_adaptive_hmc(
          1,
          jd_model,
          num_adaptation_steps=525,
          num_leapfrog_steps=16,
          discard_tuning=False,
          y=y_val,
          seed=stream())

      # Get the final scaling used for the mass matrix - this is a measure
      # of how well the windowed adaptation recovered the true variance
      final_scaling = 1. / trace['variance_scaling'][0][-1, 0, :]
      return final_scaling, true_var
    final_scaling, true_var = do_sample()
    self.assertAllClose(true_var, final_scaling, rtol=0.15)

  def test_nuts_fitting_gaussian(self):
    # See docstring to _gen_gaussian_updating_example
    x_dim = 3
    y_dim = 12

    stream = test_util.test_seed_stream()

    # Compute everything in a function so it is consistent in graph mode
    @tf.function
    def do_sample():
      jd_model, true_var = _gen_gaussian_updating_example(
          x_dim, y_dim, stream())
      y_val = jd_model.sample(seed=stream()).y
      _, trace = tfp.experimental.mcmc.windowed_adaptive_nuts(
          1,
          jd_model,
          num_adaptation_steps=525,
          max_tree_depth=5,
          discard_tuning=False,
          y=y_val,
          seed=stream())

      # Get the final scaling used for the mass matrix - this is a measure
      # of how well the windowed adaptation recovered the true variance
      final_scaling = 1. / trace['variance_scaling'][0][-1, 0, :]
      return final_scaling, true_var
    final_scaling, true_var = do_sample()
    self.assertAllClose(true_var, final_scaling, rtol=0.1, atol=1e-3)

  def test_f64_step_size(self):
    dist = tfd.JointDistributionSequential([
        tfd.Normal(
            tf.constant(0., dtype=tf.float64),
            tf.constant(1., dtype=tf.float64))
    ])
    (target_log_prob_fn, initial_transformed_position, _, _, _
     ) = windowed_sampling._setup_mcmc(
         dist, n_chains=5, init_position=None, seed=test_util.test_seed())
    init_step_size = windowed_sampling._get_step_size(
        initial_transformed_position, target_log_prob_fn)
    self.assertDTypeEqual(init_step_size, np.float64)
    self.assertAllFinite(init_step_size)

  def test_batch_of_problems_autobatched(self):
    use_multinomial = tf.executing_eagerly()

    def model_fn():
      x = yield tfd.MultivariateNormalDiag(
          tf.zeros([10, 3]), tf.ones(3), name='x')
      if use_multinomial:
        # TODO(b/188215322): Use Multinomial in graph mode.
        yield tfd.Multinomial(
            logits=tfb.Pad([(0, 1)])(x), total_count=10, name='y')
      else:
        yield tfd.MultivariateNormalDiag(
            tfb.Pad([(0, 1)])(x), tf.ones(4), name='y')

    model = tfd.JointDistributionCoroutineAutoBatched(model_fn, batch_ndims=1)
    samp = model.sample(seed=test_util.test_seed())
    self.assertEqual((10, 3), samp.x.shape)
    self.assertEqual((10, 4), samp.y.shape)

    states, trace = self.evaluate(tfp.experimental.mcmc.windowed_adaptive_hmc(
        2, model.experimental_pin(y=samp.y), num_leapfrog_steps=3,
        num_adaptation_steps=100, init_step_size=tf.ones([10, 1]),
        seed=test_util.test_seed()))
    self.assertEqual((2, 64, 10, 3), states.x.shape)
    self.assertEqual((2, 10, 1), trace['step_size'].shape)

  def test_batch_of_problems_named(self):
    use_multinomial = tf.executing_eagerly()

    def mk_y(x):
      if use_multinomial:
        # TODO(b/188215322): Use Multinomial in graph mode.
        return tfd.Multinomial(logits=tfb.Pad([(0, 1)])(x), total_count=10)
      return tfd.MultivariateNormalDiag(tfb.Pad([(0, 1)])(x), tf.ones(4))

    model = tfd.JointDistributionNamed(dict(
        x=tfd.MultivariateNormalDiag(tf.zeros([10, 3]), tf.ones(3)),
        y=mk_y))

    samp = model.sample(seed=test_util.test_seed())
    self.assertEqual((10, 3), samp['x'].shape)
    self.assertEqual((10, 4), samp['y'].shape)

    states, trace = self.evaluate(
        tfp.experimental.mcmc.windowed_adaptive_hmc(
            2,
            model.experimental_pin(y=samp['y']),
            num_leapfrog_steps=3,
            num_adaptation_steps=100,
            init_step_size=tf.ones([10, 1]),
            seed=test_util.test_seed()))
    self.assertEqual((2, 64, 10, 3), states['x'].shape)
    self.assertEqual((2, 10, 1), trace['step_size'].shape)


@test_util.test_graph_and_eager_modes
class WindowedSamplingStepSizeTest(test_util.TestCase):

  def test_supply_full_step_size(self):
    stream = test_util.test_seed_stream()

    jd_model = tfd.JointDistributionNamed({
        'a': tfd.Normal(0., 1.),
        'b': tfd.MultivariateNormalDiag(
            loc=tf.zeros(3), scale_diag=tf.constant([1., 2., 3.]))
    })

    init_step_size = {'a': tf.reshape(tf.linspace(1., 2., 3), (3, 1)),
                      'b': tf.reshape(tf.linspace(1., 2., 9), (3, 3))}

    _, actual_step_size = tfp.experimental.mcmc.windowed_adaptive_hmc(
        1,
        jd_model,
        num_adaptation_steps=25,
        n_chains=3,
        init_step_size=init_step_size,
        num_leapfrog_steps=5,
        discard_tuning=False,
        trace_fn=lambda *args: unnest.get_innermost(args[-1], 'step_size'),
        seed=stream(),
    )

    # Gets a newaxis because step size needs to have an event dimension.
    self.assertAllCloseNested([init_step_size['a'],
                               init_step_size['b']],
                              [j[0] for j in actual_step_size])

  def test_supply_partial_step_size(self):
    stream = test_util.test_seed_stream()

    jd_model = tfd.JointDistributionNamed({
        'a': tfd.Normal(0., 1.),
        'b': tfd.MultivariateNormalDiag(
            loc=tf.zeros(3), scale_diag=tf.constant([1., 2., 3.]))
    })

    init_step_size = {'a': 1., 'b': 2.}
    _, actual_step_size = tfp.experimental.mcmc.windowed_adaptive_hmc(
        1,
        jd_model,
        num_adaptation_steps=25,
        n_chains=3,
        init_step_size=init_step_size,
        num_leapfrog_steps=5,
        discard_tuning=False,
        trace_fn=lambda *args: unnest.get_innermost(args[-1], 'step_size'),
        seed=stream(),
    )

    actual_step = [j[0] for j in actual_step_size]
    expected_step = [1., 2.]
    self.assertAllCloseNested(expected_step, actual_step)

  def test_supply_single_step_size(self):
    stream = test_util.test_seed_stream()

    jd_model = tfd.JointDistributionNamed({
        'a': tfd.Normal(0., 1.),
        'b': tfd.MultivariateNormalDiag(
            loc=tf.zeros(3), scale_diag=tf.constant([1., 2., 3.]))
    })

    init_step_size = 1.
    _, traced_step_size = self.evaluate(
        tfp.experimental.mcmc.windowed_adaptive_hmc(
            1,
            jd_model,
            num_adaptation_steps=25,
            n_chains=20,
            init_step_size=init_step_size,
            num_leapfrog_steps=5,
            discard_tuning=False,
            trace_fn=lambda *args: unnest.get_innermost(args[-1], 'step_size'),
            seed=stream()))

    self.assertEqual((25 + 1,), traced_step_size.shape)
    self.assertAllClose(1., traced_step_size[0])

  def test_sequential_step_size(self):
    stream = test_util.test_seed_stream()

    jd_model = tfd.JointDistributionSequential(
        [tfd.HalfNormal(scale=1., name=f'dist_{idx}') for idx in range(4)])
    init_step_size = [1., 2., 3.]
    _, actual_step_size = tfp.experimental.mcmc.windowed_adaptive_nuts(
        1,
        jd_model,
        num_adaptation_steps=25,
        n_chains=3,
        init_step_size=init_step_size,
        discard_tuning=False,
        trace_fn=lambda *args: unnest.get_innermost(args[-1], 'step_size'),
        dist_3=tf.constant(1.),
        seed=stream(),
    )

    self.assertAllCloseNested(init_step_size,
                              [j[0] for j in actual_step_size])


def _beta_binomial(trials):
  """Returns a function that constructs a beta binomial distribution."""

  def _beta_binomial_distribution(mean, inverse_concentration):
    """Returns a beta binomial distribution with the given parameters."""
    # Mean and inverse concentration are broadcast across days.
    mean = mean[..., tf.newaxis]
    inverse_concentration = inverse_concentration[..., tf.newaxis]

    beta_binomial = tfd.BetaBinomial(
        total_count=trials,
        concentration0=(1 - mean) / inverse_concentration,
        concentration1=mean / inverse_concentration)
    return tfd.Independent(beta_binomial, reinterpreted_batch_ndims=2)

  return _beta_binomial_distribution


def get_joint_distribution(
    trials,
    mean_prior=lambda: tfd.Uniform(0., 1.),
    inverse_concentration_prior=lambda: tfd.HalfNormal(5.)):
  """Returns a joint distribution over parameters and successes."""
  param_shape = ps.shape(trials)[:1]
  mean = tfd.Sample(mean_prior(), param_shape)
  inverse_concentration = tfd.Sample(inverse_concentration_prior(), param_shape)
  return tfd.JointDistributionNamed(
      dict(mean=mean,
           inverse_concentration=inverse_concentration,
           successes=_beta_binomial(trials)),
      name='jd')


@test_util.disable_test_for_backend(disable_jax=True,
                                    reason='Only applies to TF')
class PrecompiledTest(test_util.TestCase):

  def setUp(self):
    super().setUp()
    arms = 2
    days = 3

    strm = test_util.test_seed_stream()
    self.trials = tfd.Poisson(100.).sample([arms, days], seed=strm())
    dist = get_joint_distribution(self.trials)
    self.true_values = dist.sample(seed=strm())

  def nuts_kwargs(self):
    return {'max_tree_depth': 2}

  def hmc_kwargs(self):
    return {'num_leapfrog_steps': 3, 'store_parameters_in_results': True}

  @parameterized.named_parameters(('hmc_jit_sig', 'hmc'),
                                  ('nuts_jit_sig', 'nuts'))
  def test_base_kernel(self, kind):
    self.skip_if_no_xla()

    input_signature = (
        tf.TensorSpec(
            shape=[None, None], dtype=tf.float32, name='trials'),
        tf.TensorSpec(
            shape=[None, None], dtype=tf.float32, name='successes'),
        tf.TensorSpec(
            shape=[2], dtype=tf.int32, name='seed'))
    @tf.function(jit_compile=True, input_signature=input_signature)
    def do(trials, successes, seed):
      if kind == 'hmc':
        proposal_kernel_kwargs = self.hmc_kwargs()
      else:
        proposal_kernel_kwargs = self.nuts_kwargs()

      return windowed_sampling._windowed_adaptive_impl(
          n_draws=9,
          joint_dist=get_joint_distribution(trials),
          kind=kind,
          n_chains=11,
          proposal_kernel_kwargs=proposal_kernel_kwargs,
          num_adaptation_steps=50,
          current_state=None,
          dual_averaging_kwargs={'target_accept_prob': 0.76},
          trace_fn=None,
          return_final_kernel_results=False,
          discard_tuning=True,
          seed=seed,
          successes=successes)

    self.evaluate(do(self.trials + 0., self.true_values['successes'],
                     test_util.test_seed(sampler_type='stateless')))


if __name__ == '__main__':
  tf.test.main()
