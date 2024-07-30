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

from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow_probability.python.bijectors import identity
from tensorflow_probability.python.bijectors import pad
from tensorflow_probability.python.distributions import autoregressive
from tensorflow_probability.python.distributions import beta_binomial
from tensorflow_probability.python.distributions import dirichlet
from tensorflow_probability.python.distributions import half_normal
from tensorflow_probability.python.distributions import independent
from tensorflow_probability.python.distributions import joint_distribution_auto_batched as jdab
from tensorflow_probability.python.distributions import joint_distribution_coroutine as jdc
from tensorflow_probability.python.distributions import joint_distribution_named as jdn
from tensorflow_probability.python.distributions import joint_distribution_sequential as jds
from tensorflow_probability.python.distributions import multinomial
from tensorflow_probability.python.distributions import mvn_diag
from tensorflow_probability.python.distributions import normal
from tensorflow_probability.python.distributions import poisson
from tensorflow_probability.python.distributions import sample
from tensorflow_probability.python.distributions import uniform
from tensorflow_probability.python.experimental import distribute
from tensorflow_probability.python.experimental.mcmc import windowed_sampling
from tensorflow_probability.python.internal import callable_util
from tensorflow_probability.python.internal import distribute_test_lib
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.internal import unnest
from tensorflow_probability.python.mcmc import diagnostic

JAX_MODE = False

Root = jdc.JointDistributionCoroutine.Root

NUM_SCHOOLS = 8  # number of schools
TREATMENT_EFFECTS = [28., 8, -3, 7, -1, 1, 18, 12]
TREATMENT_STDDEVS = [15., 10, 16, 11, 9, 11, 10, 18]


def eight_schools_coroutine():

  @jdc.JointDistributionCoroutine
  def model():
    avg_effect = yield Root(normal.Normal(0., 5., name='avg_effect'))
    avg_stddev = yield Root(half_normal.HalfNormal(5., name='avg_stddev'))
    school_effects_std = yield Root(
        sample.Sample(
            normal.Normal(0., 1.), NUM_SCHOOLS, name='school_effects_std'))
    yield independent.Independent(
        normal.Normal(
            loc=(avg_effect[..., tf.newaxis] +
                 avg_stddev[..., tf.newaxis] * school_effects_std),
            scale=tf.constant(TREATMENT_STDDEVS)),
        reinterpreted_batch_ndims=1,
        name='treatment_effects')
  return model


def eight_schools_sequential():
  model = jds.JointDistributionSequential([
      normal.Normal(0., 5., name='avg_effect'),
      half_normal.HalfNormal(5., name='avg_stddev'),
      sample.Sample(
          normal.Normal(0., 1.), NUM_SCHOOLS, name='school_effects_std'),
      # pylint: disable=g-long-lambda
      lambda school_effects_std, avg_stddev, avg_effect: independent.
      Independent(
          normal.Normal(
              loc=(avg_effect[..., tf.newaxis] + avg_stddev[..., tf.newaxis] *
                   school_effects_std),
              scale=tf.constant(TREATMENT_STDDEVS)),
          reinterpreted_batch_ndims=1,
          name='treatment_effects')
  ])
  # pylint: enable=g-long-lambda
  return model


def eight_schools_named():
  model = jdn.JointDistributionNamed(
      dict(
          avg_effect=normal.Normal(0., 5., name='avg_effect'),
          avg_stddev=half_normal.HalfNormal(5., name='avg_stddev'),
          school_effects_std=sample.Sample(
              normal.Normal(0., 1.), NUM_SCHOOLS, name='school_effects_std'),
          # pylint: disable=g-long-lambda
          treatment_effects=lambda school_effects_std, avg_stddev, avg_effect:
          independent.Independent(
              normal.Normal(
                  loc=(avg_effect[..., tf.newaxis] + avg_stddev[..., tf.newaxis]
                       * school_effects_std),
                  scale=tf.constant(TREATMENT_STDDEVS)),
              reinterpreted_batch_ndims=1,
              name='treatment_effects')))
  # pylint: enable=g-long-lambda
  return model


def eight_schools_nested():
  model = jdn.JointDistributionNamed(
      dict(
          effect_and_stddev=jds.JointDistributionSequential(
              [
                  normal.Normal(0., 5., name='avg_effect'),
                  half_normal.HalfNormal(5., name='avg_stddev')
              ],
              name='effect_and_stddev'),
          school_effects_std=sample.Sample(
              normal.Normal(0., 1.), NUM_SCHOOLS, name='school_effects_std'),
          # pylint: disable=g-long-lambda
          treatment_effects=lambda school_effects_std, effect_and_stddev:
          independent.Independent(
              normal.Normal(
                  loc=(effect_and_stddev[0][
                      ..., tf.newaxis] + effect_and_stddev[1][..., tf.newaxis] *
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

  @jdc.JointDistributionCoroutine
  def model():
    x = yield Root(
        mvn_diag.MultivariateNormalDiag(
            x_mean, scale_diag=x_scale_diag, name='x'))
    yield mvn_diag.MultivariateNormalDiag(
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

    def do_sample(seed):
      return windowed_sampling.windowed_adaptive_hmc(
          3,
          model,
          num_leapfrog_steps=2,
          num_adaptation_steps=2,
          seed=seed,
          **pins)

    draws, _ = do_sample(test_util.test_seed())
    self.evaluate(draws)

  @parameterized.named_parameters(
      dict(testcase_name='_' + fn.__name__, model_fn=fn) for fn in
      [eight_schools_coroutine, eight_schools_named, eight_schools_sequential,
       eight_schools_nested])
  def test_nuts_type_checks(self, model_fn):
    model = model_fn()
    pins = {'treatment_effects': tf.constant(TREATMENT_EFFECTS)}

    def do_sample(seed):
      return windowed_sampling.windowed_adaptive_nuts(
          3,
          model,
          max_tree_depth=2,
          num_adaptation_steps=3,
          seed=seed,
          **pins)

    draws, _ = do_sample(test_util.test_seed())
    self.evaluate(draws)

  def test_hmc_samples_well(self):
    if tf.executing_eagerly() and not JAX_MODE:
      self.skipTest('Eager test is slow.')
    model = eight_schools_named()
    pins = {'treatment_effects': tf.constant(TREATMENT_EFFECTS)}

    @tf.function
    def do_sample(seed):
      return windowed_sampling.windowed_adaptive_hmc(
          400, model, num_leapfrog_steps=12, seed=seed, **pins)

    draws, _ = do_sample(test_util.test_seed())
    flat_draws = tf.nest.flatten(
        model.experimental_pin(**pins)._model_flatten(draws))
    max_scale_reduction = tf.reduce_max(
        tf.nest.map_structure(tf.reduce_max,
                              diagnostic.potential_scale_reduction(flat_draws)))
    self.assertLess(self.evaluate(max_scale_reduction), 1.5)

  def test_nuts_samples_well(self):
    if tf.executing_eagerly() and not JAX_MODE:
      self.skipTest('Eager test is slow.')
    model = eight_schools_named()
    pins = {'treatment_effects': tf.constant(TREATMENT_EFFECTS)}

    @tf.function
    def do_sample():
      return windowed_sampling.windowed_adaptive_nuts(
          200, model, max_tree_depth=5, seed=test_util.test_seed(), **pins)

    draws, _ = do_sample()
    flat_draws = tf.nest.flatten(
        model.experimental_pin(**pins)._model_flatten(draws))
    max_scale_reduction = tf.reduce_max(
        tf.nest.map_structure(tf.reduce_max,
                              diagnostic.potential_scale_reduction(flat_draws)))
    self.assertLess(self.evaluate(max_scale_reduction), 1.05)

  @parameterized.named_parameters(
      dict(testcase_name=f'_{num_draws}', num_draws=num_draws)
      for num_draws in [0, 1, 500, 499, 100, 10000])
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
    if num_draws == 500:
      self.assertEqual(slow_window, 25)
      self.assertEqual(first_window, 75)
      self.assertEqual(last_window, 50)

  def test_explicit_init(self):
    sample_dist = jds.JointDistributionSequential(
        [half_normal.HalfNormal(1., name=f'dist_{idx}') for idx in range(4)])

    explicit_init = [tf.ones(20) for _ in range(3)]
    _, init, bijector, _, _, _ = windowed_sampling._setup_mcmc(
        model=sample_dist,
        n_chains=[20],
        init_position=explicit_init,
        seed=test_util.test_seed(),
        dist_3=1.)

    self.assertAllEqual(self.evaluate(init),
                        tf.convert_to_tensor(bijector(explicit_init)))

  def test_explicit_init_samples(self):
    if tf.executing_eagerly() and not JAX_MODE:
      self.skipTest('Eager test is slow.')
    stream = test_util.test_seed_stream()

    # Compute everything in a function so it is consistent in graph mode
    @tf.function
    def do_sample():
      jd_model = jdn.JointDistributionNamed({
          'x': half_normal.HalfNormal(1.),
          'y': lambda x: normal.Normal(0., x)
      })
      init = {'x': tf.ones(64)}
      return windowed_sampling.windowed_adaptive_hmc(
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

    class _HalfNormal(half_normal.HalfNormal):

      def _default_event_space_bijector(self):
        # This bijector is intentionally mis-specified so that ~50% of
        # initialiations will fail.
        return identity.Identity(validate_args=self.validate_args)

    tough_dist = jds.JointDistributionSequential(
        [_HalfNormal(scale=1., name=f'dist_{idx}') for idx in range(4)])

    # Twenty chains with three parameters gives a 1 / 2^60 chance of
    # initializing with a finite log probability by chance.
    _, init, _, _, _, _ = windowed_sampling._setup_mcmc(
        model=tough_dist,
        n_chains=[20],
        seed=test_util.test_seed(),
        dist_3=1.)

    self.assertAllGreater(self.evaluate(init), 0.)

  def test_extra_pins_not_required(self):
    model = jds.JointDistributionSequential([
        normal.Normal(0., 1., name='x'),
        lambda x: normal.Normal(x, 1., name='y')
    ])
    pinned = model.experimental_pin(y=4.2)

    # No explicit pins are passed, since the model is already pinned.
    _, init, _, _, _, _ = windowed_sampling._setup_mcmc(
        model=pinned, n_chains=[20],
        seed=test_util.test_seed())
    self.assertLen(init, 1)

  def test_hmc_fitting_gaussian(self):
    if tf.executing_eagerly() and not JAX_MODE:
      self.skipTest('Eager test is slow.')
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
      _, trace = windowed_sampling.windowed_adaptive_hmc(
          1,
          jd_model,
          n_chains=1,
          num_adaptation_steps=10000,
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
    if tf.executing_eagerly() and not JAX_MODE:
      self.skipTest('Eager test is slow.')
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
      _, trace = windowed_sampling.windowed_adaptive_nuts(
          1,
          jd_model,
          n_chains=1,
          num_adaptation_steps=10000,
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
    dist = jds.JointDistributionSequential([
        normal.Normal(
            tf.constant(0., dtype=tf.float64),
            tf.constant(1., dtype=tf.float64))
    ])
    (target_log_prob_fn, initial_transformed_position, _, _, _, _
     ) = windowed_sampling._setup_mcmc(
         dist, n_chains=[5], init_position=None, seed=test_util.test_seed())
    init_step_size = windowed_sampling._get_step_size(
        initial_transformed_position, target_log_prob_fn)
    self.assertDTypeEqual(init_step_size, np.float64)
    self.assertAllFinite(init_step_size)

  def test_batch_of_problems_autobatched(self):

    def model_fn():
      x = yield mvn_diag.MultivariateNormalDiag(
          tf.zeros([10, 3]), tf.ones(3), name='x')
      yield multinomial.Multinomial(
          logits=pad.Pad([(0, 1)])(x), total_count=10, name='y')

    model = jdab.JointDistributionCoroutineAutoBatched(model_fn, batch_ndims=1)
    samp = model.sample(seed=test_util.test_seed())
    self.assertEqual((10, 3), samp.x.shape)
    self.assertEqual((10, 4), samp.y.shape)

    states, trace = self.evaluate(
        windowed_sampling.windowed_adaptive_hmc(
            2,
            model.experimental_pin(y=samp.y),
            num_leapfrog_steps=3,
            num_adaptation_steps=21,
            init_step_size=tf.ones([10, 1]),
            seed=test_util.test_seed()))
    self.assertEqual((2, 64, 10, 3), states.x.shape)
    self.assertEqual((2, 10, 1), trace['step_size'].shape)

  def test_batch_of_problems_named(self):

    def mk_y(x):
      return multinomial.Multinomial(
          logits=pad.Pad([(0, 1)])(x), total_count=10)

    model = jdn.JointDistributionNamed(
        dict(
            x=mvn_diag.MultivariateNormalDiag(tf.zeros([10, 3]), tf.ones(3)),
            y=mk_y))

    samp = model.sample(seed=test_util.test_seed())
    self.assertEqual((10, 3), samp['x'].shape)
    self.assertEqual((10, 4), samp['y'].shape)

    states, trace = self.evaluate(
        windowed_sampling.windowed_adaptive_hmc(
            2,
            model.experimental_pin(y=samp['y']),
            num_leapfrog_steps=3,
            num_adaptation_steps=21,
            init_step_size=tf.ones([10, 1]),
            seed=test_util.test_seed()))
    self.assertEqual((2, 64, 10, 3), states['x'].shape)
    self.assertEqual((2, 10, 1), trace['step_size'].shape)

  def test_bijector(self):
    dist = jds.JointDistributionSequential([dirichlet.Dirichlet(tf.ones(2))])
    bij, _ = windowed_sampling._get_flat_unconstraining_bijector(dist)
    draw = dist.sample(seed=test_util.test_seed())
    self.assertAllCloseNested(bij.inverse(bij(draw)), draw)

  @parameterized.named_parameters(*(
      (f'{kind}_{n_chains}', kind, n_chains)  # pylint: disable=g-complex-comprehension
      for kind in ('hmc', 'nuts') for n_chains in ([], 3, [2, 1], [2, 2, 2])))
  def test_batches_of_chains(self, kind, n_chains):

    def model_fn():
      x = yield mvn_diag.MultivariateNormalDiag(
          tf.zeros(3), tf.ones(3), name='x')
      yield multinomial.Multinomial(
          logits=pad.Pad([(0, 1)])(x), total_count=10, name='y')

    model = jdab.JointDistributionCoroutineAutoBatched(model_fn, batch_ndims=1)
    samp = model.sample(seed=test_util.test_seed())
    states, trace = self.evaluate(
        windowed_sampling.windowed_adaptive_hmc(
            5,
            model.experimental_pin(y=samp.y),
            n_chains=n_chains,
            num_leapfrog_steps=3,
            num_adaptation_steps=25,
            seed=test_util.test_seed()))
    if isinstance(n_chains, int):
      n_chains = [n_chains]
    self.assertEqual((5, *n_chains, 3), states.x.shape)
    self.assertEqual((5,), trace['step_size'].shape)

  def test_dynamic_batch_shape(self):
    """Test correct handling of `TensorShape(None)`."""
    if JAX_MODE:
      self.skipTest('b/203858802')
    if tf.executing_eagerly():
      self.skipTest('Eager does not support dynamic batch shapes.')

    n_features = 5
    n_timepoints = 100
    features = normal.Normal(0., 1.).sample([100, n_features],
                                            test_util.test_seed())
    ar_sigma = 1.
    rho = .25

    @jdc.JointDistributionCoroutine
    def jd_model():
      beta = yield Root(sample.Sample(normal.Normal(0., 1.), n_features))
      yhat = tf.einsum('ij,...j->...i', features, beta)

      def ar_fun(y):
        loc = tf.concat([tf.zeros_like(y[..., :1]), y[..., :-1]], axis=-1)
        return independent.Independent(
            normal.Normal(loc=loc * rho, scale=ar_sigma),
            reinterpreted_batch_ndims=1)
      # Autoregressive distribution defined as below introduce a batch shape:
      # TensorShape(None)
      yield autoregressive.Autoregressive(
          distribution_fn=ar_fun,
          sample0=tf.zeros_like(yhat),
          num_steps=yhat.shape[-1],
          name='y')

    states, _ = self.evaluate(
        windowed_sampling.windowed_adaptive_nuts(
            2,
            jd_model,
            num_adaptation_steps=25,
            n_chains=3,
            seed=test_util.test_seed()))
    self.assertEqual((2, 3, n_timepoints), states.y.shape)

  @parameterized.named_parameters(
      ('_nuts', windowed_sampling.windowed_adaptive_nuts, {}),
      ('_hmc', windowed_sampling.windowed_adaptive_hmc, {
          'num_leapfrog_steps': 1
      }),
  )
  def test_f64_state(self, method, method_kwargs):
    states, _ = callable_util.get_output_spec(lambda: method(  # pylint: disable=g-long-lambda
        5,
        normal.Normal(tf.constant(0., tf.float64), 1.),
        n_chains=2,
        num_adaptation_steps=25,
        seed=test_util.test_seed(),
        **method_kwargs))

    self.assertEqual(tf.float64, states.dtype)


@test_util.test_graph_and_eager_modes
class WindowedSamplingStepSizeTest(test_util.TestCase):

  def test_supply_full_step_size(self):
    if tf.executing_eagerly() and not JAX_MODE:
      self.skipTest('Eager test is slow.')
    stream = test_util.test_seed_stream()

    jd_model = jdn.JointDistributionNamed({
        'a':
            normal.Normal(0., 1.),
        'b':
            mvn_diag.MultivariateNormalDiag(
                loc=tf.zeros(3), scale_diag=tf.constant([1., 2., 3.]))
    })

    init_step_size = {'a': tf.reshape(tf.linspace(1., 2., 3), (3, 1)),
                      'b': tf.reshape(tf.linspace(1., 2., 9), (3, 3))}

    _, actual_step_size = windowed_sampling.windowed_adaptive_hmc(
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
    if tf.executing_eagerly() and not JAX_MODE:
      self.skipTest('Eager test is slow.')
    stream = test_util.test_seed_stream()

    jd_model = jdn.JointDistributionNamed({
        'a':
            normal.Normal(0., 1.),
        'b':
            mvn_diag.MultivariateNormalDiag(
                loc=tf.zeros(3), scale_diag=tf.constant([1., 2., 3.]))
    })

    init_step_size = {'a': 1., 'b': 2.}
    _, actual_step_size = windowed_sampling.windowed_adaptive_hmc(
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
    if tf.executing_eagerly() and not JAX_MODE:
      self.skipTest('Eager test is slow.')
    stream = test_util.test_seed_stream()

    jd_model = jdn.JointDistributionNamed({
        'a':
            normal.Normal(0., 1.),
        'b':
            mvn_diag.MultivariateNormalDiag(
                loc=tf.zeros(3), scale_diag=tf.constant([1., 2., 3.]))
    })

    init_step_size = 1.
    _, traced_step_size = self.evaluate(
        windowed_sampling.windowed_adaptive_hmc(
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
    if tf.executing_eagerly() and not JAX_MODE:
      self.skipTest('Eager test is slow.')
    # Disable eager mode since it is slow.
    stream = test_util.test_seed_stream()

    jd_model = jds.JointDistributionSequential([
        half_normal.HalfNormal(scale=1., name=f'dist_{idx}') for idx in range(4)
    ])
    init_step_size = [1., 2., 3.]
    _, actual_step_size = windowed_sampling.windowed_adaptive_nuts(
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

    bb = beta_binomial.BetaBinomial(
        total_count=trials,
        concentration0=(1 - mean) / inverse_concentration,
        concentration1=mean / inverse_concentration)
    return independent.Independent(bb, reinterpreted_batch_ndims=2)

  return _beta_binomial_distribution


def get_joint_distribution(
    trials,
    mean_prior=lambda: uniform.Uniform(0., 1.),
    inverse_concentration_prior=lambda: half_normal.HalfNormal(5.)):
  """Returns a joint distribution over parameters and successes."""
  param_shape = ps.shape(trials)[:1]
  mean = sample.Sample(mean_prior(), param_shape)
  inverse_concentration = sample.Sample(inverse_concentration_prior(),
                                        param_shape)
  return jdn.JointDistributionNamed(
      dict(
          mean=mean,
          inverse_concentration=inverse_concentration,
          successes=_beta_binomial(trials)),
      name='jd')


class PrecompiledTest(test_util.TestCase):

  def setUp(self):
    super().setUp()
    arms = 2
    days = 3

    seed = test_util.test_seed()
    trial_seed, value_seed = samplers.split_seed(seed)
    self.trials = poisson.Poisson(100.).sample([arms, days], seed=trial_seed)
    dist = get_joint_distribution(self.trials)
    self.true_values = dist.sample(seed=value_seed)

  def nuts_kwargs(self):
    return {'max_tree_depth': 2}

  def hmc_kwargs(self):
    return {'num_leapfrog_steps': 3, 'store_parameters_in_results': True}

  @parameterized.named_parameters(('hmc_jit_sig', 'hmc'),
                                  ('nuts_jit_sig', 'nuts'))
  def test_base_kernel(self, kind):
    self.skip_if_no_xla()
    self.skipTest('b/195070752')  # Test is broken by cl/393807414.

    if JAX_MODE:
      input_signature = None
    else:
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
          chain_axis_names=None,
          seed=seed,
          successes=successes)

    self.evaluate(do(self.trials + 0., self.true_values['successes'],
                     test_util.test_seed(sampler_type='stateless')))

if JAX_MODE:
  # TF runs into the `merge_call` error here (b/181800108).

  @test_util.disable_test_for_backend(
      disable_numpy=True,
      reason='Sharding not available for NumPy backend.')
  class DistributedTest(distribute_test_lib.DistributedTest):

    def setUp(self):
      super().setUp()
      arms = 2
      days = 3

      seed = test_util.test_seed()
      trial_seed, value_seed = samplers.split_seed(seed)
      self.trials = poisson.Poisson(100.).sample([arms, days], seed=trial_seed)
      dist = get_joint_distribution(self.trials)
      self.true_values = dist.sample(seed=value_seed)

    def nuts_kwargs(self):
      return {'max_tree_depth': 2}

    def hmc_kwargs(self):
      return {'num_leapfrog_steps': 3, 'store_parameters_in_results': True}

    def test_can_extract_shard_axis_names_from_model(self):
      joint_dist = distribute.JointDistributionNamed(
          dict(
              x=normal.Normal(0., 1.),
              y=lambda x: distribute.Sharded(  # pylint:disable=g-long-lambda
                  normal.Normal(x, 1.), self.axis_name),
              z=lambda y: distribute.Sharded(  # pylint:disable=g-long-lambda
                  normal.Normal(y, 1.), self.axis_name)))

      def do():
        _, _, _, _, _, shard_axis_names = windowed_sampling._setup_mcmc(
            model=joint_dist,
            n_chains=[20],
            seed=test_util.test_seed(), z=1.)
        # _setup_mcmc will flatten the distribution
        self.assertListEqual(shard_axis_names, [[], ['i']])
      self.strategy_run(do, args=(), in_axes=None)

    @parameterized.named_parameters(('hmc_jit_sig', 'hmc'),
                                    ('nuts_jit_sig', 'nuts'))
    def test_data_sharding(self, kind):
      self.skip_if_no_xla()

      joint_dist = distribute.JointDistributionNamed(
          dict(
              x=normal.Normal(0., 1.),
              y=lambda x: distribute.Sharded(  # pylint:disable=g-long-lambda
                  normal.Normal(x, 1.), self.axis_name),
              z=lambda y: distribute.Sharded(  # pylint:disable=g-long-lambda
                  normal.Normal(y, 1.), self.axis_name)))

      def do(seed, z):
        if kind == 'hmc':
          proposal_kernel_kwargs = self.hmc_kwargs()
        else:
          proposal_kernel_kwargs = self.nuts_kwargs()

        return windowed_sampling._windowed_adaptive_impl(
            n_draws=10,
            joint_dist=joint_dist,
            kind=kind,
            n_chains=2,
            proposal_kernel_kwargs=proposal_kernel_kwargs,
            num_adaptation_steps=21,
            current_state=None,
            dual_averaging_kwargs={'target_accept_prob': 0.76},
            trace_fn=None,
            return_final_kernel_results=False,
            discard_tuning=True,
            seed=seed,
            chain_axis_names=None,
            z=z)

      self.evaluate(self.strategy_run(
          do,
          in_axes=(None, 0),
          args=(samplers.zeros_seed(), self.shard_values(
              tf.ones(distribute_test_lib.NUM_DEVICES)))))

    @parameterized.named_parameters(('hmc_jit_sig', 'hmc'),
                                    ('nuts_jit_sig', 'nuts'))
    def test_chain_sharding(self, kind):
      self.skip_if_no_xla()

      joint_dist = jdn.JointDistributionNamed(
          dict(
              x=normal.Normal(0., 1.),
              y=lambda x: sample.Sample(normal.Normal(x, 1.), 4),
              z=lambda y: independent.Independent(normal.Normal(y, 1.), 1)))

      def do(seed, z):
        if kind == 'hmc':
          proposal_kernel_kwargs = self.hmc_kwargs()
        else:
          proposal_kernel_kwargs = self.nuts_kwargs()

        return windowed_sampling._windowed_adaptive_impl(
            n_draws=10,
            joint_dist=joint_dist,
            kind=kind,
            n_chains=2,
            proposal_kernel_kwargs=proposal_kernel_kwargs,
            num_adaptation_steps=21,
            current_state=None,
            dual_averaging_kwargs={'target_accept_prob': 0.76},
            trace_fn=None,
            return_final_kernel_results=False,
            discard_tuning=True,
            seed=seed,
            chain_axis_names=self.axis_name,
            z=z)

      self.evaluate(self.strategy_run(
          do,
          in_axes=None,
          args=(samplers.zeros_seed(),
                tf.ones(distribute_test_lib.NUM_DEVICES))))

if __name__ == '__main__':
  test_util.main()
