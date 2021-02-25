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
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.experimental.mcmc import windowed_sampling
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import test_util


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
    seed: For reproducibility
  Returns:
    (tfd.JointDistribution, tf.Tensor), representing the joint distribution
    above, and the posterior variance.
  """
  seeds = samplers.split_seed(seed, 5)
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

  dists, _ = model.sample_distributions()
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
  def test_hmc_samples(self, model_fn):
    model = model_fn()
    pins = {'treatment_effects': tf.constant(TREATMENT_EFFECTS)}

    @tf.function
    def do_sample():
      return tfp.experimental.mcmc.windowed_adaptive_hmc(
          200, model, num_leapfrog_steps=8, seed=test_util.test_seed(),
          **pins)

    draws, _ = do_sample()
    flat_draws = tf.nest.flatten(
        model.experimental_pin(**pins)._model_flatten(draws))
    max_scale_reduction = tf.reduce_max(
        tf.nest.map_structure(tf.reduce_max,
                              tfp.mcmc.potential_scale_reduction(flat_draws)))
    self.assertLess(self.evaluate(max_scale_reduction), 1.35)

  @parameterized.named_parameters(
      dict(testcase_name='_' + fn.__name__, model_fn=fn) for fn in
      [eight_schools_coroutine, eight_schools_named, eight_schools_sequential,
       eight_schools_nested])
  def test_nuts_samples(self, model_fn):
    model = model_fn()
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
    self.assertAllClose(true_var, final_scaling, rtol=0.1)

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
    self.assertAllClose(true_var, final_scaling, rtol=0.05)


if __name__ == '__main__':
  tf.test.main()
