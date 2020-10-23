# Lint as: python3
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
"""Test utilities for the Inference Gym."""

import collections
import importlib

from absl import flags
from absl import logging
import jax
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import test_util

flags.DEFINE_bool('use_tfds', False, 'Whether to run tests that use TFDS.',
                  allow_override=True)

FLAGS = flags.FLAGS

__all__ = [
    'InferenceGymTestCase',
    'MCMCResults',
    'multi_backend_test',
    'numpy_disable_gradient_test',
    'numpy_disable_test_missing_functionality',
    'run_hmc_on_model',
    'test_all_tf_execution_regimes',
    'uses_tfds',
]

BACKEND = None  # Rewritten by backends/rewrite.py.

# We want to test with 64 bit precision sometimes.
jax.config.update('jax_enable_x64', True)


def uses_tfds(test_fn):
  def _new_test_fn(self, *args, **kwargs):
    if FLAGS.use_tfds:
      test_fn(self, *args, **kwargs)
    else:
      self.skipTest('Uses TensorFlow Datasets. Enable using --use_tfds')

  return _new_test_fn


def numpy_disable_test_missing_functionality(reason, test_fn=None):
  """Disable a NumPy test."""
  if test_fn is None:
    return lambda test_fn: numpy_disable_test_missing_functionality(  # pylint: disable=g-long-lambda
        reason=reason, test_fn=test_fn)

  def _new_test_fn(self, *args, **kwargs):
    if BACKEND == 'backend_numpy':
      self.skipTest(reason)
    else:
      test_fn(self, *args, **kwargs)

  return _new_test_fn


def numpy_disable_gradient_test(test_fn):
  """Disable a autodiff-using NumPy test."""
  return numpy_disable_test_missing_functionality(
      'Skipping because NumPy backend doesn\'t support gradients.',
      test_fn=test_fn)


def multi_backend_test(globals_dict,
                       relative_module_name,
                       backends=('jax', 'tensorflow', 'numpy'),
                       test_case=None):
  """Multi-backend test decorator.

  The end goal of this decorator is that the decorated test case is removed, and
  replaced with a set of new test cases that have been rewritten to use one or
  more backends. E.g., a test case named `Test` will by default be rewritten to
  `Test_jax`, 'Test_tensorflow' and `Test_numpy` which use the JAX, TensorFlow
  and NumPy backends, respectively.

  The decorator works by using the dynamic rewrite system to rewrite imports of
  the module the test is defined in, and inserting the approriately renamed test
  cases into the `globals()` dictionary of the original module. A side-effect of
  this is that the global code inside the module is run `1 + len(backends)`
  times, so avoid doing anything expensive there. This does mean that the
  original module needs to be in a runnable state, i.e., when it uses symbols
  from `backend`, those must be actually present in the literal `backend`
  module.

  A subtle point about what this decorator does in the rewritten modules: the
  rewrite system changes the behavior of this decorator to act as a passthrough
  to avoid infinite rewriting loops.

  Args:
    globals_dict: Python dictionary of strings to symbols. Set this to the value
      of `globals()`.
    relative_module_name: Python string. The module name of the module where the
      decorated test resides relative to `inference_gym`. You must not use
      `__name__` for this as that is set to a defective value of `__main__`
      which is sufficiently abnormal that the rewrite system does not work on
      it.
    backends: Python iterable of strings. Which backends to test with.
    test_case: The actual test case to decorate.

  Returns:
    None, to delete the original test case.
  """
  if test_case is None:
    return lambda test_case: multi_backend_test(  # pylint: disable=g-long-lambda
        globals_dict=globals_dict,
        relative_module_name=relative_module_name,
        test_case=test_case)

  if BACKEND is not None:
    return test_case

  if relative_module_name == '__main__':
    raise ValueError(
        'module_name should be written out manually, not by passing __name__.')

  # This assumes `test_util` is 2 levels deep inside of `inference_gym`. If we
  # move it, we'd change the `-2` to equal the (negative) nesting level.
  root_name_comps = __name__.split('.')[:-2]
  relative_module_name_comps = relative_module_name.split('.')

  # Register the rewrite hooks.
  importlib.import_module('.'.join(root_name_comps + ['backends', 'rewrite']))

  new_test_case_names = []
  for backend in backends:
    new_module_name_comps = (
        root_name_comps + ['dynamic', 'backend_{}'.format(backend)] +
        relative_module_name_comps)
    # Rewrite the module.
    new_module = importlib.import_module('.'.join(new_module_name_comps))

    # Subclass the test case so that we can rename it (absl uses the class name
    # in its UI).
    base_new_test = getattr(new_module, test_case.__name__)
    new_test = type('{}_{}'.format(test_case.__name__, backend),
                    (base_new_test,), {})
    new_test_case_names.append(new_test.__name__)
    globals_dict[new_test.__name__] = new_test

  # We deliberately return None to delete the original test case from the
  # original module.


def test_all_tf_execution_regimes(test_case):
  """Only relevant for the TensorFlow backend."""
  if BACKEND == 'backend_tensorflow':
    return test_util.test_all_tf_execution_regimes(test_case)
  else:
    return test_case


class MCMCResults(
    collections.namedtuple('MCMCResults', [
        'chain',
        'accept_rate',
        'ess',
        'r_hat',
    ])):
  """Results of an MCMC run.

  Attributes:
    chain: A possibly nested structure of Tensors, representing the HMC chain.
    accept_rate: Acceptance rate of MCMC proposals.
    ess: Effective sample size.
    r_hat: Potential scale reduction.
  """


def run_hmc_on_model(
    model,
    num_chains,
    num_steps,
    num_leapfrog_steps,
    step_size,
    target_accept_prob=0.9,
    seed=None,
    dtype=tf.float32,
    use_xla=True,
):
  """Runs HMC on a target.

  Args:
    model: The model to validate.
    num_chains: Number of chains to run in parallel.
    num_steps: Total number of steps to take. The first half are used to warm up
      the sampler.
    num_leapfrog_steps: Number of leapfrog steps to take.
    step_size: Step size to use.
    target_accept_prob: Target acceptance probability.
    seed: Optional seed to use. By default, `test_util.test_seed()` is used.
    dtype: DType to use for the algorithm.
    use_xla: Whether to use XLA.

  Returns:
    mcmc_results: `MCMCResults`.
  """
  step_size = tf.convert_to_tensor(step_size, dtype)

  def target_log_prob_fn(*x):
    x = tf.nest.pack_sequence_as(model.dtype, x)
    return model.unnormalized_log_prob(x)

  if seed is None:
    seed = test_util.test_seed(sampler_type='stateless')
  current_state = tf.nest.map_structure(
      lambda b, e: b(  # pylint: disable=g-long-lambda
          tf.zeros([num_chains] + list(e), dtype=dtype)),
      model.default_event_space_bijector,
      model.event_shape)

  # tfp.mcmc only works well with lists.
  current_state = tf.nest.flatten(current_state)

  hmc = tfp.mcmc.HamiltonianMonteCarlo(
      target_log_prob_fn=target_log_prob_fn,
      num_leapfrog_steps=num_leapfrog_steps,
      step_size=[tf.fill(s.shape, step_size) for s in current_state])
  hmc = tfp.mcmc.TransformedTransitionKernel(
      hmc, tf.nest.flatten(model.default_event_space_bijector))
  hmc = tfp.mcmc.DualAveragingStepSizeAdaptation(
      hmc,
      num_adaptation_steps=int(num_steps // 2 * 0.8),
      target_accept_prob=target_accept_prob)

  # Subtle: Under JAX, there needs to be a data dependency on the input for
  # jitting to work.
  chain, is_accepted = tf.function(
      lambda current_state: tfp.mcmc.sample_chain(  # pylint: disable=g-long-lambda
          current_state=current_state,
          kernel=hmc,
          num_results=num_steps // 2,
          num_burnin_steps=num_steps // 2,
          trace_fn=lambda _, pkr:  # pylint: disable=g-long-lambda
          (pkr.inner_results.inner_results.is_accepted),
          seed=seed),
      autograph=False,
      experimental_compile=use_xla)(current_state)

  accept_rate = tf.reduce_mean(tf.cast(is_accepted, dtype))
  ess = tf.nest.map_structure(
      lambda c: tfp.mcmc.effective_sample_size(  # pylint: disable=g-long-lambda
          c,
          cross_chain_dims=1,
          filter_beyond_positive_pairs=True),
      chain)
  r_hat = tf.nest.map_structure(tfp.mcmc.potential_scale_reduction, chain)

  mcmc_results = MCMCResults(
      chain=tf.nest.pack_sequence_as(model.default_event_space_bijector, chain),
      accept_rate=accept_rate,
      ess=ess,
      r_hat=r_hat,
  )
  return mcmc_results


class InferenceGymTestCase(test_util.TestCase):
  """A TestCase mixin for common tests on inference gym targets."""

  def validate_log_prob_and_transforms(
      self,
      model,
      sample_transformation_shapes,
      check_ground_truth_mean=False,
      check_ground_truth_mean_standard_error=False,
      check_ground_truth_standard_deviation=False,
      check_ground_truth_standard_deviation_standard_error=False,
      seed=None,
  ):
    """Validate that the model's log probability and sample transformations run.

    This checks that unconstrained values passed through the event space
    bijectors into `unnormalized_log_prob` and sample transformations yield
    finite values. This also verifies that the transformed values have the
    expected shape.

    Args:
      model: The model to validate.
      sample_transformation_shapes: Shapes of the transformation outputs.
      check_ground_truth_mean: Whether to check the shape of the ground truth
        mean.
      check_ground_truth_mean_standard_error: Whether to check the shape of the
        ground truth standard error.
      check_ground_truth_standard_deviation: Whether to check the shape of the
        ground truth standard deviation.
      check_ground_truth_standard_deviation_standard_error: Whether to check the
        shape of the ground truth standard deviation standard error.
      seed: Optional seed to use. By default, `test_util.test_seed()` is used.
    """
    batch_size = 16

    if seed is None:
      seed = test_util.test_seed(sampler_type='stateless')

    def _random_element(shape, dtype, default_event_space_bijector, seed):
      unconstrained_shape = default_event_space_bijector.inverse_event_shape(
          shape)
      unconstrained_shape = tf.TensorShape([batch_size
                                           ]).concatenate(unconstrained_shape)
      return default_event_space_bijector.forward(
          tf.random.stateless_normal(
              unconstrained_shape, dtype=dtype, seed=seed))

    num_seeds = len(tf.nest.flatten(model.dtype))
    flat_seed = tf.unstack(tfp.random.split_seed(seed, num_seeds), axis=0)
    seed = tf.nest.pack_sequence_as(model.dtype, flat_seed)

    test_points = tf.nest.map_structure(_random_element, model.event_shape,
                                        model.dtype,
                                        model.default_event_space_bijector,
                                        seed)
    log_prob = self.evaluate(model.unnormalized_log_prob(test_points))

    self.assertAllFinite(log_prob)
    self.assertEqual((batch_size,), log_prob.shape)

    for name, sample_transformation in model.sample_transformations.items():
      transformed_points = self.evaluate(sample_transformation(test_points))

      def _assertions_part(expected_shape, expected_dtype, transformed_part):
        self.assertAllFinite(transformed_part)
        self.assertEqual(
            (batch_size,) + tuple(expected_shape),
            tuple(list(transformed_part.shape)))
        self.assertEqual(expected_dtype, transformed_part.dtype)

      self.assertAllAssertsNested(
          _assertions_part,
          sample_transformation_shapes[name],
          sample_transformation.dtype,
          transformed_points,
          shallow=transformed_points,
          msg='Checking outputs of: {}'.format(name))

      def _ground_truth_shape_check_part(expected_shape, ground_truth):
        self.assertEqual(
            tuple(expected_shape),
            tuple(ground_truth.shape))

      if check_ground_truth_mean:
        self.assertAllAssertsNested(
            _ground_truth_shape_check_part,
            sample_transformation_shapes[name],
            sample_transformation.ground_truth_mean,
            shallow=transformed_points,
            msg='Checking ground truth mean of: {}'.format(name))

      if check_ground_truth_mean_standard_error:
        self.assertAllAssertsNested(
            _ground_truth_shape_check_part,
            sample_transformation_shapes[name],
            sample_transformation.ground_truth_mean_standard_error,
            shallow=transformed_points,
            msg='Checking ground truth mean standard error: {}'.format(name))

      if check_ground_truth_standard_deviation:
        self.assertAllAssertsNested(
            _ground_truth_shape_check_part,
            sample_transformation_shapes[name],
            sample_transformation.ground_truth_standard_deviation,
            shallow=transformed_points,
            msg='Checking ground truth standard deviation: {}'.format(name))

      if check_ground_truth_standard_deviation_standard_error:
        self.assertAllAssertsNested(
            _ground_truth_shape_check_part,
            sample_transformation_shapes[name],
            sample_transformation
            .ground_truth_standard_deviation_standard_error,
            shallow=transformed_points,
            msg='Checking ground truth standard deviation strandard error: {}'
            .format(name))

  def validate_ground_truth_using_hmc(
      self,
      model,
      num_chains,
      num_steps,
      num_leapfrog_steps,
      step_size,
      target_accept_prob=0.9,
      seed=None,
      dtype=tf.float32,
      mean_fudge_atol=0,
      standard_deviation_fudge_atol=0,
      use_xla=True,
  ):
    """Validates the ground truth of a model using HMC.

    Args:
      model: The model to validate.
      num_chains: Number of chains to run in parallel.
      num_steps: Total number of steps to take. The first half are used to warm
        up the sampler.
      num_leapfrog_steps: Number of leapfrog steps to take.
      step_size: Step size to use.
      target_accept_prob: Target acceptance probability.
      seed: Optional seed to use. By default, `test_util.test_seed()` is used.
      dtype: DType to use for the algorithm.
      mean_fudge_atol: Scalar `Tensor`. Additional atol to use when doing the
        Z-test for the sample mean.
      standard_deviation_fudge_atol: Scalar `Tensor`. Additional atol to use
        when doing the Z-test for the sample standard deviation,
      use_xla: Whether to use XLA.
    """
    mcmc_results = self.evaluate(
        run_hmc_on_model(
            model,
            num_chains=num_chains,
            num_steps=num_steps,
            num_leapfrog_steps=num_leapfrog_steps,
            step_size=step_size,
            target_accept_prob=target_accept_prob,
            seed=seed,
            dtype=dtype,
            use_xla=use_xla))

    logging.info('Acceptance rate: %s', mcmc_results.accept_rate)
    logging.info('ESS: %s', mcmc_results.ess)
    logging.info('r_hat: %s', mcmc_results.r_hat)

    for name, sample_transformation in model.sample_transformations.items():
      transformed_chain = self.evaluate(
          tf.nest.map_structure(tf.identity,
                                sample_transformation(mcmc_results.chain)))

      # tfp.mcmc.effective_sample_size only works well with lists.
      flat_transformed_chain = tf.nest.flatten(transformed_chain)
      cross_chain_dims = [1] * len(flat_transformed_chain)
      flat_ess = self.evaluate(
          tfp.mcmc.effective_sample_size(
              flat_transformed_chain,
              cross_chain_dims=cross_chain_dims,
              filter_beyond_positive_pairs=True))
      self._z_test(
          name=name,
          sample_transformation=sample_transformation,
          transformed_samples=transformed_chain,
          num_samples=tf.nest.pack_sequence_as(transformed_chain, flat_ess),
          mean_fudge_atol=mean_fudge_atol,
          standard_deviation_fudge_atol=standard_deviation_fudge_atol,
          sample_dims=(0, 1),
      )

  def validate_ground_truth_using_monte_carlo(
      self,
      model,
      num_samples,
      seed=None,
      dtype=tf.float32,
  ):
    """Validates the ground truth of a model using forward sampling.

    This requires a model to have a `sample` method. This is typically only
    applicable to synthetic models.

    Args:
      model: The model to validate. It must have a `sample` method.
      num_samples: Number of samples to generate.
      seed: Optional seed to use. By default, `test_util.test_seed()` is used.
      dtype: DType to use for the algorithm.
    """
    if seed is None:
      seed = test_util.test_seed()
    samples = model.sample(num_samples, seed=seed)

    for name, sample_transformation in model.sample_transformations.items():
      transformed_samples = self.evaluate(
          tf.identity(sample_transformation(samples)))
      nested_num_samples = tf.nest.map_structure(lambda _: num_samples,
                                                 transformed_samples)
      self._z_test(
          name=name,
          sample_transformation=sample_transformation,
          transformed_samples=transformed_samples,
          num_samples=nested_num_samples,
          sample_dims=0,
      )

  def _z_test(
      self,
      name,
      sample_transformation,
      transformed_samples,
      num_samples,
      mean_fudge_atol=0,
      standard_deviation_fudge_atol=0,
      sample_dims=0,
  ):
    """Does a two-sample, two-sided Z-test between samples and ground truth."""
    sample_mean = tf.nest.map_structure(
        lambda transformed_samples: np.mean(  # pylint: disable=g-long-lambda
            transformed_samples,
            axis=sample_dims),
        transformed_samples)
    sample_variance = tf.nest.map_structure(
        lambda transformed_samples: np.var(  # pylint: disable=g-long-lambda
            transformed_samples,
            axis=sample_dims),
        transformed_samples)
    # TODO(b/144524123): As written, this does a two sided Z-test at an
    # alpha=O(1e-7). It definitely has very little power as a result.
    # Currently it also uses the sample variance to compute the Z-score. In
    # principle, we can use the ground truth variance, but it's unclear
    # whether that's appropriate. Heuristically, a typical error that HMC has
    # is getting stuck, meaning that the sample variance is too low,
    # causing the test to fail more often. HMC can also in principle
    # over-estimate the variance, but that seems less typical.
    #
    # We should re-examine the literature for Z-testing and justify these
    # choices on formal grounds.
    def _mean_assertions_part(ground_truth_mean,
                              ground_truth_mean_standard_error, sample_mean,
                              sample_variance, num_samples):
      sample_standard_error_sq = sample_variance / num_samples
      self.assertAllClose(
          ground_truth_mean,
          sample_mean,
          # TODO(b/144290399): Use the full atol vector.
          atol=(np.array(
              5. * np.sqrt(sample_standard_error_sq +
                           ground_truth_mean_standard_error**2)).max() +
                mean_fudge_atol),
      )

    ground_truth_mean_standard_error = (
        sample_transformation.ground_truth_mean_standard_error)
    if ground_truth_mean_standard_error is None:
      # Note that this is NOT the intended general interpretaton of SEM being
      # None, but only the interpetation we choose for these tests.
      ground_truth_mean_standard_error = tf.nest.map_structure(
          np.zeros_like, sample_transformation.ground_truth_mean)

    self.assertAllAssertsNested(
        _mean_assertions_part,
        sample_transformation.ground_truth_mean,
        ground_truth_mean_standard_error,
        sample_mean,
        sample_variance,
        num_samples,
        msg='Comparing mean of "{}"'.format(name))

    if sample_transformation.ground_truth_standard_deviation is not None:
      # From https://math.stackexchange.com/q/72975
      fourth_moment = tf.nest.map_structure(
          lambda transformed_samples, sample_mean: np.mean(  # pylint: disable=g-long-lambda
              (transformed_samples - sample_mean)**4,
              axis=tuple(tf.nest.flatten(sample_dims))),
          transformed_samples,
          sample_mean)

      ground_truth_standard_deviation_standard_error = (
          sample_transformation.ground_truth_standard_deviation_standard_error)
      if ground_truth_standard_deviation_standard_error is None:
        # Note that this is NOT the intended general interpretaton of SESD being
        # None, but only the interpetation we choose for these tests.
        ground_truth_standard_deviation_standard_error = tf.nest.map_structure(
            np.zeros_like,
            sample_transformation.ground_truth_standard_deviation)

      def _var_assertions_part(ground_truth_standard_deviation,
                               ground_truth_standard_deviation_standard_error,
                               sample_variance, fourth_moment, num_samples):
        sample_standard_error_sq = (
            fourth_moment / num_samples - sample_variance**2 *
            (num_samples - 3) / num_samples / (num_samples - 1))
        self.assertAllClose(
            np.square(ground_truth_standard_deviation),
            sample_variance,
            # TODO(b/144290399): Use the full atol vector.
            atol=np.array(
                5. *
                np.sqrt(sample_standard_error_sq +
                        ground_truth_standard_deviation_standard_error)).max() +
            standard_deviation_fudge_atol,
        )

      self.assertAllAssertsNested(
          _var_assertions_part,
          sample_transformation.ground_truth_standard_deviation,
          ground_truth_standard_deviation_standard_error,
          sample_variance,
          fourth_moment,
          num_samples,
          msg='Comparing variance of "{}"'.format(name),
      )

  def validate_deferred_materialization(self, model_fn, **kwargs):
    """Validates that a model does not materialize args too early.

    Given a `model_fn` and a set of NumPy arrays in `kwargs` this verifies that
    none of the arrays are actually accessed by anything other than `dtype` and
    `shape` properties when accessing similarly lightweight properties of the
    model.

    Args:
      model_fn: A function that returns a Model.
      **kwargs: Keyword arguments to pass to `model_fn`. Each value should be a
        NumPy array.
    """
    deferred_kwargs = {}

    def make_loud_materialization(name, value):
      if value is None:
        return None

      def do_assert():
        raise AssertionError('Erroneously materialized {}'.format(name))

      empty = np.zeros(0)
      return tfp.util.DeferredTensor(
          empty, lambda _: do_assert(), shape=value.shape, dtype=value.dtype)

    for k, v in kwargs.items():
      deferred_kwargs[k] = make_loud_materialization(k, v)

    model = model_fn(**deferred_kwargs)
    _ = model.dtype
    _ = model.name
    _ = model.event_shape
    _ = model.default_event_space_bijector
    _ = model.sample_transformations
    _ = str(model)
