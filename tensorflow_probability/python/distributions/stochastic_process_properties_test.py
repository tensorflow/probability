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
"""Property-based testing for stochastic processes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import hypothesis as hp
from hypothesis import strategies as hps
import numpy as np
import six
import tensorflow.compat.v2 as tf

from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.internal import hypothesis_testlib as tfp_hps
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.math.psd_kernels import hypothesis_testlib as kernel_hps


PARAM_EVENT_NDIMS_BY_PROCESS_NAME = {
    'GaussianProcess': dict(observation_noise_variance=0),
    'GaussianProcessRegressionModel': dict(observation_noise_variance=0),
    'StudentTProcess': dict(df=0),
    'VariationalGaussianProcess': dict(obseration_noise_variance=0)
}


MUTEX_PARAMS = set()


MAX_CONVERSIONS_BY_CLASS = dict(
    GaussianProcessRegressionModel=4,
    VariationalGaussianProcess=9)


def _stochastic_process_specific_hp_settings(test_method):
  return tfp_hps.tfp_hp_settings(
      default_max_examples=5,
      suppress_health_check=[
          hp.HealthCheck.too_slow,
          hp.HealthCheck.filter_too_much,
          hp.HealthCheck.data_too_large])(test_method)


# pylint is unable to handle @hps.composite (e.g. complains "No value for
# argument 'batch_shape' in function call"), so disable this lint for the file.

# pylint: disable=no-value-for-parameter


@hps.composite
def broadcasting_params(draw,
                        process_name,
                        batch_shape,
                        event_dim=None,
                        enable_vars=False):
  """Strategy for drawing parameters broadcasting to `batch_shape`."""
  if process_name not in PARAM_EVENT_NDIMS_BY_PROCESS_NAME:
    raise ValueError('Unknown Process name {}'.format(process_name))

  params_event_ndims = PARAM_EVENT_NDIMS_BY_PROCESS_NAME[process_name]

  def _constraint(param):
    return constraint_for(process_name, param)

  return draw(
      tfp_hps.broadcasting_params(
          batch_shape,
          params_event_ndims,
          event_dim=event_dim,
          enable_vars=enable_vars,
          constraint_fn_for=_constraint,
          mutex_params=MUTEX_PARAMS,
          dtype=np.float64))


@hps.composite
def stochastic_processes(draw,
                         process_name=None,
                         kernel_name=None,
                         batch_shape=None,
                         event_dim=None,
                         feature_dim=None,
                         feature_ndims=None,
                         enable_vars=False):
  if process_name is None:
    process_name = draw(hps.sampled_from(sorted(
        PARAM_EVENT_NDIMS_BY_PROCESS_NAME.keys())))
  if batch_shape is None:
    batch_shape = draw(tfp_hps.shapes())
  if event_dim is None:
    event_dim = draw(hps.integers(min_value=2, max_value=4))
  if feature_dim is None:
    feature_dim = draw(hps.integers(min_value=2, max_value=4))
  if feature_ndims is None:
    feature_ndims = draw(hps.integers(min_value=1, max_value=3))

  if process_name == 'GaussianProcess':
    return draw(gaussian_processes(
        kernel_name=kernel_name,
        batch_shape=batch_shape,
        event_dim=event_dim,
        feature_dim=feature_dim,
        feature_ndims=feature_ndims,
        enable_vars=enable_vars))
  elif process_name == 'GaussianProcessRegressionModel':
    return draw(gaussian_process_regression_models(
        kernel_name=kernel_name,
        batch_shape=batch_shape,
        event_dim=event_dim,
        feature_dim=feature_dim,
        feature_ndims=feature_ndims,
        enable_vars=enable_vars))
  elif process_name == 'VariationalGaussianProcess':
    return draw(variational_gaussian_processes(
        kernel_name=kernel_name,
        batch_shape=batch_shape,
        event_dim=event_dim,
        feature_dim=feature_dim,
        feature_ndims=feature_ndims,
        enable_vars=enable_vars))
  elif process_name == 'StudentTProcess':
    return draw(student_t_processes(
        kernel_name=kernel_name,
        batch_shape=batch_shape,
        event_dim=event_dim,
        feature_dim=feature_dim,
        feature_ndims=feature_ndims,
        enable_vars=enable_vars))
  raise ValueError('Stochastic process "{}" not found.'.format(process_name))


@hps.composite
def gaussian_processes(draw,
                       kernel_name=None,
                       batch_shape=None,
                       event_dim=None,
                       feature_dim=None,
                       feature_ndims=None,
                       enable_vars=False):
  # First draw a kernel.
  k, _ = draw(kernel_hps.base_kernels(
      kernel_name=kernel_name,
      batch_shape=batch_shape,
      event_dim=event_dim,
      feature_dim=feature_dim,
      feature_ndims=feature_ndims,
      # Disable variables
      enable_vars=False))
  compatible_batch_shape = draw(
      tfp_hps.broadcast_compatible_shape(k.batch_shape))
  index_points = draw(kernel_hps.kernel_input(
      batch_shape=compatible_batch_shape,
      example_ndims=1,
      feature_dim=feature_dim,
      feature_ndims=feature_ndims,
      enable_vars=enable_vars,
      name='index_points'))
  params = draw(broadcasting_params(
      'GaussianProcess',
      compatible_batch_shape,
      event_dim=event_dim,
      enable_vars=enable_vars))

  gp = tfd.GaussianProcess(
      kernel=k, index_points=index_points,
      observation_noise_variance=params['observation_noise_variance'])
  return gp


@hps.composite
def gaussian_process_regression_models(draw,
                                       kernel_name=None,
                                       batch_shape=None,
                                       event_dim=None,
                                       feature_dim=None,
                                       feature_ndims=None,
                                       enable_vars=False):
  # First draw a kernel.
  k, _ = draw(kernel_hps.base_kernels(
      kernel_name=kernel_name,
      batch_shape=batch_shape,
      event_dim=event_dim,
      feature_dim=feature_dim,
      feature_ndims=feature_ndims,
      # Disable variables
      enable_vars=False))
  compatible_batch_shape = draw(
      tfp_hps.broadcast_compatible_shape(k.batch_shape))
  index_points = draw(kernel_hps.kernel_input(
      batch_shape=compatible_batch_shape,
      example_ndims=1,
      feature_dim=feature_dim,
      feature_ndims=feature_ndims,
      enable_vars=enable_vars,
      name='index_points'))
  hp.note('Index points:\n{}'.format(repr(index_points)))

  observation_index_points = draw(
      kernel_hps.kernel_input(
          batch_shape=compatible_batch_shape,
          example_ndims=1,
          feature_dim=feature_dim,
          feature_ndims=feature_ndims,
          enable_vars=enable_vars,
          name='observation_index_points'))
  hp.note('Observation index points:\n{}'.format(
      repr(observation_index_points)))

  observations = draw(kernel_hps.kernel_input(
      batch_shape=compatible_batch_shape,
      example_ndims=1,
      # This is the example dimension suggested observation_index_points.
      example_dim=int(observation_index_points.shape[-(feature_ndims + 1)]),
      # No feature dimensions.
      feature_dim=0,
      feature_ndims=0,
      enable_vars=enable_vars,
      name='observations'))
  hp.note('Observations:\n{}'.format(repr(observations)))

  params = draw(broadcasting_params(
      'GaussianProcessRegressionModel',
      compatible_batch_shape,
      event_dim=event_dim,
      enable_vars=enable_vars))
  hp.note('Params:\n{}'.format(repr(params)))

  gp = tfd.GaussianProcessRegressionModel(
      kernel=k,
      index_points=index_points,
      observation_index_points=observation_index_points,
      observations=observations,
      observation_noise_variance=params[
          'observation_noise_variance'])
  return gp


@hps.composite
def variational_gaussian_processes(draw,
                                   kernel_name=None,
                                   batch_shape=None,
                                   event_dim=None,
                                   feature_dim=None,
                                   feature_ndims=None,
                                   enable_vars=False):
  # First draw a kernel.
  k, _ = draw(kernel_hps.base_kernels(
      kernel_name=kernel_name,
      batch_shape=batch_shape,
      event_dim=event_dim,
      feature_dim=feature_dim,
      feature_ndims=feature_ndims,
      # Disable variables
      enable_vars=False))
  compatible_batch_shape = draw(
      tfp_hps.broadcast_compatible_shape(k.batch_shape))
  index_points = draw(kernel_hps.kernel_input(
      batch_shape=compatible_batch_shape,
      example_ndims=1,
      feature_dim=feature_dim,
      feature_ndims=feature_ndims,
      enable_vars=enable_vars,
      name='index_points'))
  hp.note('Index points:\n{}'.format(repr(index_points)))

  inducing_index_points = draw(kernel_hps.kernel_input(
      batch_shape=compatible_batch_shape,
      example_ndims=1,
      feature_dim=feature_dim,
      feature_ndims=feature_ndims,
      enable_vars=enable_vars,
      name='inducing_index_points'))
  hp.note('Inducing index points:\n{}'.format(repr(inducing_index_points)))
  num_inducing_points = int(inducing_index_points.shape[-(feature_ndims + 1)])

  variational_inducing_observations_loc = draw(kernel_hps.kernel_input(
      batch_shape=compatible_batch_shape,
      example_ndims=1,
      example_dim=num_inducing_points,
      feature_dim=0,
      feature_ndims=0,
      enable_vars=enable_vars,
      name='variational_inducing_observations_loc'))
  hp.note('Variational inducing observations loc:\n{}'.format(
      repr(variational_inducing_observations_loc)))

  variational_inducing_observations_scale = draw(tfp_hps.tensors_in_support(
      support=tfp_hps.Support.MATRIX_LOWER_TRIL_POSITIVE_DEFINITE,
      batch_shape=compatible_batch_shape.as_list(),
      event_dim=num_inducing_points,
      dtype=np.float64))
  hp.note('Variational inducing observations scale:\n{}'.format(
      repr(variational_inducing_observations_scale)))

  params = draw(broadcasting_params(
      'GaussianProcessRegressionModel',
      compatible_batch_shape,
      event_dim=event_dim,
      enable_vars=enable_vars))
  hp.note('Params:\n{}'.format(repr(params)))

  vgp = tfd.VariationalGaussianProcess(
      kernel=k,
      index_points=index_points,
      inducing_index_points=inducing_index_points,
      variational_inducing_observations_loc=(
          variational_inducing_observations_loc),
      variational_inducing_observations_scale=(
          variational_inducing_observations_scale),
      observation_noise_variance=params[
          'observation_noise_variance'])
  return vgp


@hps.composite
def student_t_processes(draw,
                        kernel_name=None,
                        batch_shape=None,
                        event_dim=None,
                        feature_dim=None,
                        feature_ndims=None,
                        enable_vars=False):
  # First draw a kernel.
  k, _ = draw(kernel_hps.base_kernels(
      kernel_name=kernel_name,
      batch_shape=batch_shape,
      event_dim=event_dim,
      feature_dim=feature_dim,
      feature_ndims=feature_ndims,
      # Disable variables
      enable_vars=False))
  compatible_batch_shape = draw(
      tfp_hps.broadcast_compatible_shape(k.batch_shape))
  index_points = draw(kernel_hps.kernel_input(
      batch_shape=compatible_batch_shape,
      example_ndims=1,
      feature_dim=feature_dim,
      feature_ndims=feature_ndims,
      enable_vars=enable_vars,
      name='index_points'))
  params = draw(broadcasting_params(
      'StudentTProcess',
      compatible_batch_shape,
      event_dim=event_dim,
      enable_vars=enable_vars))
  stp = tfd.StudentTProcess(
      kernel=k,
      index_points=index_points,
      # The Student-T Process can encounter cholesky decomposition errors,
      # so use a large jitter to avoid that.
      jitter=1e-1,
      df=params['df'])
  return stp


def assert_shapes_unchanged(target_shaped_dict, possibly_bcast_dict):
  for param, target_param_val in six.iteritems(target_shaped_dict):
    np.testing.assert_array_equal(
        tensorshape_util.as_list(target_param_val.shape),
        tensorshape_util.as_list(possibly_bcast_dict[param].shape))


@test_util.test_all_tf_execution_regimes
class StochasticProcessParamsAreVarsTest(test_util.TestCase):

  @parameterized.named_parameters(
      {'testcase_name': '_{}'.format(sname), 'process_name': sname}
      for sname in PARAM_EVENT_NDIMS_BY_PROCESS_NAME)
  @hp.given(hps.data())
  @_stochastic_process_specific_hp_settings
  def testVariableParametersArePreserved(self, process_name, data):
    # Check that the process passes Variables through to the accessor
    # properties (without converting them to Tensor or anything like that).
    process = data.draw(stochastic_processes(process_name, enable_vars=True))
    self.evaluate([var.initializer for var in process.variables])

    for k, v in six.iteritems(process.parameters):
      if not tensor_util.is_ref(v):
        continue
      self.assertIs(
          getattr(process, k), v,
          'Parameter equivalance assertion failed for parameter `{}`'.format(k))

  @parameterized.named_parameters(
      {'testcase_name': '_' + sname, 'process_name': sname}
      for sname in PARAM_EVENT_NDIMS_BY_PROCESS_NAME)
  @hp.given(hps.data())
  @_stochastic_process_specific_hp_settings
  def testExcessiveConcretizationInZeroArgPublicMethods(
      self, process_name, data):
    tfp_hps.guitar_skip_if_matches(
        'VariationalGaussianProcess', process_name, 'b/147770193')
    # Check that standard statistics do not concretize variables/deferred
    # tensors more than the allowed amount.
    process = data.draw(stochastic_processes(process_name, enable_vars=True))
    self.evaluate([var.initializer for var in process.variables])

    for stat in ['mean', 'covariance', 'stddev', 'variance', 'sample']:
      hp.note('Testing excessive concretization in {}.{}'.format(process_name,
                                                                 stat))
      try:
        with tfp_hps.assert_no_excessive_var_usage(
            'method `{}` of `{}`'.format(stat, process),
            max_permissible=MAX_CONVERSIONS_BY_CLASS.get(process_name, 1)):
          getattr(process, stat)()

      except NotImplementedError:
        pass

  @parameterized.named_parameters(
      {'testcase_name': '_' + sname, 'process_name': sname}
      for sname in PARAM_EVENT_NDIMS_BY_PROCESS_NAME)
  @hp.given(hps.data())
  @_stochastic_process_specific_hp_settings
  def testGradientsThroughSample(self, process_name, data):
    tfp_hps.guitar_skip_if_matches(
        'VariationalGaussianProcess', process_name, 'b/147770193')
    process = data.draw(stochastic_processes(
        process_name=process_name, enable_vars=True))
    self.evaluate([var.initializer for var in process.variables])

    with tf.GradientTape() as tape:
      sample = process.sample()
    if process.reparameterization_type == tfd.FULLY_REPARAMETERIZED:
      grads = tape.gradient(sample, process.variables)
      for grad, var in zip(grads, process.variables):
        self.assertIsNotNone(
            grad,
            'Grad of sample was `None` for var: {}.'.format(var))

  @parameterized.named_parameters(
      {'testcase_name': '_' + sname, 'process_name': sname}
      for sname in PARAM_EVENT_NDIMS_BY_PROCESS_NAME)
  @hp.given(hps.data())
  @_stochastic_process_specific_hp_settings
  def testGradientsThroughLogProb(self, process_name, data):
    tfp_hps.guitar_skip_if_matches(
        'VariationalGaussianProcess', process_name, 'b/147770193')
    process = data.draw(stochastic_processes(
        process_name=process_name, enable_vars=True))
    self.evaluate([var.initializer for var in process.variables])

    # Test that log_prob produces non-None gradients.
    sample = process.sample()
    with tf.GradientTape() as tape:
      lp = process.log_prob(sample)
    grads = tape.gradient(lp, process.variables)
    for grad, var in zip(grads, process.variables):
      self.assertIsNotNone(
          grad,
          'Grad of log_prob was `None` for var: {}.'.format(var))

  @parameterized.named_parameters(
      {'testcase_name': '_' + sname, 'process_name': sname}
      for sname in PARAM_EVENT_NDIMS_BY_PROCESS_NAME)
  @hp.given(hps.data())
  @_stochastic_process_specific_hp_settings
  def testExcessiveConcretizationInLogProb(self, process_name, data):
    # Check that log_prob computations avoid reading process parameters
    # more than once.
    tfp_hps.guitar_skip_if_matches(
        'VariationalGaussianProcess', process_name, 'b/147770193')
    process = data.draw(stochastic_processes(
        process_name=process_name, enable_vars=True))
    self.evaluate([var.initializer for var in process.variables])

    hp.note('Testing excessive var usage in {}.log_prob'.format(process_name))
    sample = process.sample()
    try:
      with tfp_hps.assert_no_excessive_var_usage(
          'method `log_prob` of `{}`'.format(process),
          max_permissible=MAX_CONVERSIONS_BY_CLASS.get(process_name, 1)):
        process.log_prob(sample)
    except NotImplementedError:
      pass


def greater_than_twenty(x):
  return tf.math.softplus(x) + 20


def strictly_greater_than(epsilon=1e-1):
  return lambda x: tf.math.softplus(x) + epsilon


CONSTRAINTS = {
    'observation_noise_variance': strictly_greater_than(1e-1),
    # This will resemble a GP, and be more numerically stable as a result.
    'df': greater_than_twenty
}


def constraint_for(process=None, param=None):
  if param is not None:
    return CONSTRAINTS.get('{}.{}'.format(process, param),
                           CONSTRAINTS.get(param, tfp_hps.identity_fn))
  return CONSTRAINTS.get(process, tfp_hps.identity_fn)


if __name__ == '__main__':
  tf.test.main()
