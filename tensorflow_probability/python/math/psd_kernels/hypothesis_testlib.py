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
"""Utilities for hypothesis testing of psd_kernels."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import hypothesis as hp
from hypothesis.extra import numpy as hpnp
import hypothesis.strategies as hps
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.internal import hypothesis_testlib as tfp_hps
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.math import psd_kernels as tfpk


INSTANTIABLE_BASE_KERNELS = {
    'ExpSinSquared': dict(amplitude=0, length_scale=0, period=0),
    'ExponentiatedQuadratic': dict(amplitude=0, length_scale=0),
    'Linear': dict(bias_variance=0, slope_variance=0, shift=0),
    'MaternOneHalf': dict(amplitude=0, length_scale=0),
    'MaternThreeHalves': dict(amplitude=0, length_scale=0),
    'MaternFiveHalves': dict(amplitude=0, length_scale=0),
    # TODO(b/146073659): Polynomial as currently configured often produces
    # numerically ill-conditioned matrices. Disabled until we can make it more
    # reliable in the context of hypothesis tests.
    # 'Polynomial': dict(
    #    bias_variance=0, slope_variance=0, shift=0, exponent=0),
    'RationalQuadratic': dict(
        amplitude=0, length_scale=0, scale_mixture_rate=0),
}


SPECIAL_KERNELS = [
    'FeatureScaled',
    'SchurComplement'
]

MUTEX_PARAMS = tuple()


# pylint is unable to handle @hps.composite (e.g. complains "No value for
# argument 'batch_shape' in function call"), so disable this lint for the file.

# pylint: disable=no-value-for-parameter


@hps.composite
def kernel_input(
    draw,
    batch_shape,
    example_dim=None,
    example_ndims=None,
    feature_dim=None,
    feature_ndims=None,
    enable_vars=False,
    name=None):
  """Strategy for drawing arbitrary Kernel input.

  In order to avoid duplicates (or even numerically near-duplicates), we
  generate inputs on a grid. We let hypothesis generate the number of grid
  points and distance between grid points, within some reasonable pre-defined
  ranges. The result will be a batch of example sets, within which each set of
  examples has no duplicates (but no such duplication avoidance is applied
  accross batches).

  Args:
    draw: Hypothesis function supplied by `@hps.composite`.
    batch_shape: `TensorShape`. The batch shape of the resulting
      kernel input.
    example_dim: Optional Python int giving the size of each example dimension.
      If omitted, Hypothesis will choose one.
    example_ndims: Optional Python int giving the number of example dimensions
      of the input. If omitted, Hypothesis will choose one.
    feature_dim: Optional Python int giving the size of each feature dimension.
      If omitted, Hypothesis will choose one.
    feature_ndims: Optional Python int stating the number of feature dimensions
      inputs will have. If omitted, Hypothesis will choose one.
    enable_vars: If `False`, the returned parameters are all Tensors, never
      Variables or DeferredTensor.
    name: Name to give the variable.

  Returns:
    kernel_input: A strategy for drawing kernel_input with the prescribed shape
      (or an arbitrary one if omitted).
  """
  if example_ndims is None:
    example_ndims = draw(hps.integers(min_value=1, max_value=2))
  if example_dim is None:
    example_dim = draw(hps.integers(min_value=2, max_value=4))

  if feature_ndims is None:
    feature_ndims = draw(hps.integers(min_value=1, max_value=2))
  if feature_dim is None:
    feature_dim = draw(hps.integers(min_value=2, max_value=4))

  batch_shape = tensorshape_util.as_list(batch_shape)
  example_shape = [example_dim] * example_ndims
  feature_shape = [feature_dim] * feature_ndims

  batch_size = int(np.prod(batch_shape))
  example_size = example_dim ** example_ndims
  feature_size = feature_dim ** feature_ndims

  # We would like each batch of examples to be unique, to avoid computing kernel
  # matrices that are semi-definite. hypothesis.extra.numpy.arrays doesn't have
  # a sense of tolerance, so we need to do some extra work to get points
  # sufficiently far from each other.
  grid_size = draw(hps.integers(min_value=10, max_value=100))
  grid_spacing = draw(hps.floats(min_value=1e-2, max_value=2))
  hp.note('Grid size {} and spacing {}'.format(grid_size, grid_spacing))

  def _grid_indices_to_values(grid_indices):
    return (grid_spacing *
            (np.array(grid_indices, dtype=np.float64) - np.float64(grid_size)))

  # We'll construct the result by stacking onto flattened batch, example and
  # feature dims, then reshape to unflatten at the end.
  result = np.zeros([0, example_size, feature_size])
  for _ in range(batch_size):
    seen = set()
    index_array_strategy = hps.tuples(
        *([hps.integers(0, grid_size + 1)] * feature_size)).filter(
            lambda x, seen=seen: x not in seen)  # Default param to sate pylint.
    examples = np.zeros([1, 0, feature_size])
    for _ in range(example_size):
      feature_grid_locations = draw(index_array_strategy)
      seen.add(feature_grid_locations)
      example = _grid_indices_to_values(feature_grid_locations)
      example = example[np.newaxis, np.newaxis, ...]
      examples = np.concatenate([examples, example], axis=1)
    result = np.concatenate([result, examples], axis=0)
  result = np.reshape(result, batch_shape + example_shape + feature_shape)

  if enable_vars and draw(hps.booleans()):
    result = tf.Variable(result, name=name)
    if draw(hps.booleans()):
      result = tfp_hps.defer_and_count_usage(result)
  return result


@hps.composite
def broadcasting_params(draw,
                        kernel_name,
                        batch_shape,
                        event_dim=None,
                        enable_vars=False):
  """Draws a dict of parameters which should yield the given batch shape."""
  if kernel_name not in INSTANTIABLE_BASE_KERNELS:
    raise ValueError('Unknown Kernel name {}'.format(kernel_name))
  params_event_ndims = INSTANTIABLE_BASE_KERNELS.get(kernel_name, {})

  def _constraint(param):
    return constraint_for(kernel_name, param)

  return draw(
      tfp_hps.broadcasting_params(
          batch_shape,
          params_event_ndims,
          event_dim=event_dim,
          enable_vars=enable_vars,
          constraint_fn_for=_constraint,
          mutex_params=MUTEX_PARAMS,
          dtype=np.float64))


def depths():
  # TODO(b/139841600): Increase the depth after we can generate kernel inputs
  # that are not too close to each other.
  return hps.integers(min_value=0, max_value=1)


@hps.composite
def feature_scaleds(
    draw,
    batch_shape=None,
    event_dim=None,
    feature_dim=None,
    feature_ndims=None,
    enable_vars=None,
    depth=None):
  """Strategy for drawing `FeatureScaled` kernels.

  The underlying kernel is drawn from the `kernels` strategy.

  Args:
    draw: Hypothesis strategy sampler supplied by `@hps.composite`.
    batch_shape: An optional `TensorShape`.  The batch shape of the resulting
      Kernel.  Hypothesis will pick a batch shape if omitted.
    event_dim: Optional Python int giving the size of each of the
      kernel's parameters' event dimensions.  This is shared across all
      parameters, permitting square event matrices, compatible location and
      scale Tensors, etc. If omitted, Hypothesis will choose one.
    feature_dim: Optional Python int giving the size of each feature dimension.
      If omitted, Hypothesis will choose one.
    feature_ndims: Optional Python int stating the number of feature dimensions
      inputs will have. If omitted, Hypothesis will choose one.
    enable_vars: TODO(bjp): Make this `True` all the time and put variable
      initialization in slicing_test.  If `False`, the returned parameters are
      all Tensors, never Variables or DeferredTensor.
    depth: Python `int` giving maximum nesting depth of compound kernel.

  Returns:
    kernels: A strategy for drawing `FeatureScaled` kernels with the specified
      `batch_shape` (or an arbitrary one if omitted).
  """
  if depth is None:
    depth = draw(depths())
  if batch_shape is None:
    batch_shape = draw(tfp_hps.shapes())
  if event_dim is None:
    event_dim = draw(hps.integers(min_value=2, max_value=6))
  if feature_dim is None:
    feature_dim = draw(hps.integers(min_value=2, max_value=6))
  if feature_ndims is None:
    feature_ndims = draw(hps.integers(min_value=2, max_value=6))

  base_kernel, kernel_variable_names = draw(kernels(
      batch_shape=batch_shape,
      event_dim=event_dim,
      feature_dim=feature_dim,
      feature_ndims=feature_ndims,
      enable_vars=False,
      depth=depth-1))
  scale_diag = tfp_hps.softplus_plus_eps()(draw(kernel_input(
      batch_shape=batch_shape,
      example_ndims=0,
      feature_dim=feature_dim,
      feature_ndims=feature_ndims)))

  hp.note('Forming FeatureScaled kernel with scale_diag: {} '.format(
      scale_diag))

  if enable_vars and draw(hps.booleans()):
    kernel_variable_names.append('scale_diag')
    scale_diag = tf.Variable(scale_diag, name='scale_diag')
    # Don't enable variable counting. This is because rescaling is
    # done for each input, which will exceed two convert_to_tensor calls.
  result_kernel = tfp.math.psd_kernels.FeatureScaled(
      kernel=base_kernel,
      scale_diag=scale_diag,
      validate_args=True)
  return result_kernel, kernel_variable_names


@hps.composite
def schur_complements(
    draw,
    batch_shape=None,
    event_dim=None,
    feature_dim=None,
    feature_ndims=None,
    enable_vars=None,
    depth=None):
  """Strategy for drawing `SchurComplement` kernels.

  The underlying kernel is drawn from the `kernels` strategy.

  Args:
    draw: Hypothesis strategy sampler supplied by `@hps.composite`.
    batch_shape: An optional `TensorShape`.  The batch shape of the resulting
      Kernel.  Hypothesis will pick a batch shape if omitted.
    event_dim: Optional Python int giving the size of each of the
      kernel's parameters' event dimensions.  This is shared across all
      parameters, permitting square event matrices, compatible location and
      scale Tensors, etc. If omitted, Hypothesis will choose one.
    feature_dim: Optional Python int giving the size of each feature dimension.
      If omitted, Hypothesis will choose one.
    feature_ndims: Optional Python int stating the number of feature dimensions
      inputs will have. If omitted, Hypothesis will choose one.
    enable_vars: TODO(bjp): Make this `True` all the time and put variable
      initialization in slicing_test.  If `False`, the returned parameters are
      all Tensors, never Variables or DeferredTensor.
    depth: Python `int` giving maximum nesting depth of compound kernel.

  Returns:
    kernels: A strategy for drawing `SchurComplement` kernels with the specified
      `batch_shape` (or an arbitrary one if omitted).
  """
  if depth is None:
    depth = draw(depths())
  if batch_shape is None:
    batch_shape = draw(tfp_hps.shapes())
  if event_dim is None:
    event_dim = draw(hps.integers(min_value=2, max_value=6))
  if feature_dim is None:
    feature_dim = draw(hps.integers(min_value=2, max_value=6))
  if feature_ndims is None:
    feature_ndims = draw(hps.integers(min_value=2, max_value=6))

  base_kernel, kernel_variable_names = draw(kernels(
      batch_shape=batch_shape,
      event_dim=event_dim,
      feature_dim=feature_dim,
      feature_ndims=feature_ndims,
      enable_vars=False,
      depth=depth-1))

  # SchurComplement requires the inputs to have one example dimension.
  fixed_inputs = draw(kernel_input(
      batch_shape=batch_shape,
      example_ndims=1,
      feature_dim=feature_dim,
      feature_ndims=feature_ndims))
  # Positive shift to ensure the divisor matrix is PD.
  diag_shift = np.float64(draw(hpnp.arrays(
      dtype=np.float64,
      shape=tensorshape_util.as_list(batch_shape),
      elements=hps.floats(1, 100, allow_nan=False, allow_infinity=False))))

  hp.note('Forming SchurComplement kernel with fixed_inputs: {} '
          'and diag_shift: {}'.format(fixed_inputs, diag_shift))

  schur_complement_params = {
      'fixed_inputs': fixed_inputs,
      'diag_shift': diag_shift
  }
  for param_name in schur_complement_params:
    if enable_vars and draw(hps.booleans()):
      kernel_variable_names.append(param_name)
      schur_complement_params[param_name] = tf.Variable(
          schur_complement_params[param_name], name=param_name)
      if draw(hps.booleans()):
        schur_complement_params[param_name] = tfp_hps.defer_and_count_usage(
            schur_complement_params[param_name])
  result_kernel = tfp.math.psd_kernels.SchurComplement(
      base_kernel=base_kernel,
      fixed_inputs=schur_complement_params['fixed_inputs'],
      diag_shift=schur_complement_params['diag_shift'],
      validate_args=True)
  return result_kernel, kernel_variable_names


@hps.composite
def base_kernels(
    draw,
    kernel_name=None,
    batch_shape=None,
    event_dim=None,
    feature_dim=None,
    feature_ndims=None,
    enable_vars=False):
  """Strategy for drawing kernels that don't depend on other kernels.

  Args:
    draw: Hypothesis function supplied by `@hps.composite`.
    kernel_name: Optional Python `str`.  If given, the produced kernels
      will all have this type.
    batch_shape: An optional `TensorShape`.  The batch shape of the resulting
      Kernel.  Hypothesis will pick a batch shape if omitted.
    event_dim: Optional Python int giving the size of each of the
      kernel's parameters' event dimensions.  This is shared across all
      parameters, permitting square event matrices, compatible location and
      scale Tensors, etc. If omitted, Hypothesis will choose one.
    feature_dim: Optional Python int giving the size of each feature dimension.
      If omitted, Hypothesis will choose one.
    feature_ndims: Optional Python int stating the number of feature dimensions
      inputs will have. If omitted, Hypothesis will choose one.
    enable_vars: TODO(bjp): Make this `True` all the time and put variable
      initialization in slicing_test.  If `False`, the returned parameters are
      all Tensors, never Variables or DeferredTensor.
  Returns:
    kernels: A strategy for drawing Kernels with the specified `batch_shape`
      (or an arbitrary one if omitted).
    kernel_variable_names: List of kernel parameters that are variables.
  """

  if kernel_name is None:
    kernel_name = draw(hps.sampled_from(sorted(INSTANTIABLE_BASE_KERNELS)))
  if batch_shape is None:
    batch_shape = draw(tfp_hps.shapes())
  if event_dim is None:
    event_dim = draw(hps.integers(min_value=2, max_value=6))
  if feature_dim is None:
    feature_dim = draw(hps.integers(min_value=2, max_value=6))
  if feature_ndims is None:
    feature_ndims = draw(hps.integers(min_value=2, max_value=6))

  kernel_params = draw(
      broadcasting_params(kernel_name, batch_shape, event_dim=event_dim,
                          enable_vars=enable_vars))
  kernel_variable_names = [
      k for k in kernel_params if tensor_util.is_ref(kernel_params[k])]
  hp.note('Forming kernel {} with feature_ndims {} and constrained parameters '
          '{}'.format(kernel_name, feature_ndims, kernel_params))
  ctor = getattr(tfpk, kernel_name)
  result_kernel = ctor(
      validate_args=True,
      feature_ndims=feature_ndims,
      **kernel_params)
  if batch_shape != result_kernel.batch_shape:
    msg = ('Kernel strategy generated a bad batch shape '
           'for {}, should have been {}.').format(result_kernel, batch_shape)
    raise AssertionError(msg)
  return result_kernel, kernel_variable_names


@hps.composite
def kernels(
    draw,
    kernel_name=None,
    batch_shape=None,
    event_dim=None,
    feature_dim=None,
    feature_ndims=None,
    enable_vars=False,
    depth=None):
  """Strategy for drawing arbitrary Kernels.

  Args:
    draw: Hypothesis function supplied by `@hps.composite`.
    kernel_name: Optional Python `str`.  If given, the produced kernels
      will all have this type.
    batch_shape: An optional `TensorShape`.  The batch shape of the resulting
      Kernel.  Hypothesis will pick a batch shape if omitted.
    event_dim: Optional Python int giving the size of each of the
      kernel's parameters' event dimensions.  This is shared across all
      parameters, permitting square event matrices, compatible location and
      scale Tensors, etc. If omitted, Hypothesis will choose one.
    feature_dim: Optional Python int giving the size of each feature dimension.
      If omitted, Hypothesis will choose one.
    feature_ndims: Optional Python int stating the number of feature dimensions
      inputs will have. If omitted, Hypothesis will choose one.
    enable_vars: TODO(bjp): Make this `True` all the time and put variable
      initialization in slicing_test.  If `False`, the returned parameters are
      all Tensors, never Variables or DeferredTensor.
    depth: Python `int` giving maximum nesting depth of compound kernel.
  Returns:
    kernels: A strategy for drawing Kernels with the specified `batch_shape`
      (or an arbitrary one if omitted).
    kernel_variable_names: List of kernel parameters that are variables.
  """

  if depth is None:
    depth = draw(depths())
  if kernel_name is None and depth > 0:
    bases = hps.just(None)
    compounds = hps.sampled_from(SPECIAL_KERNELS)
    kernel_name = draw(hps.one_of([bases, compounds]))
  if kernel_name is None or kernel_name in INSTANTIABLE_BASE_KERNELS:
    return draw(
        base_kernels(
            kernel_name,
            batch_shape=batch_shape,
            event_dim=event_dim,
            feature_dim=feature_dim,
            feature_ndims=feature_ndims,
            enable_vars=enable_vars))

  if kernel_name == 'SchurComplement':
    return draw(schur_complements(
        batch_shape=batch_shape,
        event_dim=event_dim,
        feature_dim=feature_dim,
        feature_ndims=feature_ndims,
        enable_vars=enable_vars,
        depth=depth))
  elif kernel_name == 'FeatureScaled':
    return draw(feature_scaleds(
        batch_shape=batch_shape,
        event_dim=event_dim,
        feature_dim=feature_dim,
        feature_ndims=feature_ndims,
        enable_vars=enable_vars,
        depth=depth))

  raise ValueError('Kernel name not found.')


# This will be used for most positive parameters to ensure matrices
# are well-conditioned.
def constrain_to_range(low, high):
  return lambda x: (high - low) * tf.math.sigmoid(x) + low


CONSTRAINTS = {
    # Keep parameters large enough but not too large so matrices are
    # well-conditioned. The ranges below were chosen to ensure kernel
    # matrices are positive definite.
    'amplitude': constrain_to_range(1., 2.),
    'bias_variance': constrain_to_range(0.1, 0.5),
    'slope_variance': constrain_to_range(0.1, 0.5),
    'exponent': constrain_to_range(1, 1.5),
    'length_scale': constrain_to_range(1., 6.),
    'period': constrain_to_range(1., 6.),
    'scale_mixture_rate': constrain_to_range(1., 6.),
    # Ensure shift isn't too large such that all inputs are mapped
    # to the same place.
    'shift': lambda x: 5. * tf.math.tanh(x)
}


def constraint_for(kernel_name=None, param=None):
  if param is not None:
    return CONSTRAINTS.get('{}.{}'.format(kernel_name, param),
                           CONSTRAINTS.get(param, tfp_hps.identity_fn))
  return CONSTRAINTS.get(kernel_name, tfp_hps.identity_fn)
