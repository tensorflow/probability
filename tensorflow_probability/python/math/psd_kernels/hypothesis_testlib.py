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

import collections
import contextlib
import inspect
import logging
import re

import hypothesis as hp
from hypothesis.extra import numpy as hpnp
import hypothesis.strategies as hps
import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow_probability.python.experimental.distributions import marginal_fns
from tensorflow_probability.python.internal import hypothesis_testlib as tfp_hps
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.math import psd_kernels as tfpk


SPECIAL_KERNELS = [
    'ChangePoint',
    'FeatureScaled',
    'KumaraswamyTransformed',
    'PointwiseExponential',
    'SchurComplement'
]


NON_INSTANTIABLE_SPECIAL_KERNELS = [
    'AutoCompositeTensorPsdKernel',
    'ExponentialCurve',  # TODO(jburnim, srvasude): Enable this kernel.
    'FeatureTransformed',
    'PositiveSemidefiniteKernel',
]


class KernelInfo(collections.namedtuple(
    'KernelInfo', ['cls', 'params_event_ndims'])):
  """Sufficient information to instantiate a Kernel.

  To wit

  - The Python class `cls` giving the class, and
  - A Python dict `params_event_ndims` giving the event dimensions for the
    parameters (so that parameters can be built with predictable batch shapes).

  Specifically, the `params_event_ndims` dict maps string parameter names to
  Python integers.  Each integer gives how many (trailing) dimensions of that
  parameter are part of the event.
  """
  __slots__ = ()


def _instantiable_base_kernels():
  """Computes the table of mechanically instantiable base Kernels.

  A Kernel is mechanically instantiable if

  - The class appears as a symbol binding in `tfp.math.psd_kernels`;
  - The class defines a `_params_event_ndims` method (necessary
    to generate parameter Tensors with predictable batch shapes); and
  - The name is not blocklisted in `SPECIAL_KERNELS`.

  Compound kernels have their own
  instantiation rules hard-coded in the `kernel` strategy.

  Returns:
    instantiable_base_kernels: A Python dict mapping kernel name
      (as a string) to a `KernelInfo` carrying the information necessary to
      instantiate it.
  """
  result = {}
  for kernel_name in dir(tfpk):
    kernel_class = getattr(tfpk, kernel_name)
    if (not inspect.isclass(kernel_class) or
        not issubclass(kernel_class, tfpk.PositiveSemidefiniteKernel) or
        kernel_name in SPECIAL_KERNELS or
        kernel_name in NON_INSTANTIABLE_SPECIAL_KERNELS):
      continue
    try:
      params_event_ndims = {
          k: p.event_ndims
          for (k, p) in kernel_class.parameter_properties().items()
          if p.is_tensor and p.event_ndims is not None
      }
    except NotImplementedError:
      logging.warning(
          'Unable to test tfd.%s: `parameter_properties()` is not '
          'implemented or does not define concrete (integer) `event_ndims` '
          'for all parameters.',
          kernel_name)
    result[kernel_name] = KernelInfo(kernel_class, params_event_ndims)

  return result


# INSTANTIABLE_BASE_KERNELS is a map from str->(KernelClass, params_event_ndims)
INSTANTIABLE_BASE_KERNELS = _instantiable_base_kernels()
del _instantiable_base_kernels


MUTEX_PARAMS = (
    set(['length_scale', 'inverse_length_scale']),
    set(['scale_diag', 'inverse_scale_diag']),
)

# pylint is unable to handle @hps.composite (e.g. complains "No value for
# argument 'batch_shape' in function call"), so disable this lint for the file.

# pylint: disable=no-value-for-parameter


class ConstrainToUnit(tfpk.FeatureTransformed):
  """Constrain inputs to `[0, 1]`."""

  def __init__(self, kernel, validate_args=False):
    parameters = dict(locals())
    self._kernel = kernel

    super(ConstrainToUnit, self).__init__(
        kernel,
        transformation_fn=lambda x, f, e: tf.math.sigmoid(x),
        validate_args=validate_args,
        parameters=parameters)

  @property
  def kernel(self):
    return self._kernel

  def _batch_shape(self):
    return self.kernel.batch_shape

  def _batch_shape_tensor(self):
    return self.kernel.batch_shape_tensor()

  def __getitem__(self, slices):
    overrides = {}
    if self.parameters.get('kernel', None) is not None:
      overrides['kernel'] = self.kernel[slices]

    return self.copy(**overrides)

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
    return dict(kernel=parameter_properties.BatchedComponentProperties())


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


@contextlib.contextmanager
def no_pd_errors():
  """Catch and ignore examples where a Cholesky decomposition fails.

  This will typically occur when the matrix is not positive definite.

  Yields:
    None
  """
  # TODO(b/174591555): Instead of catching and `assume`ing away positive
  # definite errors, avoid them in the first place.
  try:
    yield
  except tf.errors.OpError as e:
    # NOTE: When tf.linalg.cholesky fails, it returns a matrix with nans on and
    # below the diagonal.  When we use the Cholesky decomposition in a solve,
    # TF will raise an error that the matrix of nans is not invertible.
    if re.search(r'Input matrix is not invertible', str(e)):
      hp.assume(False)
    else:
      raise


@hps.composite
def broadcasting_params(draw,
                        kernel_name,
                        batch_shape,
                        event_dim=None,
                        enable_vars=False):
  """Draws a dict of parameters which should yield the given batch shape."""
  if kernel_name not in INSTANTIABLE_BASE_KERNELS:
    raise ValueError('Unknown Kernel name {}'.format(kernel_name))
  params_event_ndims = INSTANTIABLE_BASE_KERNELS[kernel_name].params_event_ndims

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
def changepoints(
    draw,
    batch_shape=None,
    event_dim=None,
    feature_dim=None,
    feature_ndims=None,
    enable_vars=None,
    depth=None):
  """Strategy for drawing `Changepoint` kernels.

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
    kernels: A strategy for drawing `Changepoint` kernels with the specified
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

  num_kernels = draw(hps.integers(min_value=2, max_value=4))

  inner_kernels = []
  kernel_variable_names = []
  for _ in range(num_kernels):
    base_kernel, variable_names = draw(kernels(
        batch_shape=batch_shape,
        event_dim=event_dim,
        feature_dim=feature_dim,
        feature_ndims=feature_ndims,
        enable_vars=False,
        depth=depth-1))
    inner_kernels.append(base_kernel)
    kernel_variable_names += variable_names

  constraints = dict(
      locs=lambda x: tf.cumsum(tf.math.abs(x) + 1e-3, axis=-1),
      slopes=tfp_hps.softplus_plus_eps())

  params = draw(tfp_hps.broadcasting_params(
      batch_shape,
      event_dim=num_kernels - 1,
      params_event_ndims=dict(locs=1, slopes=1),
      constraint_fn_for=constraints.get))
  params = {k: tf.cast(params[k], tf.float64) for k in params}

  if enable_vars and draw(hps.booleans()):
    kernel_variable_names.append('locs')
    kernel_variable_names.append('slopes')
    params['locs'] = tf.Variable(params['locs'], name='locs')
    params['slopes'] = tf.Variable(params['slopes'], name='slopes')
  result_kernel = tfpk.ChangePoint(
      kernels=inner_kernels, validate_args=True, **params)
  return result_kernel, kernel_variable_names


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
  result_kernel = tfpk.FeatureScaled(
      kernel=base_kernel,
      scale_diag=scale_diag,
      validate_args=True)
  return result_kernel, kernel_variable_names


@hps.composite
def feature_transformeds(
    draw,
    batch_shape=None,
    event_dim=None,
    feature_dim=None,
    feature_ndims=None,
    enable_vars=None,
    depth=None):
  """Strategy for drawing `FeatureTransformed` kernels.

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
    kernels: A strategy for drawing `FeatureTransformed` kernels with the
      specified `batch_shape` (or an arbitrary one if omitted).
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
      enable_vars=enable_vars,
      depth=depth-1))

  hp.note('Forming FeatureTransformed kernel')

  result_kernel = tfpk.FeatureTransformed(
      kernel=base_kernel,
      transformation_fn=lambda x, feature_ndims, example_ndims: x ** 2.,
      validate_args=True)

  return result_kernel, kernel_variable_names


@hps.composite
def kumaraswamy_transformeds(
    draw,
    batch_shape=None,
    event_dim=None,
    feature_dim=None,
    feature_ndims=None,
    enable_vars=None,
    depth=None):
  """Strategy for drawing `KumaraswamyTransformed` kernels.

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
    kernels: A strategy for drawing `KumaraswamyTransformed` kernels with the
      specified `batch_shape` (or an arbitrary one if omitted).
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

  base_kernel, _ = draw(kernels(
      batch_shape=batch_shape,
      event_dim=event_dim,
      feature_dim=feature_dim,
      feature_ndims=feature_ndims,
      enable_vars=False,
      depth=depth-1))

  concentration1 = constrain_to_range(1., 2.)(draw(kernel_input(
      batch_shape=batch_shape,
      example_ndims=0,
      feature_dim=feature_dim,
      feature_ndims=feature_ndims)))

  concentration0 = constrain_to_range(1., 2.)(draw(kernel_input(
      batch_shape=batch_shape,
      example_ndims=0,
      feature_dim=feature_dim,
      feature_ndims=feature_ndims)))

  concentrations = {
      'concentration1': concentration1,
      'concentration0': concentration0
  }

  kernel_variable_names = []

  for param_name in concentrations:
    if enable_vars and draw(hps.booleans()):
      kernel_variable_names.append(param_name)
      concentrations[param_name] = tf.Variable(
          concentrations[param_name], name=param_name)
      if draw(hps.booleans()):
        concentrations[param_name] = tfp_hps.defer_and_count_usage(
            concentrations[param_name])

  hp.note('Forming KumaraswamyTransformed kernel with '
          'concentrations: {}'.format(concentrations))

  # We compose with a FeatureTransformed to ensure inputs are positive to
  # Kumaraswamy.

  result_kernel = tfpk.KumaraswamyTransformed(
      kernel=base_kernel, validate_args=True, **concentrations)
  result_kernel = ConstrainToUnit(kernel=result_kernel, validate_args=True)
  return result_kernel, kernel_variable_names


@hps.composite
def pointwise_exponentials(
    draw,
    batch_shape=None,
    event_dim=None,
    feature_dim=None,
    feature_ndims=None,
    enable_vars=None,
    depth=None):
  """Strategy for drawing `PointwiseExponential` kernels.

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
      enable_vars=enable_vars,
      depth=depth-1))

  hp.note('Forming PointwiseExponential kernel.')

  result_kernel = tfpk.PointwiseExponential(
      kernel=base_kernel, validate_args=True)
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
  result_kernel = tfpk.SchurComplement(
      base_kernel=base_kernel,
      fixed_inputs=schur_complement_params['fixed_inputs'],
      diag_shift=schur_complement_params['diag_shift'],
      cholesky_fn=lambda x: marginal_fns.retrying_cholesky(x)[0],
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

  if kernel_name == 'ChangePoint':
    return draw(changepoints(
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
  elif kernel_name == 'FeatureTransformed':
    return draw(feature_transformeds(
        batch_shape=batch_shape,
        event_dim=event_dim,
        feature_dim=feature_dim,
        feature_ndims=feature_ndims,
        enable_vars=enable_vars,
        depth=depth))
  elif kernel_name == 'KumaraswamyTransformed':
    return draw(kumaraswamy_transformeds(
        batch_shape=batch_shape,
        event_dim=event_dim,
        feature_dim=feature_dim,
        feature_ndims=feature_ndims,
        enable_vars=enable_vars,
        depth=depth))
  elif kernel_name == 'PointwiseExponential':
    return draw(pointwise_exponentials(
        batch_shape=batch_shape,
        event_dim=event_dim,
        feature_dim=feature_dim,
        feature_ndims=feature_ndims,
        enable_vars=enable_vars,
        depth=depth))
  elif kernel_name == 'SchurComplement':
    return draw(schur_complements(
        batch_shape=batch_shape,
        event_dim=event_dim,
        feature_dim=feature_dim,
        feature_ndims=feature_ndims,
        enable_vars=enable_vars,
        depth=depth))

  raise ValueError('Kernel name {} not found.'.format(kernel_name))


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
    'constant': constrain_to_range(0.1, 0.5),
    'concentration0': constrain_to_range(1., 2.),
    'concentration1': constrain_to_range(1., 2.),
    'df': constrain_to_range(2., 5.),
    'slope_variance': constrain_to_range(0.1, 0.5),
    'exponent': lambda x: tf.math.floor(constrain_to_range(1, 4.)(x)),
    'length_scale': constrain_to_range(1., 6.),
    'inverse_length_scale': constrain_to_range(0., 2.),
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
