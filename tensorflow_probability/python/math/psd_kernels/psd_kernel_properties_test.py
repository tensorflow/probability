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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
from absl.testing import parameterized
import hypothesis as hp
from hypothesis import strategies as hps
from hypothesis.extra import numpy as hpnp
import numpy as np

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.internal import hypothesis_testlib as tfp_hps
from tensorflow_probability.python.internal import tensor_util
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


flags.DEFINE_enum('tf_mode', 'graph', ['eager', 'graph'],
                  'TF execution mode to use')

FLAGS = flags.FLAGS

TF2_FRIENDLY_KERNELS = (
    'ExpSinSquared',
    'ExponentiatedQuadratic',
    'MaternOneHalf',
    'MaternThreeHalves',
    'MaternFiveHalves',
    'RationalQuadratic',
)

KERNEL_PARAMS_NDIMS = {
    'ExpSinSquared': dict(amplitude=0, length_scale=0, period=0),
    'ExponentiatedQuadratic': dict(amplitude=0, length_scale=0),
    'Linear': dict(bias_variance=0, slope_variance=0, shift=1),
    'MaternOneHalf': dict(amplitude=0, length_scale=0),
    'MaternThreeHalves': dict(amplitude=0, length_scale=0),
    'MaternFiveHalves': dict(amplitude=0, length_scale=0),
    'Polynomial': dict(bias_variance=0, slope_variance=0, exponent=0, shift=1),
    'RationalQuadratic': dict(
        amplitude=0, length_scale=0, scale_mixture_rate=0),
}
MUTEX_PARAMS = tuple()


# pylint is unable to handle @hps.composite (e.g. complains "No value for
# argument '...' in function call"), so disable this lint for the file.

# pylint: disable=no-value-for-parameter


@hps.composite
def broadcasting_params(draw,
                        kernel_name,
                        batch_shape,
                        event_dim=None,
                        enable_vars=False):
  """Draws a dict of parameters which should yield the given batch shape."""
  params_event_ndims = KERNEL_PARAMS_NDIMS.get(kernel_name, {})

  def _constraint(param):
    return constraint_for(kernel_name, param)

  return draw(
      tfp_hps.broadcasting_params(
          batch_shape,
          params_event_ndims,
          event_dim=event_dim,
          enable_vars=enable_vars,
          constraint_fn_for=_constraint,
          mutex_params=MUTEX_PARAMS))


@hps.composite
def kernel_input(
    draw,
    batch_shape,
    example_dim=None,
    example_ndims=None,
    feature_dim=None,
    feature_ndims=None):
  """Strategy for drawing arbitrary Kernel input.

  Args:
    draw: Hypothesis function supplied by `@hps.composite`.
    batch_shape: An optional `TensorShape`.  The batch shape of the resulting
      kernel input.  Hypothesis will pick a batch shape if omitted.
    example_dim: Optional Python int giving the size of each example dimension.
      If omitted, Hypothesis will choose one.
    example_ndims: Optional Python int giving the number of example dimensions
      of the input. If omitted, Hypothesis will choose one.
    feature_dim: Optional Python int giving the size of each feature dimension.
      If omitted, Hypothesis will choose one.
    feature_ndims: Optional Python int stating the number of feature dimensions
      inputs will have. If omitted, Hypothesis will choose one.
  Returns:
    kernel_input: A strategy for drawing kernel_input with the prescribed shape
      (or an arbitrary one if omitted).
  """
  if example_ndims is None:
    example_ndims = draw(hps.integers(min_value=1, max_value=4))
  if example_dim is None:
    example_dim = draw(hps.integers(min_value=2, max_value=6))

  if feature_ndims is None:
    feature_ndims = draw(hps.integers(min_value=1, max_value=4))
  if feature_dim is None:
    feature_dim = draw(hps.integers(min_value=2, max_value=6))

  input_shape = batch_shape
  input_shape += [example_dim] * example_ndims
  input_shape += [feature_dim] * feature_ndims
  return draw(hpnp.arrays(dtype=np.float32, shape=input_shape))


@hps.composite
def kernels(
    draw,
    kernel_name=None,
    batch_shape=None,
    event_dim=None,
    feature_ndims=None,
    enable_vars=False):
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
    kernel_name = draw(hps.sampled_from(TF2_FRIENDLY_KERNELS))
  if batch_shape is None:
    batch_shape = draw(tfp_hps.shapes())
  if event_dim is None:
    event_dim = draw(hps.integers(min_value=2, max_value=6))
  if feature_ndims is None:
    feature_ndims = draw(hps.integers(min_value=2, max_value=6))

  kernel_params = draw(
      broadcasting_params(kernel_name, batch_shape, event_dim=event_dim,
                          enable_vars=enable_vars))
  kernel_variable_names = [
      k for k in kernel_params if tensor_util.is_ref(kernel_params[k])]
  ctor = getattr(tfp.math.psd_kernels, kernel_name)
  result_kernel = ctor(
      validate_args=True,
      feature_ndims=feature_ndims,
      **kernel_params)

  if batch_shape != result_kernel.batch_shape:
    msg = ('Kernel strategy generated a bad batch shape '
           'for {}, should have been {}.').format(result_kernel, batch_shape)
    raise AssertionError(msg)
  return result_kernel, kernel_variable_names


def assert_no_none_grad(kernel, method, wrt_vars, grads):
  for var, grad in zip(wrt_vars, grads):
    if grad is None:
      raise AssertionError('Missing `{}` -> {} grad for kernel {}'.format(
          method, var, kernel))


@test_util.run_all_in_graph_and_eager_modes
class KernelPropertiesTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters((bname,) for bname in TF2_FRIENDLY_KERNELS)
  @hp.given(hps.data())
  @tfp_hps.tfp_hp_settings(default_max_examples=10)
  def testKernelGradient(self, kernel_name, data):
    if tf.executing_eagerly() != (FLAGS.tf_mode == 'eager'):
      return
    event_dim = data.draw(hps.integers(min_value=2, max_value=6))
    feature_ndims = data.draw(hps.integers(min_value=1, max_value=4))
    kernel, kernel_parameter_variable_names = data.draw(
        kernels(
            kernel_name=kernel_name,
            event_dim=event_dim,
            feature_ndims=feature_ndims,
            enable_vars=True))

    # Check that variable parameters get passed to the kernel.variables
    kernel_variables_names = [
        v.name.strip('_0123456789:') for v in kernel.variables]
    self.assertEqual(
        set(kernel_parameter_variable_names),
        set(kernel_variables_names))

    example_ndims = data.draw(hps.integers(min_value=1, max_value=3))
    input_batch_shape = data.draw(tfp_hps.broadcast_compatible_shape(
        kernel.batch_shape))
    xs = tf.identity(data.draw(kernel_input(
        batch_shape=input_batch_shape,
        example_ndims=example_ndims,
        feature_ndims=feature_ndims)))

    # Check that we pick up all relevant kernel parameters.
    wrt_vars = [xs] + list(kernel.variables)

    with tf.GradientTape() as tape:
      with tfp_hps.assert_no_excessive_var_usage(
          'method `apply` of {}'.format(kernel)):
        tape.watch(wrt_vars)
        diag = kernel.apply(xs, xs, example_ndims=example_ndims)
    grads = tape.gradient(diag, wrt_vars)
    assert_no_none_grad(kernel, 'apply', wrt_vars, grads)


CONSTRAINTS = {
    'amplitude': tfp_hps.softplus_plus_eps(),
    'length_scale': tfp_hps.softplus_plus_eps(),
    'period': tfp_hps.softplus_plus_eps(),
    'scale_mixture_rate': tfp_hps.softplus_plus_eps(),
    'bias_variance': tfp_hps.softplus_plus_eps(),
    'slope_variance': tfp_hps.softplus_plus_eps(),
    'exponent': tfp_hps.softplus_plus_eps(),
}


def constraint_for(kernel_name=None, param=None):
  if param is not None:
    return CONSTRAINTS.get('{}.{}'.format(kernel_name, param),
                           CONSTRAINTS.get(param, tfp_hps.identity_fn))
  return CONSTRAINTS.get(kernel_name, tfp_hps.identity_fn)


if __name__ == '__main__':
  tf.enable_v2_behavior()
  tf.test.main()
