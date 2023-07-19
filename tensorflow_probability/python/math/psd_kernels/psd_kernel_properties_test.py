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

import collections

from absl.testing import parameterized
import hypothesis as hp
from hypothesis import strategies as hps
import numpy as np

import tensorflow.compat.v2 as tf
from tensorflow_probability.python.internal import hypothesis_testlib as tfp_hps
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.math.psd_kernels import hypothesis_testlib as kernel_hps


# pylint is unable to handle @hps.composite (e.g. complains "No value for
# argument '...' in function call"), so disable this lint for the file.

# pylint: disable=no-value-for-parameter

EXTRA_TENSOR_CONVERSION_KERNELS = {
    # The transformation is applied to each input individually.
    'KumaraswamyTransformed': 1,
}

INSTANTIABLE_BUT_NOT_SLICABLE = [
    'FeatureTransformed',  # Requires slicing in to the `transformation_fn`.
]


KERNELS_OK_TO_SLICE = (set(list(kernel_hps.INSTANTIABLE_BASE_KERNELS.keys()) +
                           list(kernel_hps.SPECIAL_KERNELS)) -
                       set(INSTANTIABLE_BUT_NOT_SLICABLE))


def assert_no_none_grad(kernel, method, wrt_vars, grads):
  for var, grad in zip(wrt_vars, grads):
    # For the GeneralizedMatern kernel, gradients with respect to `df` don't
    # exist.
    if tensor_util.is_ref(var) and var.name.strip('_0123456789:') == 'df':
      continue
    if grad is None:
      raise AssertionError('Missing `{}` -> {} grad for kernel {}'.format(
          method, var, kernel))


@test_util.test_all_tf_execution_regimes
class KernelPropertiesTest(test_util.TestCase):

  @parameterized.named_parameters(
      {'testcase_name': dname, 'kernel_name': dname}
      for dname in sorted(list(kernel_hps.INSTANTIABLE_BASE_KERNELS.keys()) +
                          list(kernel_hps.SPECIAL_KERNELS)))
  @hp.given(hps.data())
  @tfp_hps.tfp_hp_settings(
      default_max_examples=10,
      suppress_health_check=[
          hp.HealthCheck.too_slow,
          hp.HealthCheck.data_too_large])
  def testKernelGradient(self, kernel_name, data):
    event_dim = data.draw(hps.integers(min_value=2, max_value=3))
    feature_ndims = data.draw(hps.integers(min_value=0, max_value=2))
    feature_dim = data.draw(hps.integers(min_value=2, max_value=4))
    batch_shape = data.draw(tfp_hps.shapes(max_ndims=2))

    kernel, kernel_parameter_variable_names = data.draw(
        kernel_hps.kernels(
            batch_shape=batch_shape,
            kernel_name=kernel_name,
            event_dim=event_dim,
            feature_dim=feature_dim,
            feature_ndims=feature_ndims,
            enable_vars=True))

    # Check that variable parameters get passed to the kernel.variables
    kernel_variables_names = [
        v.name.strip('_0123456789:') for v in kernel.variables]
    kernel_parameter_variable_names = [
        n.strip('_0123456789:') for n in kernel_parameter_variable_names]
    self.assertEqual(
        set(kernel_parameter_variable_names),
        set(kernel_variables_names))

    example_ndims = data.draw(hps.integers(min_value=1, max_value=2))
    input_batch_shape = data.draw(tfp_hps.broadcast_compatible_shape(
        kernel.batch_shape))
    xs = tf.identity(data.draw(kernel_hps.kernel_input(
        batch_shape=input_batch_shape,
        example_ndims=example_ndims,
        feature_dim=feature_dim,
        feature_ndims=feature_ndims)))

    # Check that we pick up all relevant kernel parameters.
    wrt_vars = [xs] + list(kernel.variables)
    self.evaluate([v.initializer for v in kernel.variables])

    max_permissible = 2 + EXTRA_TENSOR_CONVERSION_KERNELS.get(kernel_name, 0)

    with tf.GradientTape() as tape:
      with tfp_hps.assert_no_excessive_var_usage(
          'method `apply` of {}'.format(kernel),
          max_permissible=max_permissible
      ):
        tape.watch(wrt_vars)
        with tfp_hps.no_tf_rank_errors():
          diag = kernel.apply(xs, xs, example_ndims=example_ndims)
    grads = tape.gradient(diag, wrt_vars)
    assert_no_none_grad(kernel, 'apply', wrt_vars, grads)

    # Check that copying the kernel works.
    with tfp_hps.no_tf_rank_errors():
      diag2 = self.evaluate(kernel.copy().apply(
          xs, xs, example_ndims=example_ndims))
    self.assertAllClose(diag, diag2)

  @parameterized.named_parameters(
      {'testcase_name': dname, 'kernel_name': dname}
      for dname in sorted(list(kernel_hps.INSTANTIABLE_BASE_KERNELS.keys()) +
                          list(kernel_hps.SPECIAL_KERNELS)))
  @hp.given(hps.data())
  @tfp_hps.tfp_hp_settings(
      default_max_examples=10,
      suppress_health_check=[
          hp.HealthCheck.too_slow,
          hp.HealthCheck.data_too_large])
  def testCompositeTensor(self, kernel_name, data):
    kernel, _ = data.draw(
        kernel_hps.kernels(
            kernel_name=kernel_name,
            event_dim=2,
            feature_dim=2,
            feature_ndims=1,
            enable_vars=True))
    self.assertIsInstance(kernel, tf.__internal__.CompositeTensor)

    xs = tf.identity(data.draw(kernel_hps.kernel_input(
        batch_shape=[],
        example_ndims=1,
        feature_dim=2,
        feature_ndims=1)))
    with tfp_hps.no_tf_rank_errors():
      diag = kernel.apply(xs, xs, example_ndims=1)

    # Test flatten/unflatten.
    flat = tf.nest.flatten(kernel, expand_composites=True)
    unflat = tf.nest.pack_sequence_as(kernel, flat, expand_composites=True)

    # Test tf.function.
    @tf.function
    def diag_fn(k):
      return k.apply(xs, xs, example_ndims=1)

    self.evaluate([v.initializer for v in kernel.variables])
    with tfp_hps.no_tf_rank_errors():
      self.assertAllClose(diag, diag_fn(kernel))
      self.assertAllClose(diag, diag_fn(unflat))

    self.assertConvertVariablesToTensorsWorks(kernel)


@test_util.test_all_tf_execution_regimes
class PSDKernelSlicingTest(test_util.TestCase):

  def _test_slicing(
      self,
      data,
      kernel_name,
      kernel,
      feature_dim,
      feature_ndims):
    example_ndims = data.draw(hps.integers(min_value=0, max_value=2))
    batch_shape = kernel.batch_shape
    slices = data.draw(tfp_hps.valid_slices(batch_shape))
    slice_str = 'kernel[{}]'.format(', '.join(tfp_hps.stringify_slices(
        slices)))
    # Make sure the slice string appears in Hypothesis' attempted example log
    hp.note('Using slice ' + slice_str)
    if not slices:  # Nothing further to check.
      return
    sliced_zeros = np.zeros(batch_shape)[slices]
    sliced_kernel = kernel[slices]
    hp.note('Using sliced kernel {}.'.format(sliced_kernel))
    hp.note('Using sliced zeros {}.'.format(sliced_zeros.shape))

    # Check that slicing modifies batch shape as expected.
    self.assertAllEqual(sliced_zeros.shape, sliced_kernel.batch_shape)

    xs = tf.identity(data.draw(kernel_hps.kernel_input(
        batch_shape=[],
        example_ndims=example_ndims,
        feature_dim=feature_dim,
        feature_ndims=feature_ndims)))

    # Check that apply of sliced kernels executes.
    with tfp_hps.no_tf_rank_errors():
      results = self.evaluate(kernel.apply(xs, xs, example_ndims=example_ndims))
      hp.note('Using results shape {}.'.format(results.shape))
      sliced_results = self.evaluate(
          sliced_kernel.apply(xs, xs, example_ndims=example_ndims))

    # Come up with the slices for apply (which must also include example dims).
    apply_slices = (
        tuple(slices) if isinstance(slices, collections.abc.Sequence) else
        (slices,))
    apply_slices += tuple([slice(None)] * example_ndims)

    # Check that sampling a sliced kernel produces the same shape as
    # slicing the samples from the original.
    self.assertAllClose(results[apply_slices], sliced_results)

  @parameterized.named_parameters(
      {'testcase_name': dname, 'kernel_name': dname}
      for dname in sorted(KERNELS_OK_TO_SLICE))
  @hp.given(hps.data())
  @tfp_hps.tfp_hp_settings()
  def testKernels(self, kernel_name, data):
    event_dim = data.draw(hps.integers(min_value=2, max_value=3))
    feature_ndims = data.draw(hps.integers(min_value=0, max_value=2))
    feature_dim = data.draw(hps.integers(min_value=2, max_value=4))

    kernel, _ = data.draw(
        kernel_hps.kernels(
            kernel_name=kernel_name,
            event_dim=event_dim,
            feature_dim=feature_dim,
            feature_ndims=feature_ndims,
            enable_vars=False))

    # Check that all kernels still register as non-iterable despite
    # defining __getitem__.  (Because __getitem__ magically makes an object
    # iterable for some reason.)
    with self.assertRaisesRegex(TypeError, 'not iterable'):
      iter(kernel)

    # Test slicing
    self._test_slicing(data, kernel_name, kernel, feature_dim, feature_ndims)

if __name__ == '__main__':
  test_util.main()
