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

from absl.testing import parameterized
import hypothesis as hp
from hypothesis import strategies as hps
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
    feature_ndims = data.draw(hps.integers(min_value=1, max_value=2))
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

    # Check that reconstructing the kernel works
    with tfp_hps.no_tf_rank_errors():
      diag2 = self.evaluate(type(kernel)(**kernel._parameters).apply(
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


if __name__ == '__main__':
  test_util.main()
