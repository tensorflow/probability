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
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.math.psd_kernels import hypothesis_testlib as kernel_hps


TF2_FRIENDLY_KERNELS = (
    'ExpSinSquared',
    'ExponentiatedQuadratic',
    'FeatureScaled',
    'Linear',
    'MaternOneHalf',
    'MaternThreeHalves',
    'MaternFiveHalves',
    # TODO(b/146073659): Polynomial as currently configured often produces
    # numerically ill-conditioned matrices. Disabled until we can make it more
    # reliable in the context of hypothesis tests.
    # 'Polynomial',
    'RationalQuadratic',
    'SchurComplement',
)


# pylint is unable to handle @hps.composite (e.g. complains "No value for
# argument '...' in function call"), so disable this lint for the file.

# pylint: disable=no-value-for-parameter


def assert_no_none_grad(kernel, method, wrt_vars, grads):
  for var, grad in zip(wrt_vars, grads):
    if grad is None:
      raise AssertionError('Missing `{}` -> {} grad for kernel {}'.format(
          method, var, kernel))


@test_util.test_all_tf_execution_regimes
class KernelPropertiesTest(test_util.TestCase):

  @parameterized.named_parameters(dict(testcase_name=kname, kernel_name=kname)
                                  for kname in TF2_FRIENDLY_KERNELS)
  @hp.given(hps.data())
  @tfp_hps.tfp_hp_settings(
      default_max_examples=10,
      suppress_health_check=[
          hp.HealthCheck.too_slow,
          hp.HealthCheck.data_too_large])
  def testKernelGradient(self, kernel_name, data):
    event_dim = data.draw(hps.integers(min_value=2, max_value=4))
    feature_ndims = data.draw(hps.integers(min_value=1, max_value=2))
    feature_dim = data.draw(hps.integers(min_value=2, max_value=4))

    kernel, kernel_parameter_variable_names = data.draw(
        kernel_hps.kernels(
            kernel_name=kernel_name,
            event_dim=event_dim,
            feature_dim=feature_dim,
            feature_ndims=feature_ndims,
            enable_vars=True))

    # Check that variable parameters get passed to the kernel.variables
    kernel_variables_names = [
        v.name.strip('_0123456789:') for v in kernel.variables]
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

    with tf.GradientTape() as tape:
      with tfp_hps.assert_no_excessive_var_usage(
          'method `apply` of {}'.format(kernel)):
        tape.watch(wrt_vars)
        with tfp_hps.no_tf_rank_errors():
          diag = kernel.apply(xs, xs, example_ndims=example_ndims)
    grads = tape.gradient(diag, wrt_vars)
    assert_no_none_grad(kernel, 'apply', wrt_vars, grads)

    self.assertAllClose(
        diag,
        type(kernel)(**kernel._parameters).apply(
            xs, xs, example_ndims=example_ndims))


CONSTRAINTS = {
    # Keep amplitudes large enough so that the matrices are well conditioned.
    'amplitude': tfp_hps.softplus_plus_eps(1.),
    'bias_variance': tfp_hps.softplus_plus_eps(1.),
    'slope_variance': tfp_hps.softplus_plus_eps(1.),
    'exponent': tfp_hps.softplus_plus_eps(),
    'length_scale': tfp_hps.softplus_plus_eps(),
    'period': tfp_hps.softplus_plus_eps(),
    'scale_mixture_rate': tfp_hps.softplus_plus_eps(),
}


def constraint_for(kernel_name=None, param=None):
  if param is not None:
    return CONSTRAINTS.get('{}.{}'.format(kernel_name, param),
                           CONSTRAINTS.get(param, tfp_hps.identity_fn))
  return CONSTRAINTS.get(kernel_name, tfp_hps.identity_fn)


if __name__ == '__main__':
  tf.enable_v2_behavior()
  tf.test.main()
