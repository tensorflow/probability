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
# ============================================================================
"""Tests for util_tfp."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from absl.testing import parameterized
from jax.config import config as jax_config
import numpy as np
import tensorflow.compat.v2 as real_tf

from discussion import fun_mcmc
from discussion.fun_mcmc import backend

tf = backend.tf
tfp = backend.tfp
util = backend.util
util_tfp = fun_mcmc.util_tfp

real_tf.enable_v2_behavior()
jax_config.update('jax_enable_x64', True)


class DupBijector(tfp.bijectors.Bijector):
  """Test, multi-part bijector."""

  def __init__(self):
    super(DupBijector, self).__init__(
        forward_min_event_ndims=0,
        validate_args=False,
        parameters={},
        name='dup')

  def forward(self, x, **kwargs):
    return [x, x]

  def inverse(self, y, **kwargs):
    return y[0]

  def forward_event_shape(self, x_shape, **kwargs):
    return [x_shape, x_shape]

  def inverse_event_shape(self, y_shape, **kwargs):
    return y_shape

  def forward_log_det_jacobian(self, x, event_ndims, **kwargs):
    return 0.

  def inverse_log_det_jacobian(self, y, event_ndims, **kwargs):
    return 0.

  def forward_dtype(self, x_dtype, **kwargs):
    return [x_dtype, x_dtype]

  def inverse_dtype(self, y_dtype, **kwargs):
    return y_dtype[0]


class UtilTFPTestTensorFlow32(real_tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(UtilTFPTestTensorFlow32, self).setUp()
    backend.set_backend(backend.TENSORFLOW, backend.MANUAL_TRANSFORMS)

  @property
  def _dtype(self):
    return tf.float32

  def _constant(self, value):
    return tf.constant(value, self._dtype)

  def testWrapTransitionKernel(self):

    class TestKernel(tfp.mcmc.TransitionKernel):

      def one_step(self, current_state, previous_kernel_results):
        return [x + 1 for x in current_state], previous_kernel_results + 1

      def bootstrap_results(self, current_state):
        return sum(current_state)

      def is_calibrated(self):
        return True

    def kernel(state, pkr):
      return util_tfp.transition_kernel_wrapper(state, pkr, TestKernel())

    state = {'x': self._constant(0.), 'y': self._constant(1.)}
    kr = 1.
    (final_state, final_kr), _ = fun_mcmc.trace(
        (state, kr),
        kernel,
        2,
        trace_fn=lambda *args: (),
    )
    self.assertAllEqual({
        'x': 2.,
        'y': 3.
    }, util.map_tree(np.array, final_state))
    self.assertAllEqual(1. + 2., final_kr)

  def testBijectorToTransformFn(self):
    bijectors = [
        tfp.bijectors.Identity(),
        tfp.bijectors.Scale(self._constant([
            [1., 2.],
            [3., 4.],
        ]))
    ]
    state = [
        tf.ones([2, 1], dtype=self._dtype),
        tf.ones([2, 2], dtype=self._dtype)
    ]
    transform_fn = util_tfp.bijector_to_transform_fn(
        bijectors, state_structure=state, batch_ndims=1)

    fwd, (_, fwd_ldj1), fwd_ldj2 = fun_mcmc.call_transport_map_with_ldj(
        transform_fn, state)
    self.assertAllClose(
        [np.ones([2, 1]), np.array([
            [1., 2.],
            [3., 4],
        ])], fwd)

    true_fwd_ldj = np.array([
        np.log(1) + np.log(2),
        np.log(3) + np.log(4),
    ])

    self.assertAllClose(true_fwd_ldj, fwd_ldj1)
    self.assertAllClose(true_fwd_ldj, fwd_ldj2)

    inverse_transform_fn = backend.util.inverse_fn(transform_fn)
    inv, (_, inv_ldj1), inv_ldj2 = fun_mcmc.call_transport_map_with_ldj(
        inverse_transform_fn, state)
    self.assertAllClose(
        [np.ones([2, 1]),
         np.array([
             [1., 1. / 2.],
             [1. / 3., 1. / 4.],
         ])], inv)
    self.assertAllClose(-true_fwd_ldj, inv_ldj1)
    self.assertAllClose(-true_fwd_ldj, inv_ldj2)

  def testBijectorToTransformFnMulti(self):
    bijector = DupBijector()
    state = tf.ones([1, 2], dtype=self._dtype)
    transform_fn = util_tfp.bijector_to_transform_fn(
        bijector, state_structure=state, batch_ndims=1)

    fwd, (_, fwd_ldj1), fwd_ldj2 = fun_mcmc.call_transport_map_with_ldj(
        transform_fn, state)
    self.assertAllClose([np.ones([1, 2]), np.ones([1, 2])], fwd)

    self.assertAllClose(0., fwd_ldj1)
    self.assertAllClose(0., fwd_ldj2)

    inverse_transform_fn = backend.util.inverse_fn(transform_fn)
    inv, (_, inv_ldj1), inv_ldj2 = fun_mcmc.call_transport_map_with_ldj(
        inverse_transform_fn, [
            tf.ones([1, 2], dtype=self._dtype),
            tf.ones([2, 1], dtype=self._dtype)
        ])
    self.assertAllClose(np.ones([1, 2]), inv)
    self.assertAllClose(0., inv_ldj1)
    self.assertAllClose(0., inv_ldj2)


class UtilTFPTestJAX32(UtilTFPTestTensorFlow32):

  def setUp(self):
    super(UtilTFPTestJAX32, self).setUp()
    backend.set_backend(backend.JAX, backend.MANUAL_TRANSFORMS)


class UtilTFPTestTensorFlow64(UtilTFPTestTensorFlow32):

  @property
  def _dtype(self):
    return tf.float64


class UtilTFPTestJAX64(UtilTFPTestJAX32):

  @property
  def _dtype(self):
    return tf.float64


if __name__ == '__main__':
  real_tf.test.main()
