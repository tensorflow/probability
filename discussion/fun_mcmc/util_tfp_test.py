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
import numpy as np
import tensorflow.compat.v2 as real_tf

from discussion import fun_mcmc
from discussion.fun_mcmc import backend

tf = backend.tf
tfp = backend.tfp
util = backend.util
util_tfp = fun_mcmc.util_tfp

real_tf.enable_v2_behavior()


class UtilTFPTestTensorFlow(real_tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(UtilTFPTestTensorFlow, self).setUp()
    backend.set_backend(backend.TENSORFLOW, backend.MANUAL_TRANSFORMS)

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

    state = {'x': 0., 'y': 1.}
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
        tfp.bijectors.Scale([
            [1., 2.],
            [3., 4.],
        ])
    ]
    state = [tf.ones([2, 1]), tf.ones([2, 2])]
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


class UtilTFPTestJAX(UtilTFPTestTensorFlow):

  def setUp(self):
    super(UtilTFPTestJAX, self).setUp()
    backend.set_backend(backend.JAX, backend.MANUAL_TRANSFORMS)


if __name__ == '__main__':
  real_tf.test.main()
