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
"""Tests for kernel_bias."""

from tensorflow_probability.python.experimental.nn.util import kernel_bias
from tensorflow_probability.python.internal import test_util


@test_util.test_all_tf_execution_regimes
class KernelBiasTest(test_util.TestCase):

  def test_make_kernel_bias(self):
    kernel_shape = (2, 3, 3, 2)
    bias_shape = (3,)
    kernel_batch_ndims = 2
    bias_batch_ndims = 2
    kernel, bias = kernel_bias.make_kernel_bias(
        kernel_shape,
        bias_shape=bias_shape,
        kernel_batch_ndims=kernel_batch_ndims,
        bias_batch_ndims=bias_batch_ndims)
    self.assertAllEqual(kernel.shape, kernel_shape)
    self.assertAllEqual(bias.shape, bias_shape)

  def test_make_kernel_bias_prior_spike_and_slab(self):
    kernel_shape = (2, 3, 3, 2)
    bias_shape = (3,)
    kernel_batch_ndims = 2
    bias_batch_ndims = 2
    prior = kernel_bias.make_kernel_bias_prior_spike_and_slab(
        kernel_shape,
        bias_shape=bias_shape,
        kernel_batch_ndims=kernel_batch_ndims,
        bias_batch_ndims=bias_batch_ndims)
    kernel, bias = prior.sample()
    self.assertAllEqual(kernel.shape, kernel_shape)
    self.assertAllEqual(bias.shape, bias_shape)

  def test_make_kernel_bias_posterior_mvn_diag(self):
    kernel_shape = (2, 3, 3, 2)
    bias_shape = (3,)
    kernel_batch_ndims = 2
    bias_batch_ndims = 2
    posterior = kernel_bias.make_kernel_bias_posterior_mvn_diag(
        kernel_shape,
        bias_shape=bias_shape,
        kernel_batch_ndims=kernel_batch_ndims,
        bias_batch_ndims=bias_batch_ndims)
    kernel, bias = posterior.sample()
    self.assertAllEqual(kernel.shape, kernel_shape)
    self.assertAllEqual(bias.shape, bias_shape)

if __name__ == '__main__':
  test_util.main()
