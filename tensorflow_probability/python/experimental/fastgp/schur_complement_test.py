# Copyright 2024 The TensorFlow Probability Authors.
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
"""Tests for schur_complement.py."""
from absl.testing import parameterized
import jax
from jax import config
import numpy as np
from tensorflow_probability.python.experimental.fastgp import preconditioners
from tensorflow_probability.python.experimental.fastgp import schur_complement
from tensorflow_probability.substrates import jax as tfp
from absl.testing import absltest


class _SchurComplementTest(parameterized.TestCase):

  @parameterized.parameters(
      {'dims': 3},
      {'dims': 4},
      {'dims': 5},
      {'dims': 7},
      {'dims': 11})
  def testValuesAreCorrect(self, dims):
    np.random.seed(42)
    num_obs = 5
    num_x = 3
    num_y = 7

    shape = [dims]

    base_kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(
        self.dtype(5.), self.dtype(1.), feature_ndims=1)

    fixed_inputs = np.random.uniform(
        -1., 1., size=[num_obs] + shape).astype(self.dtype)

    expected_k = tfp.math.psd_kernels.SchurComplement(
        base_kernel=base_kernel, fixed_inputs=fixed_inputs)

    # Can use a dummy matrix since this is ignored.
    pc = preconditioners.IdentityPreconditioner(
        np.ones([num_obs, num_obs]))

    actual_k = schur_complement.SchurComplement(
        base_kernel=base_kernel,
        preconditioner_fn=pc.full_preconditioner().solve,
        fixed_inputs=fixed_inputs)

    for i in range(5):
      x = jax.random.uniform(
          jax.random.PRNGKey(i),
          minval=-1,
          maxval=1,
          shape=[num_x] + shape).astype(self.dtype)
      y1 = jax.random.uniform(
          jax.random.PRNGKey(
              2 * i), minval=-1, maxval=1, shape=[num_x] + shape).astype(
                  self.dtype)
      y2 = jax.random.uniform(
          jax.random.PRNGKey(
              2 * i + 1), minval=-1, maxval=1, shape=[num_y] + shape).astype(
                  self.dtype)
      np.testing.assert_allclose(
          expected_k.apply(x, y1, example_ndims=1),
          actual_k.apply(x, y1, example_ndims=1),
          rtol=6e-4)
      np.testing.assert_allclose(
          expected_k.matrix(x, y2),
          actual_k.matrix(x, y2),
          rtol=3e-3)


class SchurComplementTestFloat32(_SchurComplementTest):
  dtype = np.float32


class SchurComplementTestFloat64(_SchurComplementTest):
  dtype = np.float64


del _SchurComplementTest


if __name__ == '__main__':
  config.update('jax_enable_x64', True)
  absltest.main()
