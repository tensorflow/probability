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
"""Tests for fast_gprm.py."""

import jax
import numpy as np
from tensorflow_probability.python.experimental.fastgp import fast_gp
from tensorflow_probability.python.experimental.fastgp import fast_gprm
from tensorflow_probability.substrates import jax as tfp
from tensorflow_probability.substrates.jax.internal import test_util

tfd = tfp.distributions


class _FastGprmTest(test_util.TestCase):

  def testShapes(self):
    # 5x5 grid of index points in R^2 and flatten to 25x2
    index_points = np.linspace(-4., 4., 5)
    index_points = np.stack(np.meshgrid(index_points, index_points), axis=-1)
    index_points = np.reshape(index_points, [-1, 2]).astype(self.dtype)
    # ==> shape = [25, 2]

    seeds = jax.random.split(test_util.test_seed(sampler_type="stateless"), 5)

    kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(
        self.dtype(1.), self.dtype(1.))

    observation_noise_variance = self.dtype(1e-6)
    observation_index_points = jax.random.uniform(
        seeds[0], shape=(7, 2)).astype(self.dtype)
    observations = jax.random.uniform(seeds[1], shape=(7,)).astype(self.dtype)
    jitter = self.dtype(1e-6)

    dist = fast_gprm.GaussianProcessRegressionModel(
        kernel,
        seeds[2],
        index_points,
        observation_index_points,
        observations,
        observation_noise_variance,
        config=fast_gp.GaussianProcessConfig(
            preconditioner="partial_cholesky_plus_scaling",
            preconditioner_num_iters=25),
        jitter=jitter,
    )

    true_dist = tfd.GaussianProcessRegressionModel(
        kernel,
        index_points,
        observation_index_points,
        observations,
        observation_noise_variance,
        jitter=jitter)

    self.assertEqual(true_dist.event_shape, dist.event_shape)
    np.testing.assert_allclose(
        dist.mean(), true_dist.mean(), rtol=3e-1, atol=1e-4)
    np.testing.assert_allclose(
        dist.variance(), true_dist.variance(), rtol=3e-1, atol=1e-3)


class FastGprmTestFloat32(_FastGprmTest):
  dtype = np.float32


class FastGprmTestFloat64(_FastGprmTest):
  dtype = np.float64


del _FastGprmTest


if __name__ == "__main__":
  test_util.main()
