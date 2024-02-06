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
"""Tests for fast_gp.py."""

import jax
from jax import config
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.python.experimental.fastgp import fast_gp
from tensorflow_probability.python.experimental.fastgp import fast_mtgp
from tensorflow_probability.substrates import jax as tfp
from absl.testing import absltest

jtf = tfp.tf2jax
tfd = tfp.distributions
tfed = tfp.experimental.distributions


class _FastMultiTaskGpTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.points = np.random.rand(100, 30).astype(self.dtype)

  def test_gaussian_process_copy(self):
    kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(
        self.dtype(0.5), self.dtype(2.0)
    )
    kernel = tfp.experimental.psd_kernels.Independent(3, kernel)
    index_points = jax.random.uniform(
        jax.random.PRNGKey(0), shape=(10, 2), minval=-1.0, maxval=1.0,
        dtype=self.dtype)
    my_gp = fast_mtgp.MultiTaskGaussianProcess(
        kernel, index_points, observation_noise_variance=self.dtype(3e-3)
    )
    my_gp_copy = my_gp.copy(config=fast_gp.GaussianProcessConfig(
        preconditioner_num_iters=20))
    my_gp_params = my_gp.parameters.copy()
    my_gp_copy_params = my_gp_copy.parameters.copy()
    self.assertNotEqual(my_gp_params.pop("config"),
                        my_gp_copy_params.pop("config"))
    self.assertEqual(my_gp_params, my_gp_copy_params)

  def test_gaussian_process_log_prob(self):
    kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(
        self.dtype(0.5), self.dtype(2.0)
    )
    kernel = tfp.experimental.psd_kernels.Independent(3, kernel)
    index_points = jax.random.uniform(
        jax.random.PRNGKey(0), shape=(10, 2), minval=-1.0, maxval=1.0,
        dtype=self.dtype)
    my_gp = fast_mtgp.MultiTaskGaussianProcess(
        kernel,
        index_points,
        observation_noise_variance=self.dtype(3e-3),
        config=fast_gp.GaussianProcessConfig(
            preconditioner="partial_cholesky_plus_scaling")
    )
    slow_gp = tfed.MultiTaskGaussianProcess(
        kernel,
        index_points,
        observation_noise_variance=self.dtype(3e-3),
        cholesky_fn=jnp.linalg.cholesky
    )
    samples = slow_gp.sample(5, seed=jax.random.PRNGKey(0))
    np.testing.assert_allclose(
        my_gp.log_prob(samples, key=jax.random.PRNGKey(1)),
        slow_gp.log_prob(samples),
        rtol=2e-3,
    )

  def test_gaussian_process_log_prob_separable(self):
    kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(
        self.dtype(0.5), self.dtype(2.0)
    )
    task_cholesky = np.array([
        [1.4, 0., 0.],
        [0.5, 1.23, 0.],
        [0.25, 0.3, 1.34]], dtype=self.dtype)
    task_cholesky_linop = jtf.linalg.LinearOperatorLowerTriangular(
        task_cholesky)
    kernel = tfp.experimental.psd_kernels.Separable(
        3, task_kernel_scale_linop=task_cholesky_linop, base_kernel=kernel)
    index_points = jax.random.uniform(
        jax.random.PRNGKey(0), shape=(10, 2), minval=-1.0, maxval=1.0,
        dtype=self.dtype
    )
    my_gp = fast_mtgp.MultiTaskGaussianProcess(
        kernel,
        index_points,
        observation_noise_variance=self.dtype(3e-3),
        config=fast_gp.GaussianProcessConfig(
            preconditioner="partial_cholesky_plus_scaling")
    )
    slow_gp = tfed.MultiTaskGaussianProcess(
        kernel,
        index_points,
        observation_noise_variance=self.dtype(3e-3),
        cholesky_fn=jnp.linalg.cholesky,
    )
    samples = slow_gp.sample(5, seed=jax.random.PRNGKey(0))
    np.testing.assert_allclose(
        my_gp.log_prob(samples, key=jax.random.PRNGKey(1)),
        slow_gp.log_prob(samples),
        rtol=8e-3,
    )

  def test_gaussian_process_log_prob_single_sample(self):
    kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(
        self.dtype(0.5), self.dtype(2.0)
    )
    kernel = tfp.experimental.psd_kernels.Independent(3, kernel)
    index_points = jax.random.uniform(
        jax.random.PRNGKey(0), shape=(10, 2), minval=-1.0, maxval=1.0,
        dtype=self.dtype
    )
    my_gp = fast_mtgp.MultiTaskGaussianProcess(
        kernel,
        index_points,
        observation_noise_variance=self.dtype(3e-3),
        config=fast_gp.GaussianProcessConfig(
            preconditioner="partial_cholesky_plus_scaling")
    )
    slow_gp = tfed.MultiTaskGaussianProcess(
        kernel,
        index_points,
        observation_noise_variance=self.dtype(3e-3),
        cholesky_fn=jnp.linalg.cholesky
    )
    single_sample = slow_gp.sample(seed=jax.random.PRNGKey(0))
    lp = my_gp.log_prob(single_sample, key=jax.random.PRNGKey(1))
    self.assertEqual(single_sample.ndim, 2)
    self.assertEmpty(lp.shape)
    np.testing.assert_allclose(
        lp,
        slow_gp.log_prob(single_sample),
        rtol=7e-4,
    )

  def test_gaussian_process_log_prob2(self):
    index_points = jax.random.uniform(
        jax.random.PRNGKey(0), shape=(10, 2), minval=-1.0, maxval=1.0,
        dtype=self.dtype)
    samples = jnp.array([[
        [-0.0980842, -0.0980842, -0.0980842],
        [-0.27192444, -0.27192444, -0.27192444],
        [-0.22313793, -0.22313793, -0.22313793],
        [-0.07691351, -0.07691351, -0.07691351],
        [-0.1314459, -0.1314459, -0.1314459],
        [-0.2322599, -0.2322599, -0.2322599],
        [-0.1493263, -0.1493263, -0.1493263],
        [-0.11629149, -0.11629149, -0.11629149],
        [-0.34304297, -0.34304297, -0.34304297],
        [-0.24659207, -0.24659207, -0.24659207]
    ]]).astype(self.dtype)

    k = tfp.math.psd_kernels.ExponentiatedQuadratic(
        self.dtype(1.0), self.dtype(1.0))
    k = tfp.experimental.psd_kernels.Independent(3, k)
    fgp = fast_mtgp.MultiTaskGaussianProcess(
        k,
        index_points,
        observation_noise_variance=self.dtype(1e-3),
        config=fast_gp.GaussianProcessConfig(
            preconditioner_rank=30,
            preconditioner="partial_cholesky_plus_scaling"),
    )
    sgp = tfed.MultiTaskGaussianProcess(
        k,
        index_points,
        observation_noise_variance=self.dtype(1e-3),
        cholesky_fn=jnp.linalg.cholesky
    )

    fast_ll = jnp.sum(fgp.log_prob(samples, key=jax.random.PRNGKey(1)))
    slow_ll = jnp.sum(sgp.log_prob(samples))
    np.testing.assert_allclose(fast_ll, slow_ll, rtol=2e-4)

  def test_gaussian_process_log_prob_jits(self):
    kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(
        self.dtype(0.5), self.dtype(2.0))
    kernel = tfp.experimental.psd_kernels.Independent(3, kernel)
    index_points = jax.random.uniform(
        jax.random.PRNGKey(0), shape=(10, 2), minval=-1.0, maxval=1.0,
        dtype=self.dtype
    )
    my_gp = fast_mtgp.MultiTaskGaussianProcess(
        kernel,
        index_points,
        observation_noise_variance=self.dtype(3e-3),
        config=fast_gp.GaussianProcessConfig(
            preconditioner="partial_cholesky_plus_scaling"),
    )
    slow_gp = tfed.MultiTaskGaussianProcess(
        kernel,
        index_points,
        observation_noise_variance=self.dtype(3e-3),
        cholesky_fn=jnp.linalg.cholesky
    )
    samples = slow_gp.sample(5, seed=jax.random.PRNGKey(0))
    my_gp_log_prob = jax.jit(my_gp.log_prob)
    np.testing.assert_allclose(
        my_gp_log_prob(samples, key=jax.random.PRNGKey(1)),
        slow_gp.log_prob(samples),
        rtol=2e-3,
    )

  def test_gp_log_prob_hard(self):
    kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(
        self.dtype(0.5), self.dtype(1.0))
    kernel = tfp.experimental.psd_kernels.Independent(3, kernel)
    index_points = jax.random.uniform(
        jax.random.PRNGKey(0), shape=(10, 2), minval=-1.0, maxval=1.0,
        dtype=self.dtype
    )
    slow_gp = tfed.MultiTaskGaussianProcess(
        kernel,
        index_points,
        observation_noise_variance=self.dtype(3e-3),
        cholesky_fn=jnp.linalg.cholesky,
    )
    samples = slow_gp.sample(5, seed=jax.random.PRNGKey(0))

    k = tfp.math.psd_kernels.ExponentiatedQuadratic(
        amplitude=self.dtype(1.0), length_scale=self.dtype(1.0))
    task_cholesky = np.array([
        [1.4, 0., 0.],
        [0.5, 1.23, 0.],
        [0.25, 0.3, 1.34]], dtype=self.dtype)
    task_cholesky_linop = jtf.linalg.LinearOperatorLowerTriangular(
        task_cholesky)
    k = tfp.experimental.psd_kernels.Separable(
        3, task_kernel_scale_linop=task_cholesky_linop, base_kernel=k)

    fgp = fast_mtgp.MultiTaskGaussianProcess(
        k,
        index_points,
        observation_noise_variance=self.dtype(1e-3),
        config=fast_gp.GaussianProcessConfig(
            preconditioner_rank=30,
            preconditioner="partial_cholesky_plus_scaling"),
    )
    sgp = tfed.MultiTaskGaussianProcess(
        k,
        index_points,
        observation_noise_variance=self.dtype(1e-3),
        cholesky_fn=jnp.linalg.cholesky,
    )
    fgp_lp = jnp.sum(fgp.log_prob(samples, key=jax.random.PRNGKey(1)))
    sgp_lp = jnp.sum(sgp.log_prob(samples))
    np.testing.assert_allclose(fgp_lp, sgp_lp, rtol=3e-4)

  def test_gp_log_prob_matern_five_halves(self):
    kernel = tfp.math.psd_kernels.MaternFiveHalves(
        self.dtype(2.0), self.dtype(1.0))
    kernel = tfp.experimental.psd_kernels.Independent(3, kernel)
    index_points = jax.random.uniform(
        jax.random.PRNGKey(0), shape=(10, 5), minval=-1.0, maxval=1.0,
        dtype=self.dtype
    )
    sgp = tfed.MultiTaskGaussianProcess(
        kernel,
        index_points,
        observation_noise_variance=self.dtype(0.1),
        cholesky_fn=jnp.linalg.cholesky,
    )
    sample = sgp.sample(1, seed=jax.random.PRNGKey(0))
    fgp = fast_mtgp.MultiTaskGaussianProcess(
        kernel,
        index_points,
        observation_noise_variance=self.dtype(0.1),
        config=fast_gp.GaussianProcessConfig(
            preconditioner_rank=30,
            preconditioner="partial_cholesky_plus_scaling"),
    )
    fgp_lp = jnp.sum(fgp.log_prob(sample, key=jax.random.PRNGKey(1)))
    sgp_lp = jnp.sum(sgp.log_prob(sample))
    np.testing.assert_allclose(fgp_lp, sgp_lp, rtol=1e-5)

  def test_gaussian_process_log_prob_gradient(self):
    kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(
        self.dtype(0.5), self.dtype(2.0))
    kernel = tfp.experimental.psd_kernels.Independent(3, kernel)
    index_points = jax.random.uniform(
        jax.random.PRNGKey(0), shape=(10, 2), minval=-1.0, maxval=1.0,
        dtype=self.dtype
    )
    slow_gp = tfed.MultiTaskGaussianProcess(
        kernel,
        index_points,
        observation_noise_variance=self.dtype(3e-3),
        cholesky_fn=jnp.linalg.cholesky
    )
    samples = slow_gp.sample(5, seed=jax.random.PRNGKey(0))

    def log_prob(amplitude, length_scale, noise):
      k = tfp.math.psd_kernels.ExponentiatedQuadratic(
          amplitude, length_scale
      )
      k = tfp.experimental.psd_kernels.Independent(3, k)
      gp = fast_mtgp.MultiTaskGaussianProcess(
          k,
          index_points,
          observation_noise_variance=noise,
          config=fast_gp.GaussianProcessConfig(
              preconditioner_rank=30,
              preconditioner="partial_cholesky_plus_scaling"),
      )
      return jnp.sum(gp.log_prob(samples, key=jax.random.PRNGKey(1)))

    value, gradient = jax.value_and_grad(log_prob, argnums=[0, 1, 2])(
        self.dtype(1.0), self.dtype(1.0), self.dtype(1e-3))
    d_amp, d_length_scale, d_noise = gradient
    self.assertFalse(jnp.isnan(value))
    self.assertFalse(jnp.isnan(d_amp))
    self.assertFalse(jnp.isnan(d_length_scale))
    self.assertFalse(jnp.isnan(d_noise))

    def slow_log_prob(amplitude, length_scale, noise):
      k = tfp.math.psd_kernels.ExponentiatedQuadratic(
          amplitude, length_scale
      )
      k = tfp.experimental.psd_kernels.Independent(3, k)
      gp = tfed.MultiTaskGaussianProcess(
          k,
          index_points,
          observation_noise_variance=noise,
          cholesky_fn=jnp.linalg.cholesky,
      )
      return jnp.sum(gp.log_prob(samples))

    direct_value = log_prob(
        self.dtype(1.0), self.dtype(1.0), self.dtype(1e-3))
    direct_slow_value = slow_log_prob(
        self.dtype(1.0), self.dtype(1.0), self.dtype(1e-3))
    np.testing.assert_allclose(direct_value, direct_slow_value, rtol=4e-4)

    slow_value, slow_gradient = jax.value_and_grad(
        slow_log_prob, argnums=[0, 1, 2]
    )(self.dtype(1.0), self.dtype(1.0), self.dtype(1e-3))
    np.testing.assert_allclose(value, slow_value, rtol=4e-4)
    slow_d_amp, slow_d_length_scale, slow_d_noise = slow_gradient
    np.testing.assert_allclose(d_amp, slow_d_amp, rtol=1e-4)
    np.testing.assert_allclose(d_length_scale, slow_d_length_scale, rtol=1e-4)
    # TODO(thomaswc): Investigate why the noise gradient is so noisy.
    np.testing.assert_allclose(d_noise, slow_d_noise, rtol=1e-4)

  def test_gaussian_process_log_prob_gradient_of_index_points(self):
    samples = jnp.array([
        [-0.7, -0.1, -0.2],
        [-0.5, -0.3, -0.2],
        [-0.3, -0.1, -0.1],
    ], dtype=self.dtype)

    def fast_log_prob(pt1, pt2, pt3):
      index_points = jnp.array([[pt1], [pt2], [pt3]])
      k = tfp.math.psd_kernels.ExponentiatedQuadratic(
          self.dtype(1.1), self.dtype(0.9))
      k = tfp.experimental.psd_kernels.Independent(3, k)
      gp = fast_mtgp.MultiTaskGaussianProcess(
          k,
          index_points,
          observation_noise_variance=self.dtype(3e-3),
          config=fast_gp.GaussianProcessConfig(
              preconditioner="partial_cholesky_plus_scaling"),
      )
      lp = gp.log_prob(samples, key=jax.random.PRNGKey(1))
      return jnp.sum(lp)

    def slow_log_prob(pt1, pt2, pt3):
      index_points = jnp.array([[pt1], [pt2], [pt3]])
      k = tfp.math.psd_kernels.ExponentiatedQuadratic(
          self.dtype(1.1), self.dtype(0.9))
      k = tfp.experimental.psd_kernels.Independent(3, k)
      gp = tfed.MultiTaskGaussianProcess(
          k,
          index_points,
          observation_noise_variance=self.dtype(3e-3),
          cholesky_fn=jnp.linalg.cholesky,
      )
      lp = gp.log_prob(samples)
      return jnp.sum(lp)

    direct_slow_value = slow_log_prob(
        self.dtype(-0.5), self.dtype(0.0), self.dtype(0.5))
    direct_fast_value = fast_log_prob(
        self.dtype(-0.5), self.dtype(0.0), self.dtype(0.5))
    np.testing.assert_allclose(direct_slow_value, direct_fast_value, rtol=3e-5)

    slow_value, slow_gradient = jax.value_and_grad(
        slow_log_prob, argnums=[0, 1, 2]
    )(self.dtype(-0.5), self.dtype(0.0), self.dtype(0.5))

    fast_value, fast_gradient = jax.value_and_grad(
        fast_log_prob, argnums=[0, 1, 2]
    )(self.dtype(-0.5), self.dtype(0.0), self.dtype(0.5))
    np.testing.assert_allclose(fast_value, slow_value, rtol=3e-5)
    np.testing.assert_allclose(fast_gradient, slow_gradient, rtol=1e-4)

  def test_gaussian_process_mean(self):
    mean_fn = lambda x: jnp.stack([x[:, 0]**2, x[:, 0]**3], axis=-1)
    kernel = tfp.math.psd_kernels.ExponentiatedQuadratic()
    kernel = tfp.experimental.psd_kernels.Independent(2, kernel)
    index_points = np.expand_dims(
        np.random.uniform(-1., 1., 10).astype(self.dtype), -1)
    gp = fast_mtgp.MultiTaskGaussianProcess(
        kernel, index_points, mean_fn=mean_fn)
    expected_mean = mean_fn(index_points)
    np.testing.assert_allclose(
        expected_mean, gp.mean(), rtol=1e-5)

  def test_gaussian_process_variance(self):
    amp = self.dtype(.5)
    len_scale = self.dtype(.2)
    observation_noise_variance = self.dtype(3e-3)

    kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(amp, len_scale)
    kernel = tfp.experimental.psd_kernels.Independent(2, kernel)

    index_points = np.expand_dims(
        np.random.uniform(-1., 1., 10).astype(self.dtype), -1)

    fast = fast_mtgp.MultiTaskGaussianProcess(
        kernel,
        index_points,
        observation_noise_variance=observation_noise_variance)
    mtgp = tfed.MultiTaskGaussianProcess(
        kernel,
        index_points,
        observation_noise_variance=observation_noise_variance,
        cholesky_fn=jnp.linalg.cholesky)
    np.testing.assert_allclose(
        mtgp.variance(), fast.variance(), rtol=1e-5)


class FastMultiTaskGpTestFloat32(_FastMultiTaskGpTest):
  dtype = np.float32


class FastMultiTaskGpTestFloat64(_FastMultiTaskGpTest):
  dtype = np.float64


del _FastMultiTaskGpTest


if __name__ == "__main__":
  config.update("jax_enable_x64", True)
  absltest.main()
