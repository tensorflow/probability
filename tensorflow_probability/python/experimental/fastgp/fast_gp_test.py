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

from absl.testing import parameterized
import jax
from jax import config
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.python.experimental.fastgp import fast_gp
from tensorflow_probability.python.experimental.fastgp import preconditioners
from tensorflow_probability.substrates import jax as tfp

from absl.testing import absltest

tfd = tfp.distributions


class _FastGpTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(5)
    self.points = np.random.rand(100, 30).astype(self.dtype)

  def test_gaussian_process_copy(self):
    kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(
        self.dtype(0.5), self.dtype(2.0), feature_ndims=0
    )
    index_points = jax.random.uniform(
        jax.random.PRNGKey(0), shape=(10,), minval=-1.0, maxval=1.0,
        dtype=self.dtype)
    my_gp = fast_gp.GaussianProcess(
        kernel,
        index_points,
        observation_noise_variance=self.dtype(3e-3),
        jitter=self.dtype(1e-4)
    )
    my_gp_copy = my_gp.copy(
        config=fast_gp.GaussianProcessConfig(preconditioner_rank=50)
    )
    my_gp_params = my_gp.parameters.copy()
    my_gp_copy_params = my_gp_copy.parameters.copy()
    self.assertNotEqual(
        my_gp_params.pop("config"), my_gp_copy_params.pop("config")
    )
    self.assertEqual(my_gp.batch_shape, [])
    self.assertEqual(my_gp_params, my_gp_copy_params)

  def test_gaussian_process_mean(self):
    mean_fn = lambda x: x[:, 0]**2
    kernel = tfp.math.psd_kernels.ExponentiatedQuadratic()
    index_points = np.expand_dims(
        np.random.uniform(-1., 1., 10).astype(self.dtype), -1)
    gp = fast_gp.GaussianProcess(kernel, index_points, mean_fn=mean_fn)
    expected_mean = mean_fn(index_points)
    np.testing.assert_allclose(
        expected_mean, gp.mean(), rtol=1e-5)

  def test_gaussian_process_covariance_and_variance(self):
    amp = self.dtype(.5)
    len_scale = self.dtype(.2)
    jitter = self.dtype(1e-4)
    observation_noise_variance = self.dtype(3e-3)

    kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(amp, len_scale)

    index_points = np.expand_dims(
        np.random.uniform(-1., 1., 10).astype(self.dtype), -1)

    gp = fast_gp.GaussianProcess(
        kernel,
        index_points,
        observation_noise_variance=observation_noise_variance,
        jitter=jitter)

    def _kernel_fn(x, y):
      return amp ** 2 * np.exp(-.5 * (np.squeeze((x - y)**2)) / (len_scale**2))

    expected_covariance = (
        _kernel_fn(np.expand_dims(index_points, 0),
                   np.expand_dims(index_points, 1)) +
        observation_noise_variance * np.eye(10))

    np.testing.assert_allclose(
        expected_covariance, gp.covariance(), rtol=1e-5)
    np.testing.assert_allclose(
        np.diag(expected_covariance), gp.variance(), rtol=1e-5)

  def test_gaussian_process_log_prob(self):
    kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(
        self.dtype(0.5), self.dtype(2.0), feature_ndims=0
    )
    index_points = jax.random.uniform(
        jax.random.PRNGKey(0),
        shape=(10,),
        minval=-1.0,
        maxval=1.0,
        dtype=self.dtype
    )
    my_gp = fast_gp.GaussianProcess(
        kernel,
        index_points,
        observation_noise_variance=self.dtype(3e-3),
        jitter=self.dtype(1e-4)
    )
    slow_gp = tfd.GaussianProcess(
        kernel,
        index_points,
        observation_noise_variance=self.dtype(3e-3),
        jitter=self.dtype(1e-4)
    )
    samples = slow_gp.sample(5, seed=jax.random.PRNGKey(0))
    np.testing.assert_allclose(
        my_gp.log_prob(samples, key=jax.random.PRNGKey(1)),
        slow_gp.log_prob(samples),
        rtol=1e-5,
    )

  @parameterized.parameters(
      (fast_gp.GaussianProcessConfig(cg_iters=10), 1.0),
      (fast_gp.GaussianProcessConfig(preconditioner="identity"), 1.0),
      (fast_gp.GaussianProcessConfig(preconditioner_rank=10), 1.0),
      (fast_gp.GaussianProcessConfig(preconditioner_num_iters=10), 1.0),
      (fast_gp.GaussianProcessConfig(precondition_before_jitter="true"), 10.0),
      (fast_gp.GaussianProcessConfig(precondition_before_jitter="false"), 1.0),
      (fast_gp.GaussianProcessConfig(probe_vector_type="rademacher"), 1.0),
      (fast_gp.GaussianProcessConfig(num_probe_vectors=20), 1.0),
      (fast_gp.GaussianProcessConfig(log_det_algorithm="slq"), 1.0),
      (fast_gp.GaussianProcessConfig(log_det_iters=10), 1.0),
  )
  def test_gaussian_process_log_prob_with_configs(self, gp_config, delta):
    kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(
        self.dtype(0.5), self.dtype(2.0), feature_ndims=0
    )
    index_points = jax.random.uniform(
        jax.random.PRNGKey(0),
        shape=(3,),
        minval=-1.0,
        maxval=1.0,
        dtype=self.dtype,
    )
    my_gp = fast_gp.GaussianProcess(
        kernel,
        index_points,
        observation_noise_variance=self.dtype(3e-3),
        jitter=self.dtype(1e-4),
        config=gp_config,
    )
    samples = jnp.array([[self.dtype(1.0), self.dtype(2.0), self.dtype(3.0)]])
    lp = my_gp.log_prob(samples, key=jax.random.PRNGKey(1))
    target = -173.0 if self.dtype == np.float32 else -294.0
    self.assertAlmostEqual(lp, target, delta=delta)

  def test_gaussian_process_log_prob_plus_scaling(self):
    # Disabled because of b/323368033
    return  # EnableOnExport
    if self.dtype in [np.float32, np.float64]:
      self.skipTest("Numerically unstable.")
    kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(
        self.dtype(0.5), self.dtype(2.0), feature_ndims=0
    )
    index_points = jax.random.uniform(
        jax.random.PRNGKey(0),
        shape=(10,),
        minval=-1.0,
        maxval=1.0,
        dtype=self.dtype
    )
    my_gp = fast_gp.GaussianProcess(
        kernel,
        index_points,
        observation_noise_variance=self.dtype(3e-3),
        jitter=self.dtype(5e-4),
        config=fast_gp.GaussianProcessConfig(
            preconditioner="partial_cholesky_plus_scaling")
    )
    slow_gp = tfd.GaussianProcess(
        kernel,
        index_points,
        observation_noise_variance=self.dtype(3e-3),
        jitter=self.dtype(1e-4)
    )
    samples = slow_gp.sample(5, seed=jax.random.PRNGKey(2))
    np.testing.assert_allclose(
        my_gp.log_prob(samples, key=jax.random.PRNGKey(3)),
        slow_gp.log_prob(samples),
        rtol=1e-5,
    )

  def test_gaussian_process_log_prob_single_sample(self):
    kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(
        self.dtype(0.5), self.dtype(2.0), feature_ndims=0
    )
    index_points = jax.random.uniform(
        jax.random.PRNGKey(0),
        shape=(10,),
        minval=-1.0,
        maxval=1.0,
        dtype=self.dtype
    )
    my_gp = fast_gp.GaussianProcess(
        kernel,
        index_points,
        observation_noise_variance=self.dtype(3e-3),
        jitter=self.dtype(1e-4)
    )
    slow_gp = tfd.GaussianProcess(
        kernel,
        index_points,
        observation_noise_variance=self.dtype(3e-3),
        jitter=self.dtype(1e-4)
    )
    single_sample = slow_gp.sample(seed=jax.random.PRNGKey(0))
    lp = my_gp.log_prob(single_sample, key=jax.random.PRNGKey(1))
    self.assertEqual(single_sample.ndim, 1)
    self.assertEmpty(lp.shape)
    np.testing.assert_allclose(
        lp,
        slow_gp.log_prob(single_sample),
        rtol=1e-5,
    )

  def test_gaussian_process_log_prob2(self):
    index_points = jax.random.uniform(
        jax.random.PRNGKey(0), shape=(10,), minval=-1.0, maxval=1.0
    ).astype(self.dtype)
    samples = jnp.array([[
        -0.0980842,
        -0.27192444,
        -0.22313793,
        -0.07691351,
        -0.1314459,
        -0.2322599,
        -0.1493263,
        -0.11629149,
        -0.34304297,
        -0.24659207,
    ]]).astype(self.dtype)

    k = tfp.math.psd_kernels.ExponentiatedQuadratic(
        self.dtype(1.), self.dtype(1.), feature_ndims=0)
    fgp = fast_gp.GaussianProcess(
        k,
        index_points,
        observation_noise_variance=self.dtype(1e-3),
        jitter=self.dtype(1e-3),
        config=fast_gp.GaussianProcessConfig(preconditioner_rank=10),
    )
    sgp = tfd.GaussianProcess(
        k,
        index_points,
        observation_noise_variance=self.dtype(1e-3),
        jitter=self.dtype(1e-3)
    )

    fast_ll = jnp.sum(fgp.log_prob(samples, key=jax.random.PRNGKey(1)))
    slow_ll = jnp.sum(sgp.log_prob(samples))

    self.assertAlmostEqual(fast_ll, slow_ll, delta=5e-4)

  def test_gaussian_process_log_prob_jits(self):
    kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(
        self.dtype(0.5), self.dtype(2.0), feature_ndims=0
    )
    index_points = jax.random.uniform(
        jax.random.PRNGKey(0), shape=(10,), minval=-1.0, maxval=1.0
    ).astype(self.dtype)
    my_gp = fast_gp.GaussianProcess(
        kernel,
        index_points,
        observation_noise_variance=self.dtype(3e-3),
        jitter=self.dtype(1e-4)
    )
    slow_gp = tfd.GaussianProcess(
        kernel,
        index_points,
        observation_noise_variance=self.dtype(3e-3),
        jitter=self.dtype(1e-4)
    )
    samples = slow_gp.sample(5, seed=jax.random.PRNGKey(0))
    my_gp_log_prob = jax.jit(my_gp.log_prob)
    np.testing.assert_allclose(
        my_gp_log_prob(samples, key=jax.random.PRNGKey(1)),
        slow_gp.log_prob(samples),
        rtol=1e-5,
    )

  def test_gaussian_process_slq_log_prob_jits(self):
    kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(
        self.dtype(0.5), self.dtype(2.0), feature_ndims=0
    )
    index_points = jax.random.uniform(
        jax.random.PRNGKey(0),
        shape=(10,),
        minval=-1.0,
        maxval=1.0).astype(self.dtype)
    my_gp = fast_gp.GaussianProcess(
        kernel,
        index_points,
        observation_noise_variance=self.dtype(3e-3),
        jitter=self.dtype(1e-4),
        config=fast_gp.GaussianProcessConfig(log_det_algorithm="slq"),
    )
    slow_gp = tfd.GaussianProcess(
        kernel,
        index_points,
        observation_noise_variance=self.dtype(3e-3),
        jitter=self.dtype(1e-4))
    samples = slow_gp.sample(5, seed=jax.random.PRNGKey(0))
    my_gp_log_prob = jax.jit(my_gp.log_prob)
    np.testing.assert_allclose(
        my_gp_log_prob(samples, key=jax.random.PRNGKey(1)),
        slow_gp.log_prob(samples),
        rtol=1e-5,
    )

  def test_gaussian_process_log_prob_with_is_missing(self):
    kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(
        self.dtype(1.1), self.dtype(0.9))
    index_points = jnp.array(
        [[-1.0, 0.0], [-0.5, -0.5], [1.5, 0.0], [1.6, 1.5]],
        dtype=self.dtype)
    my_gp = fast_gp.GaussianProcess(
        kernel,
        index_points,
        observation_noise_variance=self.dtype(3e-3),
        jitter=self.dtype(1e-4))
    slow_gp = tfd.GaussianProcess(
        kernel,
        index_points,
        observation_noise_variance=self.dtype(3e-3),
        jitter=self.dtype(1e-4))
    x = slow_gp.sample(5, seed=jax.random.PRNGKey(0))
    is_missing = np.array([True, False, False, True])
    np.testing.assert_allclose(
        my_gp.log_prob(x, key=jax.random.PRNGKey(1), is_missing=is_missing),
        slow_gp.log_prob(x, is_missing=is_missing),
        rtol=1e-6,
    )

  def test_gp_log_prob_hard(self):
    kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(
        self.dtype(0.5), self.dtype(2.0), feature_ndims=0
    )
    index_points = jax.random.uniform(
        jax.random.PRNGKey(0), shape=(10,), minval=-1.0, maxval=1.0
    ).astype(self.dtype)
    slow_gp = tfd.GaussianProcess(
        kernel,
        index_points,
        observation_noise_variance=self.dtype(3e-3),
        jitter=self.dtype(1e-4))
    samples = slow_gp.sample(5, seed=jax.random.PRNGKey(0))

    k = tfp.math.psd_kernels.ExponentiatedQuadratic(
        amplitude=self.dtype(1.),
        length_scale=self.dtype(1.),
        feature_ndims=0
    )
    fgp = fast_gp.GaussianProcess(
        k,
        index_points,
        observation_noise_variance=self.dtype(1e-3),
        jitter=self.dtype(1e-3))
    sgp = tfd.GaussianProcess(
        k,
        index_points,
        observation_noise_variance=self.dtype(1e-3),
        jitter=self.dtype(1e-3))
    fgp_lp = jnp.sum(fgp.log_prob(samples, key=jax.random.PRNGKey(1)))
    sgp_lp = jnp.sum(sgp.log_prob(samples))

    self.assertAlmostEqual(fgp_lp, sgp_lp, places=3)

  def test_gp_log_prob_matern_five_halves(self):
    kernel = tfp.math.psd_kernels.MaternFiveHalves(
        self.dtype(2.0), self.dtype(1.0))
    index_points = jax.random.uniform(
        jax.random.PRNGKey(0), shape=(10, 5), minval=-1.0, maxval=1.0
    ).astype(self.dtype)
    sgp = tfd.GaussianProcess(
        kernel, index_points, observation_noise_variance=self.dtype(0.1)
    )
    sample = sgp.sample(1, seed=jax.random.PRNGKey(0))
    fgp = fast_gp.GaussianProcess(
        kernel, index_points, observation_noise_variance=self.dtype(0.1)
    )
    fgp_lp = jnp.sum(fgp.log_prob(sample, key=jax.random.PRNGKey(1)))
    sgp_lp = jnp.sum(sgp.log_prob(sample))

    self.assertAlmostEqual(fgp_lp, sgp_lp, places=3)

  def test_gaussian_process_log_prob_gradient(self):
    kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(
        self.dtype(0.5), self.dtype(2.0), feature_ndims=0
    )
    index_points = jax.random.uniform(
        jax.random.PRNGKey(0), shape=(10,), minval=-1.0, maxval=1.0,
        dtype=self.dtype)
    slow_gp = tfd.GaussianProcess(
        kernel,
        index_points,
        observation_noise_variance=self.dtype(3e-3),
        jitter=self.dtype(1e-4)
    )
    samples = slow_gp.sample(5, seed=jax.random.PRNGKey(0))

    def log_prob(amplitude, length_scale, noise, jitter):
      k = tfp.math.psd_kernels.ExponentiatedQuadratic(
          amplitude, length_scale, feature_ndims=0
      )
      gp = fast_gp.GaussianProcess(
          k, index_points, observation_noise_variance=noise, jitter=jitter
      )
      return jnp.sum(gp.log_prob(samples, key=jax.random.PRNGKey(1)))

    value, gradient = jax.value_and_grad(log_prob, argnums=[0, 1, 2, 3])(
        self.dtype(1.0), self.dtype(1.0), self.dtype(1e-3), self.dtype(1e-3)
    )
    d_amp, d_length_scale, d_noise, d_jitter = gradient
    self.assertFalse(jnp.isnan(value))
    self.assertFalse(jnp.isnan(d_amp))
    self.assertFalse(jnp.isnan(d_length_scale))
    self.assertFalse(jnp.isnan(d_noise))
    self.assertFalse(jnp.isnan(d_jitter))

    def slow_log_prob(amplitude, length_scale, noise, jitter):
      k = tfp.math.psd_kernels.ExponentiatedQuadratic(
          amplitude, length_scale, feature_ndims=0
      )
      gp = tfd.GaussianProcess(
          k, index_points, observation_noise_variance=noise, jitter=jitter
      )
      return jnp.sum(gp.log_prob(samples))

    direct_value = log_prob(
        self.dtype(1.0), self.dtype(1.0), self.dtype(1e-3), self.dtype(1e-3))
    direct_slow_value = slow_log_prob(
        self.dtype(1.0), self.dtype(1.0), self.dtype(1e-3), self.dtype(1e-3))
    self.assertAlmostEqual(direct_value, direct_slow_value, delta=4e-4)

    slow_value, slow_gradient = jax.value_and_grad(
        slow_log_prob, argnums=[0, 1, 2, 3]
    )(self.dtype(1.0), self.dtype(1.0), self.dtype(1e-3), self.dtype(1e-3))
    self.assertAlmostEqual(value, slow_value, delta=8)
    slow_d_amp, slow_d_length_scale, slow_d_noise, slow_d_jitter = slow_gradient
    self.assertAlmostEqual(d_amp, slow_d_amp, delta=0.01)
    self.assertAlmostEqual(d_length_scale, slow_d_length_scale, delta=0.002)
    # TODO(thomaswc): Investigate why the noise gradient is so noisy.
    self.assertAlmostEqual(d_noise, slow_d_noise, delta=0.5)
    # TODO(thomaswc): Investigate why slow_d_jitter is zero.
    self.assertAlmostEqual(d_jitter, slow_d_jitter, delta=1500)

  def test_gaussian_process_log_prob_gradient_of_index_points(self):
    samples = jnp.array([
        [-0.7, -0.1, -0.2],
        [-0.9, -0.4, -0.5],
        [-0.3, -0.2, -0.8],
        [-0.3, -0.1, -0.1],
        [-0.5, -0.3, -0.6],
    ], dtype=self.dtype)

    def fast_log_prob(pt1, pt2, pt3):
      index_points = jnp.array([[pt1], [pt2], [pt3]])
      k = tfp.math.psd_kernels.ExponentiatedQuadratic(
          self.dtype(1.1), self.dtype(0.9))
      gp = fast_gp.GaussianProcess(
          k,
          index_points,
          observation_noise_variance=self.dtype(3e-3),
          jitter=self.dtype(1e-4))
      lp = gp.log_prob(samples, key=jax.random.PRNGKey(1))
      return jnp.sum(lp)

    def slow_log_prob(pt1, pt2, pt3):
      index_points = jnp.array([[pt1], [pt2], [pt3]])
      k = tfp.math.psd_kernels.ExponentiatedQuadratic(
          self.dtype(1.1), self.dtype(0.9))
      gp = tfd.GaussianProcess(
          k,
          index_points,
          observation_noise_variance=self.dtype(3e-3),
          jitter=self.dtype(1e-4))
      lp = gp.log_prob(samples)
      return jnp.sum(lp)

    direct_slow_value = slow_log_prob(
        self.dtype(-0.5), self.dtype(0.), self.dtype(0.5))
    direct_fast_value = fast_log_prob(
        self.dtype(-0.5), self.dtype(0.), self.dtype(0.5))
    self.assertAlmostEqual(direct_slow_value, direct_fast_value, delta=1e-5)

    slow_value, slow_gradient = jax.value_and_grad(
        slow_log_prob, argnums=[0, 1, 2]
    )(self.dtype(-0.5), self.dtype(0.), self.dtype(0.5))
    fast_value, fast_gradient = jax.value_and_grad(
        fast_log_prob, argnums=[0, 1, 2]
    )(self.dtype(-0.5), self.dtype(0.), self.dtype(0.5))
    self.assertAlmostEqual(fast_value, slow_value, places=4)
    np.testing.assert_allclose(fast_gradient, slow_gradient, rtol=1e-4)

  def test_yt_inv_y(self):
    m = jnp.identity(100).astype(self.dtype)
    np.testing.assert_allclose(
        fast_gp.yt_inv_y(
            m,
            preconditioners.IdentityPreconditioner(
                m
            ).full_preconditioner(),
            self.points,
            max_iters=20,
        ),
        30.0,
        rtol=1e2,
    )

  def test_yt_inv_y_hard(self):
    m = jnp.array([
        [
            1.001,
            0.88311934,
            0.9894911,
            0.9695768,
            0.9987461,
            0.98577714,
            0.97863793,
            0.9880289,
            0.7110599,
            0.7718459,
        ],
        [
            0.88311934,
            1.001,
            0.9395206,
            0.7564426,
            0.86025584,
            0.94721663,
            0.7791884,
            0.8075757,
            0.9478641,
            0.9758552,
        ],
        [
            0.9894911,
            0.9395206,
            1.001,
            0.92534095,
            0.98108065,
            0.9997143,
            0.93953925,
            0.95583755,
            0.79332554,
            0.84795874,
        ],
        [
            0.9695768,
            0.7564426,
            0.92534095,
            1.001,
            0.98049456,
            0.91640615,
            0.9991695,
            0.99564964,
            0.5614807,
            0.6257758,
        ],
        [
            0.9987461,
            0.86025584,
            0.98108065,
            0.98049456,
            1.001,
            0.97622854,
            0.98763895,
            0.99449164,
            0.6813891,
            0.74358207,
        ],
        [
            0.98577714,
            0.94721663,
            0.9997143,
            0.91640615,
            0.97622854,
            1.001,
            0.9313745,
            0.9487237,
            0.80610526,
            0.859435,
        ],
        [
            0.97863793,
            0.7791884,
            0.93953925,
            0.9991695,
            0.98763895,
            0.9313745,
            1.001,
            0.99861676,
            0.5861309,
            0.65042824,
        ],
        [
            0.9880289,
            0.8075757,
            0.95583755,
            0.99564964,
            0.99449164,
            0.9487237,
            0.99861676,
            1.001,
            0.61803514,
            0.68201244,
        ],
        [
            0.7110599,
            0.9478641,
            0.79332554,
            0.5614807,
            0.6813891,
            0.80610526,
            0.5861309,
            0.61803514,
            1.001,
            0.9943819,
        ],
        [
            0.7718459,
            0.9758552,
            0.84795874,
            0.6257758,
            0.74358207,
            0.859435,
            0.65042824,
            0.68201244,
            0.9943819,
            1.001,
        ],
    ]).astype(self.dtype)
    ys = jnp.array([
        [
            -0.0980842,
            -0.27192444,
            -0.22313793,
            -0.07691352,
            -0.1314459,
            -0.2322599,
            -0.1493263,
            -0.11629149,
            -0.34304294,
            -0.24659212,
        ],
        [
            -0.12322001,
            -0.23061615,
            -0.13245171,
            -0.03604657,
            -0.18559735,
            -0.2970187,
            -0.11895001,
            -0.03382884,
            -0.28200114,
            -0.25570437,
        ],
        [
            -0.18551889,
            -0.13777351,
            -0.08382752,
            -0.17578323,
            -0.26691607,
            -0.06417686,
            -0.22161345,
            -0.18164475,
            -0.17793402,
            -0.22874065,
        ],
        [
            0.29383075,
            0.34788758,
            0.31571257,
            0.2702031,
            0.31359673,
            0.32859725,
            0.28001747,
            0.36051235,
            0.5047121,
            0.455843,
        ],
        [
            -0.47330144,
            -0.469457,
            -0.42139763,
            -0.3552108,
            -0.47754064,
            -0.47146142,
            -0.5066414,
            -0.4503611,
            -0.5367922,
            -0.5307923,
        ],
    ]).astype(self.dtype)
    preconditioner = preconditioners.PartialCholeskySplitPreconditioner(m)
    truth = jnp.einsum("ij,ij->j", ys.T, jnp.linalg.solve(m, ys.T))
    quadform = fast_gp.yt_inv_y(m, preconditioner.full_preconditioner(), ys.T)
    np.testing.assert_allclose(truth, quadform, rtol=2e-4)

    truth2 = tfp.math.hpsd_quadratic_form_solvevec(m, ys)
    np.testing.assert_allclose(truth2, quadform, rtol=2e-4)

  def test_yt_inv_y_derivative(self):
    def quadratic(scale):
      m = jnp.identity(5).astype(self.dtype) * scale
      return fast_gp.yt_inv_y(
          m,
          preconditioners.IdentityPreconditioner(m).full_preconditioner(),
          jnp.array(
              [1.0, 2.0, 3.0, 4.0, 5.0], dtype=self.dtype)[..., jnp.newaxis],
      )[0]

    d = jax.grad(quadratic)
    # quadratic(s) = 55/s, quadratic'(s) = -55 / s^2
    self.assertAlmostEqual(d(self.dtype(1.0)), -55.0)
    self.assertAlmostEqual(d(self.dtype(2.0)), -55.0/4.0)

  def test_yt_inv_y_derivative_with_diagonal_split_preconditioner(self):
    def quadratic(scale):
      m = jnp.identity(5).astype(self.dtype) * scale
      return fast_gp.yt_inv_y(
          m,
          preconditioners.DiagonalSplitPreconditioner(m).full_preconditioner(),
          jnp.array(
              [1.0, 2.0, 3.0, 4.0, 5.0], dtype=self.dtype)[..., jnp.newaxis],
      )[0]

    d = jax.grad(quadratic)
    # quadratic(s) = 55/s, quadratic'(s) = -55 / s^2
    self.assertAlmostEqual(d(self.dtype(1.0)), -55.0)
    self.assertAlmostEqual(d(self.dtype(2.0)), -55.0/4.0)

  def test_yt_inv_y_derivative_with_partial_cholesky_preconditioner(self):
    def quadratic(scale):
      m = jnp.identity(5).astype(self.dtype) * scale
      return fast_gp.yt_inv_y(
          m,
          preconditioners.PartialCholeskyPreconditioner(
              m).full_preconditioner(),
          jnp.array(
              [1.0, 2.0, 3.0, 4.0, 5.0], dtype=self.dtype)[..., jnp.newaxis],
      )[0]

    d = jax.grad(quadratic)
    # quadratic(s) = 55/s, quadratic'(s) = -55 / s^2
    self.assertAlmostEqual(d(self.dtype(1.0)), -55.0, delta=1e-5)
    self.assertAlmostEqual(d(self.dtype(2.0)), -55.0/4.0, delta=5e-4)

  def test_yt_inv_y_derivative_with_rank_one_preconditioner(self):
    def quadratic(scale):
      m = jnp.identity(5).astype(self.dtype) * scale
      return fast_gp.yt_inv_y(
          m,
          preconditioners.RankOnePreconditioner(
              m, key=jax.random.PRNGKey(5)).full_preconditioner(),
          jnp.array(
              [1.0, 2.0, 3.0, 4.0, 5.0], dtype=self.dtype)[..., jnp.newaxis],
      )[0]

    d = jax.grad(quadratic)
    # quadratic(s) = 55/s, quadratic'(s) = -55 / s^2
    self.assertAlmostEqual(d(self.dtype(1.0)), -55.0, delta=6e-2)
    self.assertAlmostEqual(d(self.dtype(2.0)), -55.0/4.0, delta=2e-2)

  def test_yt_inv_y_derivative_hard(self):
    y = jnp.array(
        [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]], dtype=self.dtype)
    b = (
        jnp.diag(jnp.full(10, 2.0))
        + jnp.diag(jnp.full(9, 1.0), 1)
        + jnp.diag(jnp.full(9, 1.0), -1)
    ).astype(self.dtype)

    def quad_form(jitter):
      m = b + jitter * jnp.identity(10).astype(self.dtype)
      pc = preconditioners.PartialCholeskySplitPreconditioner(m)
      return fast_gp.yt_inv_y(m, pc.full_preconditioner(), y.T)[0]

    d = jax.grad(quad_form)

    def quad_form2(jitter):
      m = b + jitter * jnp.identity(10).astype(self.dtype)
      return tfp.math.hpsd_quadratic_form_solvevec(m, y)[0]

    d2 = jax.grad(quad_form2)

    self.assertAlmostEqual(d(0.1), d2(0.1), delta=1e-4)
    self.assertAlmostEqual(d(1.0), d2(1.0), delta=1e-4)


class FastGpTestFloat32(_FastGpTest):
  dtype = np.float32


class FastGpTestFloat64(_FastGpTest):
  dtype = np.float64


del _FastGpTest


if __name__ == "__main__":
  config.update("jax_enable_x64", True)
  absltest.main()
