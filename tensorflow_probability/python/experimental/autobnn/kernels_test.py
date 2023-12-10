# Copyright 2023 The TensorFlow Probability Authors.
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
"""Tests for kernels.py."""

from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.python.experimental.autobnn import kernels
from tensorflow_probability.python.experimental.autobnn import util
from tensorflow_probability.substrates.jax.distributions import lognormal as lognormal_lib

from absl.testing import absltest


KERNELS = [
    kernels.IdentityBNN,
    kernels.OneLayerBNN,
    kernels.ExponentiatedQuadraticBNN,
    kernels.MaternBNN,
    kernels.PeriodicBNN,
    kernels.PolynomialBNN,
    kernels.LinearBNN,
    kernels.MultiLayerBNN,
]


class ReproduceExperimentTest(absltest.TestCase):

  def get_bnn_and_params(self):
    x_train, y_train = util.load_fake_dataset()
    linear_bnn = kernels.OneLayerBNN(width=50)
    seed = jax.random.PRNGKey(0)
    init_params = linear_bnn.init(seed, x_train)
    constant_params = jax.tree_map(
        lambda x: jnp.full(x.shape, 0.1), init_params)
    constant_params['params']['noise_scale'] = jnp.array([0.005 ** 0.5])
    return linear_bnn, constant_params, x_train, y_train

  # This now uses a Logistic noise model, not Normal as in Pearce
  @absltest.expectedFailure
  def test_log_prior_matches(self):
    # Pearce has a set `noise_scale` of 0.005 ** 0.5 that we must account for.
    linear_bnn, constant_params, _, _ = self.get_bnn_and_params()
    diff = lognormal_lib.LogNormal(
        linear_bnn.noise_min, linear_bnn.log_noise_scale
    ).log_prob(0.005**0.5)
    self.assertAlmostEqual(
        linear_bnn.log_prior(constant_params) - diff,
        31.59,  # Hardcoded from reference implementation.
        places=2)

  def test_log_likelihood_matches(self):
    linear_bnn, constant_params, x_train, y_train = self.get_bnn_and_params()
    self.assertAlmostEqual(
        linear_bnn.log_likelihood(constant_params, x_train, y_train),
        -7808.4434,
        places=2)

  # This now uses a Logistic noise model, not Normal as in Pearce
  @absltest.expectedFailure
  def test_log_prob_matches(self):
    # Pearce has a set `noise_scale` of 0.005 ** 0.5 that we must account for.
    linear_bnn, constant_params, x_train, y_train = self.get_bnn_and_params()
    diff = lognormal_lib.LogNormal(
        linear_bnn.noise_min, linear_bnn.log_noise_scale
    ).log_prob(0.005**0.5)
    self.assertAlmostEqual(
        linear_bnn.log_prob(constant_params, x_train, y_train) - diff,
        -14505.76,  # Hardcoded from reference implementation.
        places=2)


class KernelsTest(parameterized.TestCase):

  @parameterized.product(
      shape=[(5,), (5, 1), (5, 5)],
      kernel=KERNELS,
  )
  def test_default_kernels(self, shape, kernel):
    if kernel in [kernels.PeriodicBNN, kernels.MultiLayerBNN]:
      bnn = kernel(period=0.1, periodic_index=shape[-1]//2)
    else:
      bnn = kernel()
    if isinstance(bnn, kernels.PolynomialBNN):
      self.assertIn('shift', bnn.distributions())
    elif isinstance(bnn, kernels.MultiLayerBNN):
      self.assertIn('dense_1', bnn.distributions())
    elif isinstance(bnn, kernels.IdentityBNN):
      pass
    else:
      self.assertIn('dense1', bnn.distributions())
    if not isinstance(bnn, kernels.IdentityBNN):
      self.assertIn('dense2', bnn.distributions())
    params = bnn.init(jax.random.PRNGKey(0), jnp.zeros(shape))
    lprior = bnn.log_prior(params)
    params2 = params
    if 'params' in params2:
      params2 = params2['params']
    params2['noise_scale'] = params2['noise_scale'] + 100.0
    lprior2 = bnn.log_prior(params2)
    self.assertLess(lprior2, lprior)
    output = bnn.apply(params, jnp.ones(shape))
    self.assertEqual(shape[:-1] + (1,), output.shape)

  @parameterized.parameters(KERNELS)
  def test_likelihood(self, kernel):
    if kernel in [kernels.PeriodicBNN, kernels.MultiLayerBNN]:
      bnn = kernel(period=0.1)
    else:
      bnn = kernel()
    params = bnn.init(jax.random.PRNGKey(1), jnp.zeros(1))
    data = jnp.array([[0], [1], [2], [3], [4], [5]], dtype=jnp.float32)
    obs = jnp.array([1, 0, 1, 0, 1, 0], dtype=jnp.float32)
    ll = bnn.log_likelihood(params, data, obs)
    lp = bnn.log_prob(params, data, obs)
    # We are mostly just testing that ll and lp are both float-ish numbers
    # than can be compared.  In general, there is no reason to expect that
    # lp < ll because there is no reason to expect in general that the
    # log_prior will be negative.
    if kernel == kernels.MultiLayerBNN:
      self.assertLess(ll, lp)
    else:
      self.assertLess(lp, ll)

  @parameterized.parameters(
      (kernels.OneLayerBNN(width=10), 'OneLayer'),
      (kernels.ExponentiatedQuadraticBNN(width=5), 'RBF'),
      (kernels.MaternBNN(width=5), 'Matern(2.5)'),
      (kernels.PeriodicBNN(period=10, width=10), 'Periodic(period=10.00)'),
      (kernels.PolynomialBNN(degree=3, width=2), 'Polynomial(degree=3)'),
      (kernels.LinearBNN(width=5), 'Linear'),
      (kernels.QuadraticBNN(width=5), 'Quadratic'),
      (
          kernels.MultiLayerBNN(width=10, num_layers=3, period=20),
          'MultiLayer(num_layers=3,period=20)',
      ),
  )
  def test_summarize(self, bnn, expected):
    self.assertEqual(expected, bnn.summarize())

  @parameterized.parameters(KERNELS)
  def test_penultimate(self, kernel):
    if kernel in [kernels.PeriodicBNN, kernels.MultiLayerBNN]:
      bnn = kernel(period=0.1, going_to_be_multiplied=True)
    else:
      bnn = kernel(going_to_be_multiplied=True)
    self.assertNotIn('dense2', bnn.distributions())
    params = bnn.init(jax.random.PRNGKey(0), jnp.zeros(5))
    lprior = bnn.log_prior(params)
    if kernel != kernels.MultiLayerBNN:
      self.assertLess(lprior, 0.0)
    h = bnn.apply(params, jnp.ones(5), method=bnn.penultimate)
    self.assertEqual((50,), h.shape)

  def test_polynomial_is_almost_a_polynomial(self):
    poly_bnn = kernels.PolynomialBNN(degree=3)
    init_params = poly_bnn.init(jax.random.PRNGKey(0), jnp.ones((10, 1)))

    # compute power series
    func = lambda x: poly_bnn.apply(init_params, x)[0]
    params = [func(0.)]
    for _ in range(4):
      func = jax.grad(func)
      params.append(func(0.))

    # Last 4th degree coefficient should be around 0.
    self.assertAlmostEqual(params[-1], 0.)

    # Check that the random initialization is approximately a polynomial by
    # evaluating far away from the expansion.
    x = 17.0
    self.assertAlmostEqual(
        poly_bnn.apply(init_params, x)[0],
        params[0] + x * params[1] + x**2 * params[2] / 2 + x**3 * params[3] / 6,
        places=3)

  def test_make_periodic_input_warping_onedim(self):
    iw = kernels.make_periodic_input_warping(4, 0, True)
    np.testing.assert_allclose(
        jnp.array([0, 1, 1, 2, 3, 4, 5]),
        iw(jnp.array([1, 2, 3, 4, 5])),
        atol=1e-6
    )
    iw = kernels.make_periodic_input_warping(4, 0, False)
    np.testing.assert_allclose(
        jnp.array([0, 1, 2, 3, 4, 5]),
        iw(jnp.array([1, 2, 3, 4, 5])),
        atol=1e-6
    )

  def test_make_periodic_input_warping_onedim_features(self):
    iw = kernels.make_periodic_input_warping(4, 0, True)
    np.testing.assert_allclose(
        jnp.array([[1, 0, 0], [0, 1, 1], [-1, 0, 2], [0, -1, 3], [1, 0, 4]]),
        iw(jnp.array([[0], [1], [2], [3], [4]])),
        atol=1e-6
    )
    iw = kernels.make_periodic_input_warping(4, 0, False)
    np.testing.assert_allclose(
        jnp.array([[1, 0], [0, 1], [-1, 0], [0, -1], [1, 0]]),
        iw(jnp.array([[0], [1], [2], [3], [4]])),
        atol=1e-6
    )

  def test_make_periodic_input_warping_twodim(self):
    iw = kernels.make_periodic_input_warping(2, 0, True)
    np.testing.assert_allclose(
        jnp.array([[1, 0, 0, 0], [-1, 0, 1, 1], [1, 0, 2, 4], [-1, 0, 3, 9],
                   [1, 0, 4, 16]]),
        iw(jnp.array([[0, 0], [1, 1], [2, 4], [3, 9], [4, 16]])),
        atol=1e-6
    )
    iw = kernels.make_periodic_input_warping(4, 1, True)
    np.testing.assert_allclose(
        jnp.array([[0, 1, 0, 0], [1, 0, 1, 1], [2, 1, 0, 4], [3, 0, 1, 9],
                   [4, 1, 0, 16]]),
        iw(jnp.array([[0, 0], [1, 1], [2, 4], [3, 9], [4, 16]])),
        atol=1e-6
    )
    iw = kernels.make_periodic_input_warping(2, 0, False)
    np.testing.assert_allclose(
        jnp.array([[1, 0, 0], [-1, 0, 1], [1, 0, 4], [-1, 0, 9], [1, 0, 16]]),
        iw(jnp.array([[0, 0], [1, 1], [2, 4], [3, 9], [4, 16]])),
        atol=1e-6
    )


if __name__ == '__main__':
  absltest.main()
