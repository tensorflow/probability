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
"""Tests for tensorflow_probability.spinoffs.oryx.experimental.nn.normalization."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import random
from jax import test_util as jtu
import numpy as np

from oryx.core import state
from oryx.experimental.nn import normalization
from oryx.internal import test_util


class NormalizationTest(test_util.TestCase):

  def setUp(self):
    super().setUp()
    self._seed = random.PRNGKey(0)

  @parameterized.named_parameters(
      ('hwc', (0, 1), (7,), (1, 1, 7)),
      ('chw', (1, 2), (5,), (5, 1, 1)))
  def test_spec(self, axis, param_shape, moving_shape):
    key = self._seed
    net_init = normalization.BatchNorm(axis)
    in_shape = (5, 6, 7)
    out_shape = net_init.spec(state.Shape(in_shape)).shape
    net = net_init.init(key, state.Shape(in_shape))
    self.assertEqual(out_shape, in_shape)

    beta, gamma = net.params
    self.assertEqual(param_shape, beta.shape)
    self.assertEqual(param_shape, gamma.shape)
    moving_mean, moving_var = net.state.moving_mean, net.state.moving_var
    self.assertEqual(moving_shape, moving_mean.shape)
    self.assertEqual(moving_shape, moving_var.shape)

  @parameterized.named_parameters(
      ('center_scale', True, True),
      ('no_center', False, True),
      ('no_scale', True, False),
      ('no_center_no_scale', False, False))
  def test_params(self, center, scale):
    key = self._seed
    net_init = normalization.BatchNorm(center=center, scale=scale)
    in_shape = (5, 6, 7)
    out_shape = net_init.spec(state.Shape(in_shape)).shape
    net = net_init.init(key, state.Shape(in_shape))
    self.assertEqual(out_shape, in_shape)

    beta, gamma = net.params
    if center:
      self.assertEqual(beta.shape, (7,))
      np.testing.assert_almost_equal(np.zeros_like(beta), beta)
    else:
      self.assertEqual(beta, ())
    if scale:
      self.assertEqual(gamma.shape, (7,))
      np.testing.assert_almost_equal(np.ones_like(gamma), gamma)
    else:
      self.assertEqual(gamma, ())

  def test_call_no_batch(self):
    epsilon = 1e-5
    axis = (0, 1)
    net_rng, data_rng = random.split(self._seed)

    net_init = normalization.BatchNorm(axis, epsilon=epsilon)
    in_shape = (5, 6, 7)
    net = net_init.init(net_rng, state.Shape(in_shape))
    x = random.normal(data_rng, in_shape)
    net_y = net(x)
    np.testing.assert_allclose(x, net_y)

    with self.assertRaises(ValueError):
      net_y = net(x[None])

  @parameterized.named_parameters(
      ('center_scale', True, True),
      ('no_center', False, True),
      ('no_scale', True, False),
      ('no_center_no_scale', False, False))
  def test_call(self, center, scale):
    epsilon = 1e-5
    axis = (0, 1)
    net_rng, data_rng = random.split(self._seed)

    net_init = normalization.BatchNorm(axis, center=center, scale=scale)
    in_shape = (5, 6, 7)
    net = net_init.init(net_rng, state.Shape(in_shape))

    beta, gamma = net.params
    x = random.normal(data_rng, (10,) + in_shape)
    batch_axis = (0,) + tuple(a + 1 for a in axis)
    mean = np.mean(np.array(x), batch_axis, keepdims=True)[0]
    var = np.var(np.array(x), batch_axis, keepdims=True)[0]
    z = (x - mean) / np.sqrt(var + epsilon)
    if center and scale:
      y = gamma * z + beta
    elif center:
      y = z + beta
    elif scale:
      y = gamma * z
    else:
      y = z
    net_y = jax.vmap(net)(x)
    np.testing.assert_almost_equal(y, np.array(net_y), decimal=6)

  def test_no_training(self):
    epsilon = 1e-5
    axis = (0, 1)
    net_rng, data_rng = random.split(self._seed)

    net_init = normalization.BatchNorm(axis, center=False, scale=False)
    in_shape = (5, 6, 7)
    net = net_init.init(net_rng, state.Shape(in_shape))

    x = random.normal(data_rng, (4,) + in_shape)
    z = x / np.sqrt(1.0 + epsilon)
    y = jax.vmap(lambda x: net(x, training=False))(x)
    np.testing.assert_almost_equal(z, np.array(y), decimal=6)

  def test_updates_moving_mean_var(self):
    axis = (0, 1)
    net_rng, data_rng = random.split(self._seed)

    net_init = normalization.BatchNorm(axis, momentum=0.9)
    in_shape = (5, 6, 7)
    net = net_init.init(net_rng, state.Shape(in_shape))
    self.assertAlmostEqual(0.1, net.info.decay)

    x = random.normal(data_rng, (4,) + in_shape)
    batch_axis = (0,) + tuple(a + 1 for a in axis)
    mean = np.mean(np.array(x), batch_axis, keepdims=True)[0]
    var = np.var(np.array(x), batch_axis, keepdims=True)[0]

    net_state = net.state
    # Initial values
    np.testing.assert_almost_equal(np.zeros_like(mean), net_state.moving_mean)
    np.testing.assert_almost_equal(np.ones_like(var), net_state.moving_var)

    # Update state (moving_mean, moving_var)
    for _ in range(100):
      net = jax.vmap(net.update, out_axes=None)(x)
    # Final values
    np.testing.assert_almost_equal(mean, net.state.moving_mean, decimal=4)
    np.testing.assert_almost_equal(var, net.state.moving_var, decimal=4)

  def test_check_grads(self):
    axis = (0, 1, 2)
    in_shape = (4, 5, 6, 7)
    net_rng, data_rng = random.split(self._seed)

    net_init = normalization.BatchNorm(axis)

    net = net_init.init(net_rng, state.Shape(in_shape))

    x = random.normal(data_rng, in_shape)
    jtu.check_grads(net.call, (x,), 2)


def mse(x, y):
  return jax.numpy.mean(jax.numpy.square(y - x))


def reconstruct_loss(net, x, **kwargs):
  preds, net = jax.vmap(
      lambda x: net.call_and_update(x, **kwargs),  # pylint: disable=unnecessary-lambda
      out_axes=(0, None))(x)
  return mse(x, preds), net


class GradTest(test_util.TestCase):

  def setUp(self):
    super().setUp()
    self._seed = random.PRNGKey(0)

  def test_batch_norm_moving_vars_grads(self):
    net_rng, data_rng = random.split(self._seed)
    axis = (0, 1)
    in_shape = (2, 2, 2)
    network_init = normalization.BatchNorm(axis)
    network = network_init.init(net_rng, state.Shape(in_shape))

    grad_fn = jax.grad(reconstruct_loss, has_aux=True)

    x0 = random.normal(data_rng, (2,) + in_shape)

    grads, _ = grad_fn(network, x0)
    grads_moving_mean, grads_moving_var = grads.state
    np.testing.assert_almost_equal(np.zeros_like(grads_moving_mean),
                                   grads_moving_mean)
    np.testing.assert_almost_equal(np.zeros_like(grads_moving_var),
                                   grads_moving_var)

  def test_batch_norm(self):
    net_rng, data_rng = random.split(self._seed)
    axis = (0, 1)
    in_shape = (2, 2, 2)
    network_init = normalization.BatchNorm(axis)
    initial_network = network_init.init(net_rng, state.Shape(in_shape))

    grad_fn = jax.grad(reconstruct_loss, has_aux=True)

    x0 = random.normal(data_rng, (2,) + in_shape)

    # reconstruct_loss updates network state
    initial_loss, network = reconstruct_loss(initial_network, x0)
    # grad also updates network state
    grads, new_network = grad_fn(network, x0)

    self.assertGreater(initial_loss, 0.0)
    # Make sure grad_fn updates the state.
    self.assertGreater(mse(initial_network.state.moving_mean,
                           new_network.state.moving_mean),
                       0.0)
    self.assertGreater(mse(initial_network.state.moving_var,
                           new_network.state.moving_var),
                       0.0)
    final_network = new_network.replace(
        params=jax.tree_util.tree_map(lambda w, g: w - 0.1 * g, network.params,
                                      grads.params))
    final_loss, final_network = reconstruct_loss(final_network, x0)
    self.assertLess(final_loss, initial_loss)
    self.assertGreater(mse(new_network.state.moving_mean,
                           final_network.state.moving_mean), 0.0)
    self.assertGreater(mse(new_network.state.moving_var,
                           final_network.state.moving_var), 0.0)


if __name__ == '__main__':
  absltest.main()
