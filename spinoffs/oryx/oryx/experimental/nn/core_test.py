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
"""Tests for tensorflow_probability.spinoffs.oryx.experimental.nn.core."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import random
from jax import test_util as jtu
from jax.example_libraries import stax

import numpy as np

from oryx.core import state
from oryx.experimental.nn import core
from oryx.internal import test_util


class DenseTest(test_util.TestCase):

  def setUp(self):
    super().setUp()
    self._seed = random.PRNGKey(0)

  def test_spec(self):
    net_init = core.Dense(100)

    out_shape = net_init.spec(state.Shape((10,))).shape
    self.assertEqual(out_shape, (100,))

    out_shape = net_init.spec(state.Shape((5, 10))).shape
    self.assertEqual(out_shape, (5, 100))

    out_shape = net_init.spec(state.Shape((-1, 5, 10))).shape
    self.assertEqual(out_shape, (-1, 5, 100))

  def test_call(self):
    net_rng, data_rng = random.split(self._seed)

    net_init = core.Dense(100)

    net = net_init.init(net_rng, state.Shape((10,)))
    w, b = net.params
    x = random.normal(data_rng, [10, 10])
    np.testing.assert_allclose(np.dot(x, w) + b, np.array(net(x)), atol=1e-05)

    net = net_init.init(net_rng, state.Shape((50,)))
    w, b = net.params
    x = random.normal(data_rng, [10, 50])
    np.testing.assert_allclose(np.dot(x, w) + b, np.array(net(x)), atol=1e-05)

  def test_kernel_init(self):
    net_rng, data_rng = random.split(self._seed)

    net_init = core.Dense(100, kernel_init=stax.ones)

    net = net_init.init(net_rng, state.Shape((10,)))
    w, b = net.params
    x = random.normal(data_rng, [10, 10])
    np.testing.assert_allclose(np.dot(x, w) + b, np.array(net(x)), atol=1e-05)

  def test_bias_init(self):
    net_rng, data_rng = random.split(self._seed)

    net_init = core.Dense(100, bias_init=stax.ones)

    net = net_init.init(net_rng, state.Shape((10,)))
    w, b = net.params
    x = random.normal(data_rng, [10, 10])
    np.testing.assert_allclose(np.dot(x, w) + b, np.array(net(x)), atol=1e-05)

  def test_check_grads(self):
    net_rng, data_rng = random.split(self._seed)

    net_init = core.Dense(100)

    net = net_init.init(net_rng, state.Shape((10,)))
    x = random.normal(data_rng, [10, 10])
    jtu.check_grads(net, (x,), 2, atol=0.03, rtol=0.03)


class ActivationsTest(test_util.TestCase):

  def setUp(self):
    super().setUp()
    self._seed = random.PRNGKey(0)

  @parameterized.named_parameters(
      ('Relu', core.Relu),
      ('Tanh', core.Tanh),
      ('Softplus', core.Softplus),
      ('LogSoftmax', core.LogSoftmax),
      ('Softmax', core.Softmax),
      )
  def test_spec(self, activation):
    net_rng = self._seed

    net_init = activation()

    for in_shape in ((10,), (5, 10), (1, 5, 10)):
      out_shape = net_init.spec(state.Shape(in_shape)).shape
      net = net_init.init(net_rng, state.Shape(in_shape))
      self.assertTupleEqual(out_shape, in_shape)
      x = random.normal(net_rng, in_shape)
      y = net(x)
      self.assertTupleEqual(x.shape, y.shape)


class DropoutTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._seed = random.PRNGKey(0)

  def test_init(self):
    net_init = core.Dropout(0.5)

    out_shape = net_init.spec(state.Shape((10,))).shape
    self.assertEqual(out_shape, (10,))

    out_shape = net_init.spec(state.Shape((5, 10))).shape
    self.assertEqual(out_shape, (5, 10))

  def test_call(self):
    net_rng, data_rng = random.split(self._seed)

    net_init = core.Dropout(1.0)

    net = net_init.init(net_rng, state.Shape((10,)))

    x = random.normal(data_rng, [10, 10])
    np.testing.assert_allclose(x, np.array(net(x, rng=net_rng)), atol=1e-05)

  def test_missing_rng_raise_error(self):
    net_rng, data_rng = random.split(self._seed)

    net_init = core.Dropout(1.0)

    net = net_init.init(net_rng, state.Shape((10,)))

    x = random.normal(data_rng, [10, 10])
    with self.assertRaisesRegex(ValueError,
                                'rng is required when training is True'):
      net(x)

  def test_fix_state_produces_same_results(self):
    net_rng, data_rng = random.split(self._seed)

    net_init = core.Dropout(0.5)

    net = net_init.init(net_rng, state.Shape((10,)))

    x = random.normal(data_rng, [10, 10])
    y = np.array(net(x, rng=net_rng))
    y2 = np.array(net(x, rng=net_rng))
    np.testing.assert_allclose(y, y2, atol=1e-05)

  def test_multiple_calls_produces_different_results(self):
    net_rng, data_rng = random.split(self._seed)

    net_init = core.Dropout(0.5)

    net = net_init.init(net_rng, state.Shape((10,)))

    x = random.normal(data_rng, [10, 10])
    y = np.array(net(x, rng=net_rng))
    exp_x = np.where(y == 0, x, y * 0.5)
    np.testing.assert_allclose(x, exp_x, atol=1e-05)

    # Calling with different rng produces different masks and results
    net_rng, _ = random.split(net_rng)
    y2 = np.array(net(x, rng=net_rng))
    self.assertGreater(np.sum(np.isclose(y, y2)), 10)
    self.assertLess(np.sum(np.isclose(y, y2)), 90)

  def test_training_is_false(self):
    net_rng, data_rng = random.split(self._seed)

    net_init = core.Dropout(0.5)

    net = net_init.init(net_rng, state.Shape((10,)))

    x = random.normal(data_rng, [10, 10])
    y = np.array(net(x, training=False, rng=net_rng))
    np.testing.assert_allclose(x, y)

    # Calling twice produces the same results with different rng.
    net_rng, _ = random.split(net_rng)
    y2 = np.array(net(x, training=False, rng=net_rng))
    np.testing.assert_allclose(x, y2)

  def test_jit(self):
    net_rng, data_rng = random.split(self._seed)

    net_init = core.Dropout(0.5)

    net = net_init.init(net_rng, state.Shape((10,)))

    j_net = jax.jit(lambda x, rng: net(x, rng=rng))
    x = random.normal(data_rng, [10, 10])
    y = np.array(net(x, rng=net_rng))
    j_y = np.array(j_net(x, net_rng))
    np.testing.assert_allclose(y, j_y)

  def test_check_grads(self):
    net_rng, data_rng = random.split(self._seed)

    net_init = core.Dropout(0.5)

    net = net_init.init(net_rng, state.Shape((10,)))

    x = random.normal(data_rng, [10, 10])
    fixed_net = lambda x: net(x, rng=net_rng)
    jtu.check_grads(fixed_net, (x,), 2)

  def test_jvp(self):
    net_rng, data_rng = random.split(self._seed)

    net_init = core.Dropout(0.5)

    net = net_init.init(net_rng, state.Shape((10,)))

    x = random.normal(data_rng, [10, 10])
    fixed_net = lambda x: net(x, rng=net_rng)
    y, y_tangent = jax.jvp(fixed_net, (x,), (jax.numpy.ones_like(x),))
    exp_tangent = np.where(np.array(y == 0), 0., 2.)
    np.testing.assert_allclose(exp_tangent, y_tangent)

  def test_vjp(self):
    net_rng, data_rng = random.split(self._seed)

    net_init = core.Dropout(0.5)

    net = net_init.init(net_rng, state.Shape((10,)))

    x = random.normal(data_rng, [10, 10])

    fixed_net = lambda x: net(x, rng=net_rng)
    y, f_vjp = jax.vjp(fixed_net, x)
    (y_tangent,) = f_vjp(jax.numpy.ones_like(x))
    exp_tangent = np.where(np.array(y == 0), 0., 2.)
    np.testing.assert_allclose(exp_tangent, y_tangent)


def reconstruct_loss(net, x, **kwargs):
  return jax.numpy.mean(jax.numpy.square(net(x, **kwargs) - x))


class GradTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._seed = random.PRNGKey(0)

  def test_dense(self):
    net_rng = self._seed
    network_init = core.Dense(2)
    network = network_init.init(net_rng, state.Shape((-1, 2)))

    grad_fn = jax.jit(jax.grad(reconstruct_loss))

    x0 = jax.numpy.array([[1.0, 1.0], [2.0, 1.0], [3.0, 0.5]])

    initial_loss = reconstruct_loss(network, x0)
    grads = grad_fn(network, x0)
    self.assertGreater(initial_loss, 0.0)
    network = network.replace(params=jax.tree_util.tree_multimap(
        lambda w, g: w - 0.1 * g, network.params, grads.params))
    final_loss = reconstruct_loss(network, x0)
    self.assertLess(final_loss, initial_loss)

  def test_dropout(self):
    net_rng = self._seed
    network_init = core.Dropout(0.5)
    network = network_init.init(net_rng, state.Shape((-1, 2)))

    grad_fn = jax.jit(jax.grad(reconstruct_loss))

    x0 = jax.numpy.array([[1.0, 1.0], [2.0, 1.0], [3.0, 0.5]])

    initial_loss = reconstruct_loss(network, x0, rng=net_rng)
    grads = grad_fn(network, x0, rng=net_rng)
    self.assertGreater(initial_loss, 0.0)
    network = network.replace(params=jax.tree_util.tree_multimap(
        lambda w, g: w - 0.1 * g, network.params, grads.params))
    final_loss = reconstruct_loss(network, x0, rng=net_rng)
    self.assertEqual(final_loss, initial_loss)

if __name__ == '__main__':
  absltest.main()
