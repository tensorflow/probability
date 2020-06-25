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
"""Tests for tensorflow_probability.spinoffs.oryx.experimental.nn.combinator."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import random
import numpy as np

from oryx.core import state
from oryx.experimental.nn import combinator
from oryx.experimental.nn import convolution
from oryx.experimental.nn import core
from oryx.experimental.nn import normalization
from oryx.experimental.nn import reshape


def define_dnn():
  return combinator.Serial([core.Dense(20),
                            core.Relu(),
                            core.Dropout(0.5),
                            core.Dense(10),
                            core.Tanh()])


def define_cnn():
  return combinator.Serial([convolution.Conv(20, (2, 2)),
                            normalization.BatchNorm(),
                            core.Relu(),
                            reshape.Flatten(),
                            core.Dense(10),
                            core.Softmax()])


class CombinatorTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._seed = random.PRNGKey(0)

  @parameterized.named_parameters(
      ('dnn', define_dnn, (5,)),
      ('cnn', define_cnn, (5, 5, 3)))
  def test_spec(self, define_network, in_shape):
    network_init = define_network()
    out_spec = network_init.spec(state.Shape(in_shape))
    self.assertEqual(out_spec.shape, (10,))

  @parameterized.named_parameters(
      ('dnn', define_dnn, (5,)),
      ('cnn', define_cnn, (5, 5, 3)))
  def test_call_shape(self, define_network, in_shape):
    net_rng, data_rng = random.split(self._seed)
    network_init = define_network()
    network = network_init.init(net_rng, state.Shape(in_shape))
    x = random.normal(data_rng, in_shape)
    y = network(x, rng=net_rng)
    self.assertTupleEqual((10,), y.shape)

  @parameterized.named_parameters(
      ('dnn', define_dnn, (5,)),
      ('cnn', define_cnn, (5, 5, 3)))
  def test_same_rng(self, define_network, in_shape):
    net_rng, data_rng = random.split(random.PRNGKey(0))
    network_init = define_network()
    network = network_init.init(net_rng, state.Shape(in_shape))
    x = random.normal(data_rng, in_shape)
    # Using the same rng key produces the same results.
    y = network(x, rng=net_rng)
    y1 = network(x, rng=net_rng)
    np.testing.assert_allclose(y, y1)

  @parameterized.named_parameters(
      ('dnn', define_dnn, (5,)),
      ('cnn', define_cnn, (5, 5, 3)))
  def test_training_false(self, define_network, in_shape):
    net_rng, data_rng = random.split(random.PRNGKey(0))
    network_init = define_network()
    network = network_init.init(net_rng, state.Shape(in_shape))
    x = random.normal(data_rng, in_shape)
    key1, key2 = random.split(net_rng)
    # Different rng key produces the same results when training is False.
    y = network(x, training=False, rng=key1)
    y1 = network(x, training=False, rng=key2)
    np.testing.assert_allclose(y, y1)

  @parameterized.named_parameters(
      ('dnn', define_dnn, (5,)),
      ('cnn', define_cnn, (5, 5, 3)))
  def test_update_state(self, define_network, in_shape):
    net_rng, data_rng = random.split(self._seed)
    network_init = define_network()
    network = network_init.init(net_rng, state.Shape(in_shape))
    x = random.normal(data_rng, in_shape)
    next_network = network.update(x, rng=net_rng)
    next_network_1 = network.update(x, rng=net_rng)
    for s, s1 in zip(next_network.state, next_network_1.state):
      if s is None:
        self.assertIsNone(s1)
      else:
        np.testing.assert_allclose(s, s1)
    y = next_network(x, rng=net_rng)
    y1 = next_network_1(x, rng=net_rng)
    np.testing.assert_allclose(y, y1)

  @parameterized.named_parameters(
      ('dnn', define_dnn, (5,)),
      ('cnn', define_cnn, (5, 5, 3)))
  def test_vmap_update(self, define_network, in_shape):
    net_rng, data_rng = random.split(self._seed)
    network_init = define_network()
    network = network_init.init(net_rng, state.Shape(in_shape))
    x = random.normal(data_rng, (10,) + in_shape)
    next_network = jax.vmap(lambda x: network.update(x, rng=net_rng),
                            out_axes=None)(x)
    next_network_1 = jax.vmap(lambda x: network.update(x, rng=net_rng),
                              out_axes=None)(x)
    for s, s1 in zip(next_network.state, next_network_1.state):
      if s is None:
        self.assertIsNone(s1)
      else:
        np.testing.assert_allclose(s, s1)
    y = jax.vmap(lambda x: next_network(x, rng=net_rng))(x)
    y1 = jax.vmap(lambda x: next_network_1(x, rng=net_rng))(x)
    np.testing.assert_allclose(y, y1)

  @parameterized.named_parameters(
      ('dnn', define_dnn, (5,)),
      ('cnn', define_cnn, (5, 5, 3)))
  def test_vmap_call_update(self, define_network, in_shape):
    net_rng, data_rng = random.split(self._seed)
    network_init = define_network()
    network = network_init.init(net_rng, state.Shape(in_shape))
    x = random.normal(data_rng, (10,) + in_shape)
    y, next_network = jax.vmap(
        lambda x: network.call_and_update(x, rng=net_rng),
        out_axes=(0, None))(x)
    y1, next_network_1 = jax.vmap(
        lambda x: network.call_and_update(x, rng=net_rng),
        out_axes=(0, None))(x)
    for s, s1 in zip(next_network.state, next_network_1.state):
      if s is None:
        self.assertIsNone(s1)
      else:
        np.testing.assert_allclose(s, s1)
    np.testing.assert_allclose(y, y1)

  @parameterized.named_parameters(
      ('dnn', define_dnn, (5,)),
      ('cnn', define_cnn, (5, 5, 3)))
  def test_flattening(self, define_network, in_shape):
    net_rng, data_rng, key = random.split(self._seed, 3)
    network_init = define_network()
    network = network_init.init(net_rng, state.Shape(in_shape))
    xs, data = network.flatten()
    network1 = network.unflatten(data, xs)
    x = random.normal(data_rng, in_shape)
    y = network(x, training=False, rng=key)
    y1 = network1(x, training=False, rng=key)
    np.testing.assert_allclose(y, y1)

  @parameterized.named_parameters(
      ('dnn', define_dnn, (5,)))
  def test_stateless_layer(self, define_network, in_shape):
    net_rng, data_rng, key = random.split(self._seed, 3)
    network_init = define_network()
    network = network_init.init(net_rng, state.Shape(in_shape))
    x = random.normal(data_rng, in_shape)
    y, new_network = network.call_and_update(x, rng=key)
    y1 = new_network(x, rng=key)
    np.testing.assert_allclose(y, y1)

  @parameterized.named_parameters(
      ('cnn', define_cnn, (5, 5, 3)))
  def test_stateful_layer(self, define_network, in_shape):
    net_rng, data_rng = random.split(self._seed, 2)
    network_init = define_network()
    network = network_init.init(net_rng, state.Shape(in_shape))
    x = random.normal(data_rng, in_shape)
    y, new_network = network.call_and_update(x)
    y1 = new_network(x)
    np.testing.assert_allclose(y, y1)
    for s, s1 in zip(network.state, new_network.state):
      if s:
        self.assertTrue(np.any(np.not_equal(s, s1)))
      else:
        self.assertTupleEqual(s, s1)

if __name__ == '__main__':
  absltest.main()
