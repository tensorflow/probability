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
"""Tests for tensorflow_probability.spinoffs.oryx.experimental.nn.reshape."""

from absl.testing import absltest
import jax
from jax import random

from oryx.core import state
from oryx.experimental.nn import reshape


class ReshapeTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._seed = random.PRNGKey(0)

  def test_reshape_spec(self):
    net_init = reshape.Reshape([10, 10])

    out_shape = net_init.spec(state.Shape((100,))).shape
    self.assertEqual(out_shape, (10, 10))

    out_shape = net_init.spec(state.Shape((10, 10))).shape
    self.assertEqual(out_shape, (10, 10))

    out_shape = net_init.spec(state.Shape((2, 5, 10))).shape
    self.assertEqual(out_shape, (10, 10))

  def test_flatten_shape(self):
    net_init = reshape.Flatten()

    out_shape = net_init.spec(state.Shape((5, 100))).shape
    self.assertEqual(out_shape, (500,))

    out_shape = net_init.spec(state.Shape((10, 10))).shape
    self.assertEqual(out_shape, (100,))

    out_shape = net_init.spec(state.Shape((1, 2, 5, 10))).shape
    self.assertEqual(out_shape, (100,))

  def test_reshape_call(self):
    net_rng, data_rng = random.split(self._seed)

    net_init = reshape.Reshape((10, 10))

    net = net_init.init(net_rng, state.Shape((100,)))
    x = random.normal(data_rng, (1, 100))
    self.assertEqual(jax.vmap(net)(x).shape, (1, 10, 10))

    net = net_init.init(net_rng, state.Shape((20, 5)))
    x = random.normal(data_rng, (5, 20, 5))
    self.assertEqual(jax.vmap(net)(x).shape, (5, 10, 10))

  def test_flatten_call(self):
    net_rng, data_rng = random.split(random.PRNGKey(0))

    net_init = reshape.Flatten()

    net = net_init.init(net_rng, state.Shape((10, 10)))
    x = random.normal(data_rng, (1, 10, 10))
    self.assertEqual(jax.vmap(net)(x).shape, (1, 100))

    x = random.normal(data_rng, (5, 20, 5))
    self.assertEqual(jax.vmap(net)(x).shape, (5, 100))


if __name__ == '__main__':
  absltest.main()
