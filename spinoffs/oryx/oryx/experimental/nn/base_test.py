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
"""Tests for tensorflow_probability.spinoffs.oryx.experimental.nn.base."""

from absl.testing import absltest
from jax import random

from oryx.core import state
from oryx.experimental.nn import base


class LayerParamsTest(absltest.TestCase):

  def test_defaults(self):
    layer_params = base.LayerParams()
    self.assertTupleEqual(((), (), ()), layer_params)


class DummyLayer(base.Layer):
  """Dummy layer for tests."""

  @classmethod
  def initialize(cls, rng, in_spec, layer_params):
    del rng, in_spec
    return layer_params

  def _call(self, x, **kwargs):
    params = self.params
    return params, x, kwargs

  def _update(self, x, **kwargs):
    return self.replace(state=self.state+100)


class LayerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._seed = random.PRNGKey(0)

  def test_init(self):
    layer_params = base.LayerParams((1, 2), 3)
    layer_init = DummyLayer(layer_params)
    layer = layer_init.init(self._seed, state.Shape((1, 1)))
    self.assertTupleEqual((1, 2), layer.params)
    self.assertEqual(3, layer.info)

  def test_init_adds_tuple_to_params(self):
    layer_params = base.LayerParams(1, 2)
    layer_init = DummyLayer(layer_params)
    layer = layer_init.init(self._seed, state.Shape((1, 1)))
    self.assertEqual(1, layer.params)
    self.assertEqual(2, layer.info)

  def test_call_pass_params(self):
    layer_params = base.LayerParams(params=1, state=1)
    layer_init = DummyLayer(layer_params)
    layer = layer_init.init(self._seed, state.Shape((1, 1)))
    outputs = layer(3)
    self.assertTupleEqual((1, 3, {}), outputs)

  def test_call_with_info(self):
    layer_params = base.LayerParams(params=1, state=1, info=2)
    layer_init = DummyLayer(layer_params)
    layer = layer_init.init(self._seed, state.Shape((1, 1)))
    outputs = layer(3)
    self.assertTupleEqual((1, 3, {}), outputs)

  def test_call_with_needs_rng(self):
    layer_params = base.LayerParams(params=1, state=1, info=2)
    layer_init = DummyLayer(layer_params)
    layer = layer_init.init(self._seed, state.Shape((1, 1)))
    outputs = layer(3, rng=1)
    self.assertTupleEqual((1, 3, {'rng': 1}), outputs)

  def test_call_with_has_training(self):
    layer_params = base.LayerParams(params=1, state=1, info=2)
    layer_init = DummyLayer(layer_params)
    layer = layer_init.init(self._seed, state.Shape((1, 1)))
    outputs = layer(3, training=True)
    self.assertTupleEqual((1, 3, {'training': True}), outputs)

  def test_update_state(self):
    layer_params = base.LayerParams(params=1, state=2)
    layer_init = DummyLayer(layer_params)
    layer = layer_init.init(self._seed, state.Shape((1, 1)))
    self.assertEqual(layer.state, 2)
    new_layer = layer.update(3)
    self.assertEqual(new_layer.state, 102)

  def test_to_string(self):
    layer_params = base.LayerParams(params=1, state=1, info=2)
    layer_init = DummyLayer(layer_params)
    layer = layer_init.init(self._seed, state.Shape((1, 1)))
    exp_str = ('DummyLayer(params=1, info=2)')
    self.assertEqual(exp_str, repr(layer))

  def test_flatten(self):
    layer_params = base.LayerParams(params=(1, 2), state=3)
    layer_init = DummyLayer(layer_params)
    layer = layer_init.init(self._seed, state.Shape((1, 1)), name='foo')
    self.assertTupleEqual((((1, 2), 3), ((), 'foo')),
                          layer.flatten())

  def test_unflatten(self):
    layer = DummyLayer.unflatten((3, 'foo'), ((1, 2), ()))
    self.assertTupleEqual((1, 2), layer.params)
    self.assertEqual(3, layer.info)
    self.assertEqual('foo', layer.name)

  def test_flatten_and_unflatten(self):
    layer_params = base.LayerParams((1, 2), 3)
    layer_init = DummyLayer(layer_params)
    layer = layer_init.init(self._seed, state.Shape((1, 1)))
    xs, data = layer.flatten()
    new_layer = DummyLayer.unflatten(data, xs)
    self.assertTupleEqual(layer.params, new_layer.params)
    self.assertEqual(layer.info, new_layer.info)
    self.assertEqual(layer.name, new_layer.name)


if __name__ == '__main__':
  absltest.main()
