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
"""Tests for tensorflow_probability.spinoffs.oryx.experimental.nn.pooling."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import random
from jax import test_util as jtu
import numpy as np

from oryx.core import state
from oryx.experimental.nn import pooling


def shape4d_parameters():
  testcase_name = ('_window_shape={}_strides={}_padding={}_in_shape={}'
                   '_pool_class={}')
  for window_shape in [(1, 1), (2, 2), (2, 3)]:
    for strides in [None, (2, 1), (1, 2)]:
      for padding in ['SAME', 'VALID']:
        for in_shape in [(5, 6, 1)]:
          for pool_class in [pooling.MaxPooling,
                             pooling.SumPooling,
                             pooling.AvgPooling]:
            yield {'testcase_name': testcase_name.format(window_shape, strides,
                                                         padding, in_shape,
                                                         pool_class),
                   'window_shape': window_shape,
                   'strides': strides,
                   'padding': padding,
                   'in_shape': in_shape,
                   'pool_class': pool_class}


def shape3d_parameters():
  testcase_name = ('_window_shape={}_strides={}_padding={}_in_shape={}'
                   '_pool_class={}')
  for window_shape in [(1,), (2,), (3,)]:
    for strides in [None, (1,), (2,)]:
      for padding in ['SAME', 'VALID']:
        for in_shape in [(5, 1)]:
          for pool_class in [pooling.MaxPooling,
                             pooling.SumPooling,
                             pooling.AvgPooling]:
            yield {'testcase_name': testcase_name.format(window_shape, strides,
                                                         padding, in_shape,
                                                         pool_class),
                   'window_shape': window_shape,
                   'strides': strides,
                   'padding': padding,
                   'in_shape': in_shape,
                   'pool_class': pool_class}


class PoolingTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._seed = random.PRNGKey(0)

  @parameterized.named_parameters(jtu.cases_from_list(shape4d_parameters()))
  def test_shapes4d(self, window_shape, padding, strides, in_shape, pool_class):
    net_init = pool_class(window_shape, strides, padding)
    net_rng, data_rng = random.split(self._seed)
    out_shape = net_init.spec(state.Shape(in_shape)).shape
    layer = net_init.init(net_rng, state.Shape(in_shape))
    x = random.normal(data_rng, in_shape)
    result = layer(x)
    self.assertEqual(result.shape, out_shape)

  @parameterized.named_parameters(jtu.cases_from_list(shape3d_parameters()))
  def test_shapes3d(self, window_shape, padding, strides, in_shape, pool_class):
    net_init = pool_class(window_shape, strides, padding)
    net_rng, data_rng = random.split(self._seed)
    out_shape = net_init.spec(state.Shape(in_shape)).shape
    layer = net_init.init(net_rng, state.Shape(in_shape))
    x = random.normal(data_rng, in_shape)
    result = layer(x)
    self.assertEqual(result.shape, out_shape)

  def test_max_pool(self):
    in_shape = (3, 3, 1)
    net_init = pooling.MaxPooling((2, 2))
    net_rng = self._seed
    out_shape = net_init.spec(state.Shape(in_shape)).shape
    layer = net_init.init(net_rng, state.Shape(in_shape))
    x = np.array([[-1, 0, -1], [0, 1, 0], [-1, 0, -1]])
    x = np.reshape(x, in_shape)
    result = layer(x)
    self.assertEqual(result.shape, out_shape)
    np.testing.assert_equal(result, np.ones(out_shape))

  def test_max_pool_batched(self):
    in_shape = (3, 3, 1)
    batch_size = 10
    batch_in_shape = (batch_size,) + in_shape
    net_init = pooling.MaxPooling((2, 2))
    net_rng = self._seed
    with self.assertRaises(ValueError):
      _ = net_init.spec(state.Shape(batch_in_shape)).shape
    out_shape = net_init.spec(state.Shape(in_shape)).shape
    batch_out_shape = (batch_size,) + out_shape
    layer = net_init.init(net_rng, state.Shape(in_shape))

    x = np.tile(np.array([[-1, 0, -1], [0, 1, 0], [-1, 0, -1]])[None],
                (batch_size, 1, 1))
    x = np.reshape(x, batch_in_shape)
    with self.assertRaises(ValueError):
      layer(x)
    result = jax.vmap(layer)(x)
    self.assertEqual(result.shape, batch_out_shape)
    np.testing.assert_equal(result, np.ones(batch_out_shape))

  def test_sum_pool(self):
    in_shape = (3, 3, 1)
    net_init = pooling.SumPooling((2, 2))
    net_rng = self._seed
    out_shape = net_init.spec(state.Shape(in_shape)).shape
    layer = net_init.init(net_rng, state.Shape(in_shape))
    x = np.array([[-1, 0, -1], [0, 1, 0], [-1, 0, -1]])
    x = np.reshape(x, in_shape)
    result = layer(x)
    self.assertEqual(result.shape, out_shape)
    np.testing.assert_equal(result, np.zeros(out_shape))

  def test_sum_pool_batched(self):
    in_shape = (3, 3, 1)
    batch_size = 10
    batch_in_shape = (batch_size,) + in_shape
    net_init = pooling.SumPooling((2, 2))
    net_rng = self._seed
    with self.assertRaises(ValueError):
      _ = net_init.spec(state.Shape(batch_in_shape)).shape
    out_shape = net_init.spec(state.Shape(in_shape)).shape
    batch_out_shape = (batch_size,) + out_shape
    layer = net_init.init(net_rng, state.Shape(in_shape))

    x = np.tile(np.array([[-1, 0, -1], [0, 1, 0], [-1, 0, -1]])[None],
                (batch_size, 1, 1))
    x = np.reshape(x, batch_in_shape)
    with self.assertRaises(ValueError):
      layer(x)
    result = jax.vmap(layer)(x)
    self.assertEqual(result.shape, batch_out_shape)
    np.testing.assert_equal(result, np.zeros(batch_out_shape))

  def test_avg_pool(self):
    in_shape = (3, 3, 1)
    net_init = pooling.AvgPooling((2, 2))
    net_rng = self._seed
    out_shape = net_init.spec(state.Shape(in_shape)).shape
    layer = net_init.init(net_rng, state.Shape(in_shape))
    x = np.array([[-1, 0, -1], [0, 5, 0], [-1, 0, -1]])
    x = np.reshape(x, in_shape)
    result = layer(x)
    self.assertEqual(result.shape, out_shape)
    np.testing.assert_equal(result, np.ones(out_shape))

  def test_avg_pool_batched(self):
    in_shape = (3, 3, 1)
    batch_size = 10
    batch_in_shape = (batch_size,) + in_shape
    net_init = pooling.AvgPooling((2, 2))
    net_rng = self._seed
    with self.assertRaises(ValueError):
      _ = net_init.spec(state.Shape(batch_in_shape)).shape
    out_shape = net_init.spec(state.Shape(in_shape)).shape
    batch_out_shape = (batch_size,) + out_shape
    layer = net_init.init(net_rng, state.Shape(in_shape))

    x = np.tile(np.array([[-1, 0, -1], [0, 5, 0], [-1, 0, -1]])[None],
                (batch_size, 1, 1))
    x = np.reshape(x, batch_in_shape)
    with self.assertRaises(ValueError):
      layer(x)
    result = jax.vmap(layer)(x)
    self.assertEqual(result.shape, batch_out_shape)
    np.testing.assert_equal(result, np.ones(batch_out_shape))


if __name__ == '__main__':
  absltest.main()
