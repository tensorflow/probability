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
"""Tests for tensorflow_probability.spinoffs.oryx.experimental.nn.convolution."""

from absl.testing import absltest
import jax
from jax import random

from oryx.core import state
from oryx.experimental.nn import convolution


class ConvolutionTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._seed = random.PRNGKey(0)

  def test_conv_filter_shape(self):
    data_rng, net_rng = random.split(self._seed)
    x = random.normal(data_rng, (28, 28, 1))

    net_init = convolution.Conv(
        64, (3, 3),
        strides=(1, 1),
        padding='SAME'
    )
    out_shape = net_init.spec(state.Shape((28, 28, 1))).shape
    net = net_init.init(net_rng, state.Shape((28, 28, 1)))
    self.assertEqual(out_shape, (28, 28, 64))
    self.assertEqual(net(x).shape, out_shape)

  def test_conv_kernel_shape(self):
    data_rng, net_rng = random.split(self._seed)
    x = random.normal(data_rng, (28, 28, 1))

    net_init = convolution.Conv(
        64, (5, 5),
        strides=(1, 1),
        padding='VALID'
    )
    out_shape = net_init.spec(state.Shape((28, 28, 1))).shape
    net = net_init.init(net_rng, state.Shape((28, 28, 1)))
    self.assertEqual(out_shape, (24, 24, 64))
    self.assertEqual(net(x).shape, out_shape)

  def test_conv_padding_shape(self):
    data_rng, net_rng = random.split(self._seed)
    x = random.normal(data_rng, (28, 28, 1))

    net_init = convolution.Conv(
        64, (3, 3),
        strides=(1, 1),
        padding='VALID'
    )
    out_shape = net_init.spec(state.Shape((28, 28, 1))).shape
    net = net_init.init(net_rng, state.Shape((28, 28, 1)))
    self.assertEqual(out_shape, (26, 26, 64))
    self.assertEqual(net(x).shape, out_shape)

  def test_conv_strides_shape(self):
    data_rng, net_rng = random.split(self._seed)
    x = random.normal(data_rng, (28, 28, 1))

    net_init = convolution.Conv(
        64, (2, 2),
        strides=(2, 2),
        padding='VALID'
    )
    out_shape = net_init.spec(state.Shape((28, 28, 1))).shape
    net = net_init.init(net_rng, state.Shape((28, 28, 1)))
    self.assertEqual(out_shape, (14, 14, 64))

    net_init = convolution.Conv(
        64, (3, 3),
        strides=(2, 2),
        padding='VALID'
    )
    out_shape = net_init.spec(state.Shape((28, 28, 1))).shape
    net = net_init.init(net_rng, state.Shape((28, 28, 1)))
    self.assertEqual(out_shape, (13, 13, 64))
    self.assertEqual(net(x).shape, out_shape)

  def test_deconv_filter_shape(self):
    data_rng, net_rng = random.split(self._seed)
    x = random.normal(data_rng, (28, 28, 1))

    net_init = convolution.Deconv(
        64, (3, 3),
        strides=(1, 1),
        padding='SAME'
    )
    out_shape = net_init.spec(state.Shape((28, 28, 1))).shape
    net = net_init.init(net_rng, state.Shape((28, 28, 1)))
    self.assertEqual(out_shape, (28, 28, 64))
    self.assertEqual(net(x).shape, out_shape)

  def test_deconv_kernel_shape(self):
    data_rng, net_rng = random.split(self._seed)
    x = random.normal(data_rng, (28, 28, 1))

    net_init = convolution.Deconv(
        64, (5, 5),
        strides=(1, 1),
        padding='VALID'
    )
    out_shape = net_init.spec(state.Shape((28, 28, 1))).shape
    net = net_init.init(net_rng, state.Shape((28, 28, 1)))
    self.assertEqual(out_shape, (32, 32, 64))
    self.assertEqual(net(x).shape, out_shape)

  def test_deconv_padding_shape(self):
    data_rng, net_rng = random.split(self._seed)
    x = random.normal(data_rng, (28, 28, 1))

    net_init = convolution.Deconv(
        64, (3, 3),
        strides=(1, 1),
        padding='VALID'
    )
    out_shape = net_init.spec(state.Shape((28, 28, 1))).shape
    net = net_init.init(net_rng, state.Shape((28, 28, 1)))
    self.assertEqual(out_shape, (30, 30, 64))
    self.assertEqual(net(x).shape, out_shape)

  def test_deconv_strides_shape(self):
    data_rng, net_rng = random.split(self._seed)
    x = random.normal(data_rng, (28, 28, 1))

    net_init = convolution.Deconv(
        64, (2, 2),
        strides=(2, 2),
        padding='VALID'
    )
    out_shape = net_init.spec(state.Shape((28, 28, 1))).shape
    net = net_init.init(net_rng, state.Shape((28, 28, 1)))
    self.assertEqual(out_shape, (56, 56, 64))
    self.assertEqual(net(x).shape, out_shape)

    net_init = convolution.Deconv(
        64, (3, 3),
        strides=(2, 2),
        padding='VALID'
    )
    out_shape = net_init.spec(state.Shape((28, 28, 1))).shape
    net = net_init.init(net_rng, state.Shape((28, 28, 1)))
    self.assertEqual(out_shape, (57, 57, 64))
    self.assertEqual(net(x).shape, out_shape)

  def test_conv_vmap(self):
    data_rng, net_rng = random.split(self._seed)
    x = random.normal(data_rng, (10, 28, 28, 1))

    net_init = convolution.Conv(
        64, (2, 2),
        strides=(2, 2),
        padding='VALID'
    )
    with self.assertRaises(ValueError):
      out_shape = net_init.spec(state.Shape((10, 28, 28, 1))).shape

    out_shape = net_init.spec(state.Shape((28, 28, 1))).shape
    net = net_init.init(net_rng, state.Shape((28, 28, 1)))
    with self.assertRaises(ValueError):
      net(x)
    y = jax.vmap(net)(x)
    self.assertEqual(y.shape, (10,) + out_shape)

  def test_deconv_vmap(self):
    data_rng, net_rng = random.split(self._seed)
    x = random.normal(data_rng, (10, 28, 28, 1))

    net_init = convolution.Deconv(
        64, (2, 2),
        strides=(2, 2),
        padding='VALID'
    )
    with self.assertRaises(ValueError):
      out_shape = net_init.spec(state.Shape((10, 28, 28, 1))).shape

    out_shape = net_init.spec(state.Shape((28, 28, 1))).shape
    net = net_init.init(net_rng, state.Shape((28, 28, 1)))
    with self.assertRaises(ValueError):
      net(x)
    self.assertEqual(jax.vmap(net)(x).shape, (10,) + out_shape)

if __name__ == '__main__':
  absltest.main()
