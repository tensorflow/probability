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
"""Tests for tensorflow_probability.spinoffs.oryx.core.serialize."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import random
import jax.numpy as np
import numpy as onp
from oryx.core import state
from oryx.core.serialize import deserialize
from oryx.core.serialize import serialize
from oryx.experimental import nn


templates = {
    'dense': nn.Dense(200),
    'dense_serial': nn.Dense(200) >> nn.Dense(200),
    'relu': nn.Relu(),
    'dense_relu': nn.Dense(200) >> nn.Relu(),
    'convnet': (
        nn.Reshape((28, 28, 1)) >> nn.Conv(32, (5, 5))
        >> nn.MaxPooling((2, 2), (2, 2))),
    'dropout': nn.Dropout(0.5)
}


class SerializeTest(parameterized.TestCase):

  @parameterized.named_parameters(templates.items())
  def test_serialize(self, template):
    network = state.init(template)(random.PRNGKey(0), state.Shape(784))
    network2 = deserialize(serialize(network))

    rng = random.PRNGKey(0)
    onp.testing.assert_array_equal(
        jax.vmap(lambda x: network(x, rng=rng))(np.ones([10, 784])),
        jax.vmap(lambda x: network2(x, rng=rng))(np.ones([10, 784])))

if __name__ == '__main__':
  absltest.main()
