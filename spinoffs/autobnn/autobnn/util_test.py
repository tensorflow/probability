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
"""Tests for util.py."""

import jax
import jax.numpy as jnp
import numpy as np
from autobnn import kernels
from autobnn import util
from tensorflow_probability.substrates.jax.internal import test_util


class UtilTest(test_util.TestCase):

  def test_suggest_periods(self):
    self.assertListEqual([], util.suggest_periods([1 for _ in range(20)]))
    self.assertListEqual(
        [2.0], util.suggest_periods([i % 2 for i in range(20)])
    )
    np.testing.assert_allclose(
        [20.0],
        util.suggest_periods(
            [jnp.sin(2.0 * jnp.pi * i / 20.0) for i in range(100)]
        ),
    )
    # suggest_periods is robust against small linear trends ...
    np.testing.assert_allclose(
        [20.0],
        util.suggest_periods(
            [0.01 * i + jnp.sin(2.0 * jnp.pi * i / 20.0) for i in range(100)]
        ),
    )
    # but sort of falls apart currently for large linear trends.
    np.testing.assert_allclose(
        [50.0, 100.0 / 3.0],
        util.suggest_periods(
            [i + jnp.sin(2.0 * jnp.pi * i / 20.0) for i in range(100)]
        ),
    )

  def test_transform(self):
    seed = jax.random.PRNGKey(20231018)
    bnn = kernels.LinearBNN(width=5)
    bnn.likelihood_model.noise_min = 0.2
    transform, _, _ = util.make_transforms(bnn)
    p = bnn.init(seed, jnp.ones((1, 10), dtype=jnp.float32))

    # Softplus(low=0.2) bijector
    self.assertEqual(
        0.2 + jax.nn.softplus(p['params']['noise_scale']),
        transform(p)['params']['noise_scale'],
    )
    self.assertEqual(
        jnp.exp(p['params']['amplitude']), transform(p)['params']['amplitude']
    )

    # Identity bijector
    self.assertAllEqual(
        p['params']['dense2']['kernel'],
        transform(p)['params']['dense2']['kernel'],
    )


if __name__ == '__main__':
  test_util.main()
