# Copyright 2024 The TensorFlow Probability Authors.
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
import jax
import jax.numpy as jnp

from discussion.robust_inverse_graphics.util import math_util
from discussion.robust_inverse_graphics.util import test_util


class MathUtilTest(test_util.TestCase):

  def test_transform_gradients(self):
    def f(x):
      return math_util.transform_gradients(x, lambda x: x + 1)

    grad = jax.grad(f)(0.0)
    self.assertAllEqual(grad, 2.0)

  def test_sanitize_gradients(self):
    def f(x):
      return jnp.sqrt(math_util.sanitize_gradients(x['x']))

    grad = jax.grad(f)({'x': 0.0})
    self.assertAllEqual(grad['x'], 0.0)

  def test_clip_gradients(self):
    def f(x):
      return jnp.square(math_util.clip_gradients(x['x'])).sum()

    grad_small = jax.grad(f)({'x': 0.1 * jnp.ones(3)})
    grad_big = jax.grad(f)({'x': jnp.ones(3)})

    self.assertAllClose(grad_small['x'], 2 * 0.1 * jnp.ones(3))
    self.assertAllClose(grad_big['x'], jnp.ones(3) / jnp.sqrt(3))

  def test_is_finite(self):
    self.assertTrue(math_util.is_finite([]))
    self.assertTrue(math_util.is_finite(0.))
    self.assertTrue(math_util.is_finite([0., 0.]))
    self.assertFalse(math_util.is_finite(float('nan')))
    self.assertFalse(math_util.is_finite([0., float('nan')]))


if __name__ == '__main__':
  test_util.main()
