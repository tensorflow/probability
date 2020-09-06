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
# Lint as: python3
"""Tests for tensorflow_probability.experimental.lazybones.deferred using Jax."""

from absl.testing import absltest

import jax
import numpy as np
import tensorflow_probability as tfp

lb = tfp.experimental.lazybones


class DeferredJaxTest(absltest.TestCase):

  def test_jax(self):
    jnp = lb.DeferredInput(jax.numpy)
    a = jnp.array([[1.], [2]])
    x = jnp.array([[1., 2.], [2., 3.]])
    b = jnp.einsum('ij,jk->ik', x, a)
    c = jnp.sum(b)

    self.assertTrue((b.shape == (2, 1)).eval())
    with lb.DeferredScope():
      b.value = np.array([4., 5, 6.])
      self.assertEqual(c.eval(), 15)
    self.assertEqual(c.eval(), 13)

    # Type change is allowed.
    x.value = np.ones((3, 2))
    self.assertEqual(b.shape.eval(), (3, 1))
    self.assertEqual(c.eval(), 9)


if __name__ == '__main__':
  absltest.main()
