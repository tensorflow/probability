# Copyright 2021 The TensorFlow Probability Authors.
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
"""Tests for tensorflow_probability.spinoffs.oryx.experimental.autoconj.einsum."""
from absl.testing import absltest

from jax import random
import jax.numpy as jnp

import numpy as np

from oryx.experimental.autoconj import einsum
from oryx.experimental.matching import jax_rewrite as jr
from oryx.experimental.matching import matcher
from oryx.experimental.matching import rules
from oryx.internal import test_util

Var = matcher.Var
Segment = matcher.Segment
JaxVar = jr.JaxVar
Einsum = einsum.Einsum


class EinsumTest(test_util.TestCase):

  def test_can_match_einsum_components(self):
    x = JaxVar('x', (5,), jnp.float32)
    op = Einsum('a,a->', (x, x))
    pattern = Einsum(Var('formula'), (matcher.Segment('args'),))
    self.assertDictEqual(
        matcher.match(pattern, op), {
            'formula': 'a,a->',
            'args': (x, x)
        })

  def test_can_replace_einsum_operands(self):
    x = JaxVar('x', (5,), jnp.float32)
    y = JaxVar('y', (5,), jnp.float32)
    z = JaxVar('y', (5,), jnp.float32)
    op = Einsum('a,a->', (x, y))
    pattern = Einsum(Var('formula'), (matcher.Segment('args'),))
    def replace_with_z(formula, args):
      del args
      return Einsum(formula, (z, z))
    replace_rule = rules.make_rule(pattern, replace_with_z)
    replaced_op = replace_rule(op)
    self.assertEqual(replaced_op, Einsum('a,a->', (z, z)))

  def test_einsum_correctly_infers_shape_and_dtype(self):
    x = JaxVar('x', (5, 2), jnp.float32)
    y = JaxVar('y', (2, 3), jnp.float32)
    op = Einsum('ab,bc->ac', (x, y))
    self.assertEqual(op.dtype, jnp.float32)
    self.assertTupleEqual(op.shape, (5, 3))

  def test_einsum_evaluates_to_correct_value(self):
    x = JaxVar('x', (5, 2), jnp.float32)
    y = JaxVar('y', (2, 3), jnp.float32)
    op = Einsum('ab,bc->ac', (x, y))
    x_val = jnp.arange(10.).reshape((5, 2))
    y_val = jnp.arange(6.).reshape((2, 3))
    np.testing.assert_allclose(
        op.evaluate(dict(x=x_val, y=y_val)),
        jnp.einsum('ab,bc->ac', x_val, y_val))


class EinsumOperationsTest(test_util.TestCase):

  def test_can_compose_nested_einsums_to_make_single_einsum(self):
    w = JaxVar('w', (4, 5), jnp.float32)
    x = JaxVar('x', (5, 2), jnp.float32)
    y = JaxVar('y', (2, 3), jnp.float32)
    z = JaxVar('z', (3, 1), jnp.float32)

    child_op = Einsum('ab,bc->ac', (x, y))
    parent_op = Einsum('ab,bc,cd->ad', (w, child_op, z))

    single_op = einsum.compose_einsums(parent_op.formula, (w,), child_op, (z,))
    self.assertEqual(single_op.dtype, parent_op.dtype)
    self.assertTupleEqual(single_op.shape, parent_op.shape)

    keys = random.split(random.PRNGKey(0), 4)

    env = {
        a.name: random.normal(key, a.shape, dtype=a.dtype)
        for key, a in zip(keys, [w, x, y, z])
    }
    np.testing.assert_allclose(parent_op.evaluate(env), single_op.evaluate(env),
                               rtol=1e-6, atol=1e-6)


if __name__ == '__main__':
  absltest.main()
