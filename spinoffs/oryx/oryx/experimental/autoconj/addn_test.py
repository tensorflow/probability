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
"""Tests for tensorflow_probability.spinoffs.oryx.experimental.autoconj.addn."""
from absl.testing import absltest

import jax.numpy as jnp

import numpy as np

from oryx.experimental.autoconj import addn
from oryx.experimental.matching import jax_rewrite as jr
from oryx.experimental.matching import matcher
from oryx.experimental.matching import rules
from oryx.internal import test_util

Var = matcher.Var
Segment = matcher.Segment
JaxVar = jr.JaxVar
AddN = addn.AddN


class AddNTest(test_util.TestCase):

  def test_can_match_addn_components(self):
    x = JaxVar('x', (5,), jnp.float32)
    op = AddN((x, x))
    pattern = AddN((matcher.Segment('args'),))
    self.assertDictEqual(
        matcher.match(pattern, op), {
            'args': (x, x)
        })

  def test_can_replace_addn_operands(self):
    x = JaxVar('x', (5,), jnp.float32)
    y = JaxVar('y', (5,), jnp.float32)
    z = JaxVar('y', (5,), jnp.float32)
    op = AddN((x, y))
    pattern = AddN((matcher.Segment('args'),))
    def replace_with_z(args):
      del args
      return AddN((z, z))
    replace_rule = rules.make_rule(pattern, replace_with_z)
    replaced_op = replace_rule(op)
    self.assertEqual(replaced_op, AddN((z, z)))

  def test_addn_correctly_infers_shape_and_dtype(self):
    x = JaxVar('x', (5, 2), jnp.float32)
    y = JaxVar('y', (5, 2), jnp.float32)
    op = AddN((x, y))
    self.assertEqual(op.dtype, jnp.float32)
    self.assertTupleEqual(op.shape, (5, 2))

  def test_addn_evaluates_to_correct_value(self):
    x = JaxVar('x', (5, 2), jnp.float32)
    y = JaxVar('y', (5, 2), jnp.float32)
    z = JaxVar('z', (5, 2), jnp.float32)
    op = AddN((x, y, z))
    x_val = jnp.arange(10.).reshape((5, 2))
    y_val = jnp.arange(10., 20.).reshape((5, 2))
    z_val = jnp.arange(20., 30.).reshape((5, 2))
    np.testing.assert_allclose(
        op.evaluate(dict(x=x_val, y=y_val, z=z_val)),
        x_val + y_val + z_val)


if __name__ == '__main__':
  absltest.main()
