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
"""Tests for canonicalize."""
from absl.testing import absltest

from jax import lax
from jax import random
import jax.numpy as jnp
import numpy as np

from oryx.experimental.autoconj import canonicalize
from oryx.experimental.autoconj import einsum
from oryx.experimental.matching import jax_rewrite as jr
from oryx.internal import test_util

JaxVar = jr.JaxVar
Primitive = jr.Primitive
Params = jr.Params


class RewriteOperationTest(test_util.TestCase):

  def test_can_rewrite_transpose_to_einsum(self):
    x = JaxVar('x', (3, 4), jnp.float32)
    transpose_op = Primitive(lax.transpose_p, (x,), Params(permutation=(1, 0)))
    einsum_op = canonicalize.transpose_as_einsum(transpose_op)
    self.assertIsInstance(einsum_op, einsum.Einsum)

    x_val = jnp.arange(12).reshape((3, 4))
    np.testing.assert_allclose(
        transpose_op.evaluate(dict(x=x_val)),
        einsum_op.evaluate(dict(x=x_val)),
    )

  def test_can_rewrite_squeeze_to_einsum(self):
    x = JaxVar('x', (3, 1, 4), jnp.float32)
    squeeze_op = Primitive(lax.squeeze_p, (x,), Params(dimensions=(1,)))
    einsum_op = canonicalize.squeeze_as_einsum(squeeze_op)
    self.assertIsInstance(einsum_op, einsum.Einsum)

    x_val = jnp.arange(12).reshape((3, 1, 4))
    np.testing.assert_allclose(
        squeeze_op.evaluate(dict(x=x_val)),
        einsum_op.evaluate(dict(x=x_val)),
    )

  def test_can_rewrite_dot_to_einsum(self):
    x = JaxVar('x', (3, 4), jnp.float32)
    y = JaxVar('y', (4, 2), jnp.float32)
    contract_dims = ((1,), (0,))
    batch_dims = ((), ())
    dot_op = Primitive(
        lax.dot_general_p, (x, y),
        Params(
            dimension_numbers=(contract_dims, batch_dims),
            precision=None,
            preferred_element_type=None))
    einsum_op = canonicalize.dot_as_einsum(dot_op)
    self.assertIsInstance(einsum_op, einsum.Einsum)

    x_val = jnp.arange(12).reshape((3, 4))
    y_val = jnp.arange(8).reshape((4, 2))
    self.assertTupleEqual(dot_op.shape, (3, 2))
    self.assertTupleEqual(einsum_op.shape, (3, 2))
    np.testing.assert_allclose(
        dot_op.evaluate(dict(x=x_val, y=y_val)),
        einsum_op.evaluate(dict(x=x_val, y=y_val)),
    )

    x = JaxVar('x', (3, 2, 4), jnp.float32)
    y = JaxVar('y', (4, 2, 2), jnp.float32)
    contract_dims = ((1,), (1,))
    batch_dims = ((), ())
    dot_op = Primitive(
        lax.dot_general_p, (x, y),
        Params(
            dimension_numbers=(contract_dims, batch_dims),
            precision=None,
            preferred_element_type=None))
    einsum_op = canonicalize.dot_as_einsum(dot_op)
    self.assertIsInstance(einsum_op, einsum.Einsum)

    x_val = jnp.arange(24).reshape((3, 2, 4))
    y_val = jnp.arange(16).reshape((4, 2, 2))
    self.assertTupleEqual(dot_op.shape, (3, 4, 4, 2))
    self.assertTupleEqual(einsum_op.shape, (3, 4, 4, 2))
    np.testing.assert_allclose(
        dot_op.evaluate(dict(x=x_val, y=y_val)),
        einsum_op.evaluate(dict(x=x_val, y=y_val)),
    )

    x = JaxVar('x', (5, 3, 4), jnp.float32)
    y = JaxVar('y', (5, 4, 2), jnp.float32)
    contract_dims = ((2,), (1,))
    batch_dims = ((0,), (0,))
    dot_op = Primitive(
        lax.dot_general_p, (x, y),
        Params(
            dimension_numbers=(contract_dims, batch_dims),
            precision=None,
            preferred_element_type=None))
    einsum_op = canonicalize.dot_as_einsum(dot_op)
    self.assertIsInstance(einsum_op, einsum.Einsum)

    x_val = jnp.arange(60).reshape((5, 3, 4))
    y_val = jnp.arange(40).reshape((5, 4, 2))
    self.assertTupleEqual(dot_op.shape, (5, 3, 2))
    self.assertTupleEqual(einsum_op.shape, (5, 3, 2))
    np.testing.assert_allclose(
        dot_op.evaluate(dict(x=x_val, y=y_val)),
        einsum_op.evaluate(dict(x=x_val, y=y_val)),
    )

  def test_reduce_sum_as_einsum(self):
    x = JaxVar('x', (4, 2), jnp.float32)
    reduce_sum_op = Primitive(lax.reduce_sum_p, (x,), Params(axes=(0,)))
    einsum_op = canonicalize.reduce_sum_as_einsum(reduce_sum_op)
    self.assertIsInstance(einsum_op, einsum.Einsum)
    self.assertTupleEqual(reduce_sum_op.shape, (2,))
    self.assertTupleEqual(einsum_op.shape, (2,))
    x_val = jnp.arange(8).reshape((4, 2))
    np.testing.assert_allclose(
        reduce_sum_op.evaluate(dict(x=x_val)),
        einsum_op.evaluate(dict(x=x_val)),
    )

    reduce_sum_op = Primitive(lax.reduce_sum_p, (x,), Params(axes=(1,)))
    einsum_op = canonicalize.reduce_sum_as_einsum(reduce_sum_op)
    self.assertIsInstance(einsum_op, einsum.Einsum)
    self.assertTupleEqual(reduce_sum_op.shape, (4,))
    self.assertTupleEqual(einsum_op.shape, (4,))
    x_val = jnp.arange(8).reshape((4, 2))
    np.testing.assert_allclose(
        reduce_sum_op.evaluate(dict(x=x_val)),
        einsum_op.evaluate(dict(x=x_val)),
    )

    reduce_sum_op = Primitive(lax.reduce_sum_p, (x,), Params(axes=(0, 1)))
    einsum_op = canonicalize.reduce_sum_as_einsum(reduce_sum_op)
    self.assertIsInstance(einsum_op, einsum.Einsum)
    self.assertTupleEqual(reduce_sum_op.shape, ())
    self.assertTupleEqual(einsum_op.shape, ())
    x_val = jnp.arange(8).reshape((4, 2))
    np.testing.assert_allclose(
        reduce_sum_op.evaluate(dict(x=x_val)),
        einsum_op.evaluate(dict(x=x_val)),
    )

  def test_compose_einsums(self):
    w = JaxVar('w', (4, 5), jnp.float32)
    x = JaxVar('x', (5, 2), jnp.float32)
    y = JaxVar('y', (2, 3), jnp.float32)
    z = JaxVar('z', (3, 1), jnp.float32)

    child_op = einsum.Einsum('ab,bc->ac', (x, y))
    parent_op = einsum.Einsum('ab,bc,cd->ad', (w, child_op, z))

    einsum_op = canonicalize.compose_einsums(parent_op)

    self.assertLen(einsum_op.operands, 4)
    keys = random.split(random.PRNGKey(0), 4)
    env = {
        a.name: random.normal(key, a.shape, dtype=a.dtype)
        for key, a in zip(keys, [w, x, y, z])
    }
    np.testing.assert_allclose(
        parent_op.evaluate(env), einsum_op.evaluate(env), rtol=1e-6, atol=1e-6)


class CanonicalizeTest(test_util.TestCase):

  def test_can_rewrite_nested_expression_into_single_einsum(self):
    w = JaxVar('w', (4, 5), jnp.float32)
    x = JaxVar('x', (5, 2), jnp.float32)
    y = JaxVar('y', (2, 3), jnp.float32)
    z = JaxVar('z', (3, 1), jnp.float32)

    dot_op = Primitive(lax.dot_general_p, (w, x),
                       Params(
                           dimension_numbers=(((1,), (0,)), ((), ())),
                           precision=None,
                           preferred_element_type=None))  # (4, 2)
    transpose_op = Primitive(lax.transpose_p, (dot_op,),
                             Params(permutation=(1, 0)))  # (2, 4)
    squeeze_op = Primitive(lax.squeeze_p, (z,), Params(dimensions=(1,)))  # (3,)
    dot_op = Primitive(lax.dot_general_p, (squeeze_op, y),
                       Params(
                           dimension_numbers=(((0,), (1,)), ((), ())),
                           precision=None,
                           preferred_element_type=None))  # (2,)
    reduce_sum_op = Primitive(lax.reduce_sum_p, (transpose_op,),
                              Params(axes=(1,)))  # (2,)
    final_op = Primitive(lax.dot_general_p, (dot_op, reduce_sum_op),
                         Params(
                             dimension_numbers=(((0,), (0,)), ((), ())),
                             precision=None,
                             preferred_element_type=None))  # ()
    einsum_op = canonicalize.canonicalize(final_op)
    # Should have been rewritten to (einsum[ab,ca,de,ec->] z y w x)

    self.assertTupleEqual(final_op.shape, ())
    self.assertTupleEqual(einsum_op.shape, ())
    self.assertLen(einsum_op.operands, 4)
    for operand in einsum_op.operands:
      self.assertIsInstance(operand, JaxVar)

    keys = random.split(random.PRNGKey(0), 4)

    env = {
        a.name: random.normal(key, a.shape, dtype=a.dtype)
        for key, a in zip(keys, [w, x, y, z])
    }
    np.testing.assert_allclose(final_op.evaluate(env), einsum_op.evaluate(env),
                               rtol=1e-5, atol=1e-5)


if __name__ == '__main__':
  absltest.main()
