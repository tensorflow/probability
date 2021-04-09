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
"""Tests for tensorflow_probability.spinoffs.oryx.experimental.matching.jax_rewrite."""

from absl.testing import absltest

import jax
from jax import lax
import jax.numpy as jnp

from oryx.experimental.matching import jax_rewrite as jr
from oryx.internal import test_util


Exp = lambda x: jr.Primitive(lax.exp_p, (x,), jr.Params())


class JaxExpressionTest(test_util.TestCase):

  def test_evaluate_value_should_return_value(self):
    self.assertEqual(jr.evaluate(1., {}), 1.)
    self.assertTrue((jr.evaluate(jnp.ones(5), {}) == jnp.ones(5)).all())

  def test_evaluate_literal_should_evaluate_to_value(self):
    self.assertEqual(jr.evaluate(jr.Literal(1.), {}), 1.)
    self.assertTrue((jr.evaluate(jr.Literal(jnp.ones(5)),
                                 {}) == jnp.ones(5)).all())

  def test_evaluate_jaxvar_should_look_up_name_in_environment(self):
    self.assertEqual(
        jr.evaluate(jr.JaxVar('a', (), jnp.float32), {'a': 1.,
                                                      'b': 2.}), 1.)

  def test_evaluate_tuple_should_recursively_evaluate_values(self):
    self.assertTupleEqual(
        jr.evaluate((jr.Literal(1.), 2.), {}), (1., 2.))
    self.assertTupleEqual(
        jr.evaluate((jr.JaxVar('a', (), jnp.float32), 2.), {'a': 1.}), (1., 2.))

  def test_primitive_should_evaluate_to_jax_values(self):
    expr = Exp(0.)
    self.assertEqual(jr.evaluate(expr, {}), jnp.exp(0.))

    expr = jr.Primitive(lax.add_p, (1., 2.), jr.Params())
    self.assertEqual(jr.evaluate(expr, {}), 3.)

    expr = jr.Primitive(lax.add_p, (jr.JaxVar('a', (), jnp.float32), 2.),
                        jr.Params())
    self.assertEqual(jr.evaluate(expr, {'a': 1.}), 3.)

  def test_primitive_should_infer_shape_dtype_correctly(self):
    expr = Exp(0.)
    self.assertTupleEqual(expr.shape, ())
    self.assertEqual(expr.dtype, jnp.float32)

    expr = Exp(jr.JaxVar('a', (5,), jnp.float32))
    self.assertTupleEqual(expr.shape, (5,))
    self.assertEqual(expr.dtype, jnp.float32)

    expr = jr.Primitive(lax.add_p, (jr.Literal(1), jr.Literal(2)),
                        jr.Params())
    self.assertTupleEqual(expr.shape, ())
    self.assertEqual(expr.dtype, jnp.int32)

  def test_call_primitive_should_include_call_in_trace(self):
    exp_expr = Exp(jr.Literal(0.))
    call_expr = jr.CallPrimitive(jax.core.call_p, (), (exp_expr,), jr.Params(),
                                 [])
    jaxpr = jax.make_jaxpr(lambda: jr.evaluate(call_expr, {}))()
    self.assertEqual(jaxpr.jaxpr.eqns[0].primitive, jax.core.call_p)

  def test_call_primitive_shape_and_dtype_are_multi_part(self):
    exp_expr = Exp(jr.Literal(0.))
    call_expr = jr.CallPrimitive(jax.core.call_p, (), (exp_expr,), jr.Params(),
                                 [])
    self.assertTupleEqual(call_expr.shape, ((),))
    self.assertEqual(call_expr.dtype, (jnp.float32,))

  def test_part_infers_correct_shape_dtype(self):
    call_expr = jr.CallPrimitive(jax.core.call_p, (),
                                 (jr.Literal(0.), jr.Literal(1)), jr.Params(),
                                 [])
    p0_expr = jr.Part(call_expr, 0)
    p1_expr = jr.Part(call_expr, 1)
    self.assertTupleEqual(p0_expr.shape, ())
    self.assertTupleEqual(p1_expr.shape, ())
    self.assertEqual(p0_expr.dtype, jnp.float32)
    self.assertEqual(p1_expr.dtype, jnp.int32)
    self.assertEqual(jr.evaluate(p0_expr, {}), 0.)
    self.assertEqual(jr.evaluate(p1_expr, {}), 1)


if __name__ == '__main__':
  absltest.main()
