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
"""Tests for tensorflow_probability.spinoffs.oryx.core.interpreters.unzip."""

import functools
import os

from absl.testing import absltest
import jax
from jax import core as jax_core
from jax import linear_util as lu
import jax.numpy as np
import numpy as onp

from oryx.core import state
from oryx.core import trace_util
from oryx.core.interpreters import harvest
from oryx.core.interpreters import unzip
from oryx.internal import test_util

variable = state.variable
unzip_variable = functools.partial(unzip.unzip, tag=state.VARIABLE)


def call_impl(f, *args, **params):
  del params
  with jax_core.new_sublevel():
    return f.call_wrapped(*args)
call_p = jax_core.CallPrimitive('call')
call_bind = call_p.bind
call_p.def_impl(call_impl)


def call(f):
  def wrapped(*args, **kwargs):
    fun = lu.wrap_init(f, kwargs)
    flat_args, in_tree = jax.tree_flatten(args)
    flat_fun, out_tree = jax.flatten_fun_nokwargs(fun, in_tree)
    ans = call_p.bind(flat_fun, *flat_args)
    return jax.tree_unflatten(out_tree(), ans)
  return wrapped


def empty():
  return


def single(x):
  return x


def single_variable(x):
  y = variable(x, name='x')
  return y


def single_variable_plus_one(x):
  y = variable(x + 1, name='x')
  return y + 1


def two_variable_0(x, y):
  z = variable(x + 1, name='x')
  return y + z


def two_variable_1(x, y):
  z = variable(y + 1, name='x')
  return x + z


def two_variable(x, y):
  return variable(x, name='x') + variable(y, name='y')


def pytree(x, y):
  params = variable({'a': x, 'b': x + 1}, name='ab')
  return params['a'] + params['b'] + y


class UnzipTest(test_util.TestCase):

  def test_empty(self):
    init, apply = unzip_variable(empty)()
    self.assertDictEqual(init(), {})
    self.assertIsNone(apply(init()))

  def test_single(self):
    with self.assertRaisesRegex(
        ValueError, 'Variables do not cut dependence graph.'):
      unzip_variable(single)(np.ones(5))

    init, apply = unzip_variable(single, key_args=None)(np.ones(5))
    self.assertDictEqual(init(), {})
    onp.testing.assert_allclose(apply(init(), np.ones(5)), np.ones(5))

  def test_single_variable(self):
    init, apply = unzip_variable(single_variable, key_args=None)(np.ones(5))
    self.assertDictEqual(init(), {})
    onp.testing.assert_allclose(apply(init(), np.ones(5)), np.ones(5))

    init, apply = unzip_variable(single_variable)(np.ones(5))
    params = init(np.ones(5))
    truth = {'x': np.ones(5)}
    self.assertLen(params, len(truth))
    for name in truth:
      onp.testing.assert_allclose(params[name], truth[name])
    onp.testing.assert_allclose(apply(params), np.ones(5))

  def test_single_variable_plus_one(self):
    init, apply = unzip_variable(
        single_variable_plus_one, key_args=None)(
            np.ones(5))
    self.assertDictEqual(init(), {})
    onp.testing.assert_allclose(apply(init(), np.ones(5)), 3 * np.ones(5))

    init, apply = unzip_variable(single_variable_plus_one)(np.ones(5))
    params = init(np.ones(5))
    truth = {'x': 2 * np.ones(5)}
    self.assertLen(params, len(truth))
    for name in truth:
      onp.testing.assert_allclose(params[name], truth[name])
    onp.testing.assert_allclose(apply(params), 3 * np.ones(5))

  def test_two_variable_0(self):
    init, apply = unzip_variable(two_variable_0)(np.ones(5), np.ones(5))
    params = init(np.ones(5))
    truth = {'x': 2 * np.ones(5)}
    self.assertLen(params, len(truth))
    for name in truth:
      onp.testing.assert_allclose(params[name], truth[name])
    onp.testing.assert_allclose(apply(params, np.ones(5)), 3 * np.ones(5))

    with self.assertRaisesRegex(
        ValueError, 'Variables do not cut dependence graph.'):
      unzip_variable(two_variable_0, key_args=1)(np.ones(5), np.ones(5))

  def test_two_variable_1(self):
    with self.assertRaisesRegex(
        ValueError, 'Variables do not cut dependence graph.'):
      unzip_variable(two_variable_1)(np.ones(5), np.ones(5))

    init, apply = unzip_variable(
        two_variable_1, key_args=1)(np.ones(5), np.ones(5))
    params = init(np.ones(5))
    truth = {'x': 2 * np.ones(5)}
    self.assertLen(params, len(truth))
    for name in truth:
      onp.testing.assert_allclose(params[name], truth[name])
    onp.testing.assert_allclose(apply(params, np.ones(5)), 3 * np.ones(5))

  def test_two_variable(self):
    init, apply = unzip_variable(
        two_variable, key_args=0)(np.ones(5), np.ones(5))
    params = init(np.ones(5))
    truth = {'x': np.ones(5)}
    self.assertLen(params, len(truth))
    for name in truth:
      onp.testing.assert_allclose(params[name], truth[name])
    onp.testing.assert_allclose(apply(params, np.ones(5)), 2 * np.ones(5))

    init, apply = unzip_variable(
        two_variable, key_args=1)(np.ones(5), np.ones(5))
    params = init(np.ones(5))
    truth = {'y': np.ones(5)}
    self.assertLen(params, len(truth))
    for name in truth:
      onp.testing.assert_allclose(params[name], truth[name])
    onp.testing.assert_allclose(apply(params, np.ones(5)), 2 * np.ones(5))

  def test_nested_two_variable(self):
    init, apply = unzip_variable(two_variable)(np.ones(5), np.ones(5))
    bound = functools.partial(apply, init(np.ones(5)))
    bound_init, bound_apply = unzip_variable(bound)(np.ones(5))

    params = bound_init(np.ones(5))
    truth = {'y': np.ones(5)}
    self.assertLen(params, len(truth))
    for name in truth:
      onp.testing.assert_allclose(params[name], truth[name])
    onp.testing.assert_allclose(bound_apply(params), 2 * np.ones(5))

    init, apply = unzip_variable(
        two_variable, key_args=1)(np.ones(5), np.ones(5))
    bound = functools.partial(apply, init(np.ones(5)))
    bound_init, bound_apply = unzip_variable(bound)(np.ones(5))

    params = bound_init(np.ones(5))
    truth = {'x': np.ones(5)}
    self.assertLen(params, len(truth))
    for name in truth:
      onp.testing.assert_allclose(params[name], truth[name])
    onp.testing.assert_allclose(bound_apply(params), 2 * np.ones(5))

  def test_pytree(self):
    init, apply = unzip_variable(pytree)(np.ones(5), np.ones(5))
    params = init(np.ones(5))
    onp.testing.assert_allclose(params['ab']['a'], np.ones(5))
    onp.testing.assert_allclose(params['ab']['b'], 2 * np.ones(5))
    onp.testing.assert_allclose(apply(params, np.ones(5)), 4 * np.ones(5))

  def test_should_error_if_no_name_provided(self):
    def no_name(x):
      return variable(x, name=None)

    with self.assertRaisesRegex(ValueError, 'Must provide name for variable.'):
      unzip_variable(no_name)(np.ones(5))

  def test_should_error_if_duplicate_names(self):
    def duplicate_names(x):
      y1 = variable(x, name='y')
      y2 = variable(x + 1., name='y')
      return y1 + y2
    with self.assertRaisesRegex(
        ValueError, 'Cannot use duplicate variable name: y'):
      unzip_variable(duplicate_names)(np.ones(5))

  def test_should_unzip_function_with_jit_successfully(self):
    def function_with_jit(x):
      x = jax.jit(lambda x: x)(x)
      x = variable(x, name='x')
      return x

    init, apply = unzip_variable(function_with_jit)(1.)
    self.assertEqual(apply(init(1.)), 1.)

  def test_should_unzip_variables_inside_jit(self):
    def nested_jit(x):
      @jax.jit
      def bar(y):
        return variable(y, name='bar')

      init, _ = unzip_variable(bar)(x + 1.)
      return variable(init(x + 1.), name='foo')
    self.assertDictEqual(nested_jit(1.), {'bar': 2.})
    init = unzip_variable(nested_jit)(1.)[0]
    self.assertDictEqual(init(1.), {'foo': {'bar': 2.}})

  def test_should_not_inline_calls_without_variables(self):
    def inline_call(x):
      x = call(lambda x: x + 1)(x)
      x = variable(x, name='x')
      return call(lambda x: x + 1)(x)

    init, apply = unzip_variable(inline_call)(1.)
    self.assertDictEqual(init(1.), {'x': 2.})
    init_jaxpr = trace_util.stage(init)(1.)[0]
    self.assertIn(call_p, {eqn.primitive for eqn in init_jaxpr.jaxpr.eqns})
    apply_jaxpr = trace_util.stage(apply)(init(1.))[0]
    self.assertIn(call_p, {eqn.primitive for eqn in apply_jaxpr.jaxpr.eqns})

  def test_unzip_tracers_should_pass_through_call_after_variable(self):
    def inline_call(x):
      x = call(lambda x: variable(x, name='x'))(x)
      return call(lambda x: x + 1)(x)

    init, apply = unzip_variable(inline_call)(1.)
    self.assertDictEqual(init(1.), {'x': 1.})
    self.assertEqual(apply(init(1.)), 2.)

  def test_should_lift_tracers_from_closed_variables(self):
    def closure(x):
      @call
      def inner(y):
        return variable(y + x, name='y')
      return inner(x)

    init, apply = unzip_variable(closure)(1.)
    self.assertDictEqual(init(1.), {'y': 2.})
    self.assertEqual(apply(init(1.)), 2.)

  def test_should_variable_jitted_function_successfully(self):

    @jax.jit
    def jitted(x):
      return variable(x + 1., name='x')

    init, apply = unzip_variable(jitted)(1.)
    self.assertDictEqual(init(1.), {'x': 2.})
    self.assertEqual(apply(init(1.)), 2.)

  def test_should_unzip_nested_jits_successfully(self):
    @jax.jit
    def jitted(x):
      return variable(jax.jit(lambda x: x)(x + 1.), name='x')

    init, apply = unzip_variable(jitted)(1.)
    self.assertDictEqual(init(1.), {'x': 2.})
    self.assertEqual(apply(init(1.)), 2.)

  def test_unzip_should_nest_and_unzip_jitted_functions(self):
    @jax.jit
    def nested(x):
      def foo(x):
        return variable(x + 1., name='x')

      init, _ = unzip_variable(foo)(x)
      result = init(x)

      @jax.jit
      def bar(x, result):
        return variable(result['x'], name='x') + variable(x + 1., name='x2')

      init, _ = unzip_variable(bar)(x, result)
      r = variable(init(x), name='r')
      return r
    self.assertDictEqual(nested(1.), {'x2': 2.})
    init, apply = unzip_variable(nested)(1.)
    self.assertDictEqual(init(1.), {'r': {'x2': 2}})
    self.assertEqual(apply(init(1.)), {'x2': 2.})

  def test_should_unzip_pmap(self):
    @jax.pmap
    def f(x):
      x = variable(x, name='x')
      return x
    onp.testing.assert_allclose(f(np.ones(2)), np.ones(2))
    init, apply = unzip_variable(f)(np.ones(2))
    onp.testing.assert_allclose(init(np.ones(2))['x'], np.ones(2))
    onp.testing.assert_allclose(apply(init(np.ones(2))), np.ones(2))

  def test_unzip_of_nest_should_nest_variables(self):
    def f(x):
      x = variable(x, name='x')
      return x
    init, apply = unzip_variable(harvest.nest(f, scope='f'))(1.)
    self.assertDictEqual(init(1.), {'f': {'x': 1}})
    self.assertEqual(apply({'f': {'x': 2.}}), 2.)

    def g(x):
      y = harvest.nest(f, scope='f1')(x + 1.)
      z = harvest.nest(f, scope='f2')(y + 1.)
      return z
    init, apply = unzip_variable(g)(1.)
    self.assertDictEqual(init(1.), {'f1': {'x': 2}, 'f2': {'x': 3.}})
    self.assertEqual(apply({'f1': {'x': 4.}, 'f2': {'x': 100.}}), 100.)


if __name__ == '__main__':
  os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=2'
  absltest.main()
