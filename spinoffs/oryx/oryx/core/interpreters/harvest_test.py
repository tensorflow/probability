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
"""Tests for tensorflow_probability.spinoffs.oryx.core.interpreters.harvest."""
import functools
import os

from absl.testing import absltest
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np

from oryx.core import trace_util
from oryx.core.interpreters import harvest
from oryx.internal import test_util

sow = harvest.sow
reap = harvest.reap
call_and_reap = harvest.call_and_reap
plant = harvest.plant
nest = harvest.nest

variable = functools.partial(sow, tag='variable')
harvest_variables = functools.partial(harvest.harvest, tag='variable')
plant_variables = functools.partial(plant, tag='variable')
reap_variables = functools.partial(reap, tag='variable')
call_and_reap_variables = functools.partial(call_and_reap, tag='variable')


class ReapTest(absltest.TestCase):

  def test_should_reap(self):

    def f(x):
      return variable(x, name='x')

    self.assertDictEqual(reap_variables(f)(1.), {'x': 1.})

  def test_reap_should_ignore_blocklisted_variables(self):

    def f(x):
      return variable(x, name='x')

    self.assertDictEqual(reap_variables(f, blocklist=['x'])(1.), {})

  def test_reap_should_only_reap_allowlisted_variables(self):

    def f(x):
      return variable(x, name='x') + variable(x, name='y')

    self.assertDictEqual(reap_variables(f, allowlist=['x'])(1.), {'x': 1.})

  def test_should_error_in_strict_mode(self):

    def f(x):
      x = variable(x, name='x', mode='strict')
      y = variable(x + 1., name='x', mode='strict')
      return y

    with self.assertRaisesRegex(ValueError,
                                'Variable has already been reaped: x'):
      reap_variables(f)(1.)

  def test_should_reap_multiple(self):

    def f(x):
      x = variable(x, name='x')
      y = variable(x + 1., name='y')
      return y

    self.assertDictEqual(reap_variables(f)(1.), {'x': 1., 'y': 2.})

  def test_should_reap_unreturned_variable(self):

    def f(x):
      x = variable(x, name='x')
      y = variable(x + 1., name='y')
      variable(x + 2., name='z')
      return y

    self.assertDictEqual(reap_variables(f)(1.), {'x': 1., 'y': 2., 'z': 3.})

  def test_should_reap_jitted_function(self):

    @jax.jit
    def f(x):
      return variable(x, name='x')

    self.assertDictEqual(reap_variables(f)(1.), {'x': 1.})

  def test_should_reap_unreturned_variables_in_jitted_function(self):

    @jax.jit
    def f(x):
      variable(x, name='x')

    self.assertDictEqual(reap_variables(f)(1.), {'x': 1.})

  def test_should_reap_pmapped_function(self):

    @jax.pmap
    def f(x):
      return variable(x, name='x')

    np.testing.assert_allclose(reap_variables(f)(jnp.ones(2))['x'], jnp.ones(2))

  def test_reap_should_remove_sow_primitives(self):

    def f(x):
      return variable(x, name='x')

    jaxpr = jax.make_jaxpr(reap_variables(f))(1.)
    primitives = set(eqn.primitive for eqn in jaxpr.jaxpr.eqns)
    self.assertNotIn(harvest.sow_p, primitives)

  def test_reap_should_handle_closed_over_values(self):

    def foo(x):

      @jax.jit
      def bar(_):
        return variable(x, name='x')

      return bar(1.0)

    self.assertTupleEqual(harvest_variables(foo)({}, 1.), (1., {'x': 1.}))

  def test_should_reap_constant(self):

    def f(x):
      y = variable(10., name='y')
      return x + y

    self.assertDictEqual(reap_variables(f)(1.), {'y': 10.})


class PlantTest(test_util.TestCase):

  def test_should_plant_variable(self):

    def f(x):
      return variable(x, name='x')

    self.assertEqual(plant_variables(f)({}, 1.), 1.)
    self.assertEqual(plant_variables(f)({'x': 2.}, 1.), 2.)

  def test_should_not_plant_variables_in_blocklist(self):

    def f(x):
      return variable(x, name='x')

    self.assertEqual(plant_variables(f, blocklist=['x'])({'x': 2.}, 1.), 1.)

  def test_should_not_plant_variables_not_in_allowlist(self):

    def f(x):
      return variable(x, name='x')

    self.assertEqual(plant_variables(f, allowlist=['x'])({'x': 2.}, 1.), 2.)

  def test_should_plant_variables_in_allowlist(self):

    def f(x):
      return variable(x, name='x')

    self.assertEqual(plant_variables(f, allowlist=['y'])({'x': 2.}, 1.), 1.)

  def test_should_plant_multiple_variables(self):

    def f(x):
      return variable(x, name='x') + variable(x, name='y')

    self.assertEqual(plant_variables(f)({}, 1.), 2.)
    self.assertEqual(plant_variables(f)({'x': 2.}, 1.), 3.)
    self.assertEqual(plant_variables(f)({'x': 2., 'y': 2.}, 1.), 4.)

  def test_should_plant_in_jit(self):

    def f(x):
      return jax.jit(lambda x: variable(x, name='x'))(x)

    self.assertEqual(plant_variables(f)({}, 1.), 1.)
    self.assertEqual(plant_variables(f)({'x': 2.}, 1.), 2.)

  def test_jit_should_remain_in_plant(self):

    def f(x):
      return jax.jit(lambda x: variable(x, name='x') + 1.)(x)

    jaxpr = jax.make_jaxpr(plant_variables(f))({'x': 2.}, 1.)
    primitives = set(eqn.primitive for eqn in jaxpr.jaxpr.eqns)
    self.assertIn(jax.xla.xla_call_p, primitives)

  def test_should_plant_in_pmap(self):

    def f(x):
      return jax.pmap(lambda x: variable(x, name='x'))(x)

    np.testing.assert_allclose(plant_variables(f)({}, jnp.ones(2)), jnp.ones(2))
    np.testing.assert_allclose(
        plant_variables(f)({
            'x': 2 * jnp.ones(2)
        }, jnp.ones(2)), 2 * jnp.ones(2))

  def test_plant_should_handle_closed_over_values(self):

    def foo(x):

      @jax.jit
      def bar(_):
        return variable(x, name='x')

      return bar(1.0)

    self.assertTupleEqual(harvest_variables(foo)({'x': 2.}, 1.), (2., {}))

  def test_should_plant_constant(self):

    def f(x):
      y = variable(717, name='y')
      x = variable(x, name='x')
      return x + y

    self.assertEqual(plant_variables(f)({}, 1.), 718.)
    self.assertEqual(plant_variables(f)({'x': 2., 'y': 15.}, 1.), 17.)


class HarvestTest(test_util.TestCase):

  def test_should_harvest_variable(self):

    def f(x):
      return variable(x, name='x')

    self.assertTupleEqual(harvest_variables(f)({}, 1.), (1., {'x': 1.}))
    self.assertEqual(harvest_variables(f)({'x': 2.}, 1.), (2., {}))

  def test_nest_scope(self):

    def f(x):
      return variable(x, name='x')

    self.assertTupleEqual(
        harvest_variables(nest(f, scope='foo'))({}, 1.), (1., {
            'foo': {
                'x': 1.
            }
        }))
    self.assertTupleEqual(
        harvest_variables(nest(f, scope='foo'))({
            'foo': {
                'x': 2.
            }
        }, 1.), (2., {}))

  def test_harvest_should_clean_up_context(self):

    def f(x):
      raise ValueError('Intentional error!')

    with self.assertRaisesRegex(ValueError, 'Intentional error!'):
      harvest_variables(f)({}, 1.)
    self.assertDictEqual(trace_util._thread_local_state.dynamic_contexts, {})

  def test_can_jit_compile_nest(self):

    def f(x):
      return variable(x, name='x')

    self.assertTupleEqual(
        harvest_variables(jax.jit(nest(f, scope='foo')))({}, 1.), (1., {
            'foo': {
                'x': 1.
            }
        }))


class ControlFlowTest(test_util.TestCase):

  def test_strict_mode_in_scan_should_error(self):

    def body(carry, x):
      x = variable(x + carry, name='x', mode='strict')
      return x, x

    def f(init):
      return lax.scan(body, init, jnp.arange(5.))

    with self.assertRaisesRegex(
        ValueError, 'Cannot use strict mode for \'x\' inside `scan`.'):
      reap_variables(f)(1.)

  def test_harvest_append_mode_in_scan_should_accumulate(self):

    def body(carry, x):
      x = variable(x + carry, name='x', mode='append')
      return x, x

    def f(init):
      return lax.scan(body, init, jnp.arange(5.))

    (carry, out), variables = harvest_variables(f)({}, 1.)
    true_out = jnp.array([1., 2., 4., 7., 11.])
    np.testing.assert_allclose(carry, 11.)
    np.testing.assert_allclose(out, true_out)
    self.assertListEqual(['x'], list(variables.keys()))
    np.testing.assert_allclose(variables['x'], true_out)

  def test_harvest_append_mode_in_nested_scan_and_cond_should_accumulate(self):

    def body(carry, x):
      def _t(x):
        return variable(x + carry, name='x', mode='append')
      def _f(x):
        return variable(x, name='x', mode='append')
      x = lax.cond(True, _t, _f, x)
      return x, x

    def f(init):
      return lax.scan(jax.jit(body), init, jnp.arange(5.))

    (carry, out), variables = harvest_variables(f)({}, 1.)
    true_out = jnp.array([1., 2., 4., 7., 11.])
    np.testing.assert_allclose(carry, 11.)
    np.testing.assert_allclose(out, true_out)
    self.assertListEqual(['x'], list(variables.keys()))
    np.testing.assert_allclose(variables['x'], true_out)

  def test_harvest_clobber_mode_in_scan_should_return_final_value(self):

    def body(carry, x):
      x = variable(x + carry, name='x', mode='clobber')
      return x, x

    def f(init):
      return lax.scan(body, init, jnp.arange(5.))

    (carry, out), variables = harvest_variables(f)({}, 1.)
    true_out = jnp.array([1., 2., 4., 7., 11.])
    np.testing.assert_allclose(carry, 11.)
    np.testing.assert_allclose(out, true_out)
    self.assertListEqual(['x'], list(variables.keys()))
    np.testing.assert_allclose(variables['x'], true_out[-1])

  def test_non_clobber_mode_in_while_loop_should_error_with_reap_and_plant(
      self):

    def cond(carry):
      i, _ = carry
      return i < 5

    def body(carry):
      i, val = carry
      val = variable(val, name='x', mode='strict')
      return (i + 1, val + 1.)

    def f(init):
      return lax.while_loop(cond, body, init)

    with self.assertRaisesRegex(
        ValueError,
        'Must use clobber mode for \'x\' inside of a `while_loop`.'):
      reap_variables(f)((0, 0.))

    with self.assertRaisesRegex(
        ValueError,
        'Must use clobber mode for \'x\' inside of a `while_loop`.'):
      plant_variables(f)(dict(x=4.), (0, 0.))

  def test_can_reap_final_values_from_while_loop(self):

    def cond(carry):
      i, _ = carry
      return i < 5

    def body(carry):
      i, val = carry
      val = variable(val, name='x', mode='clobber')
      return (i + 1, val + 1.)

    def f(init):
      return lax.while_loop(cond, body, init)

    reaps = reap_variables(f)((0, 0.))

    self.assertDictEqual(reaps, dict(x=4.))

  def test_can_plant_values_into_each_iteration_of_while_loop(self):

    def cond(carry):
      i, _, _ = carry
      return i < 5

    def body(carry):
      i, val, val2 = carry
      val = variable(val, name='x', mode='clobber')
      return (i + 1, val + 1., val + val2)

    def f(init):
      return lax.while_loop(cond, body, init)

    out = plant_variables(f)(dict(x=5.), (0, 0., 0.))

    self.assertTupleEqual(out, (5, 6., 25.))

  def test_can_reap_and_plant_values_into_while_loop(self):

    def cond(carry):
      i, _, _ = carry
      return i < 5

    def body(carry):
      i, val, val2 = carry
      val = variable(val, name='x', mode='clobber')
      val2 = variable(val2, name='y', mode='clobber')
      return (i + 1, val + 1., val + val2)

    def f(init):
      return lax.while_loop(cond, body, init)

    out, reaps = call_and_reap_variables(plant_variables(f))(dict(x=5.),
                                                             (0, 0., 0.))

    self.assertTupleEqual(out, (5, 6., 25.))
    self.assertDictEqual(reaps, dict(y=20.))

  def test_must_have_identical_sow_in_both_branches_of_cond(self):

    def f(pred, x):

      def true_fun(x):
        x = variable(x, name='x')
        return x + 1.

      def false_fun(x):
        x = variable(x, name='y')
        return x + 2.

      return lax.cond(pred, true_fun, false_fun, x)

    with self.assertRaisesRegex(ValueError, 'Missing sow in branch: \'x\''):
      reap_variables(f)(True, 1.)

    with self.assertRaisesRegex(ValueError, 'Missing sow in branch: \'x\''):
      plant_variables(f)({}, True, 1.)

    def f2(pred, x):

      def true_fun(x):
        x = variable(x, name='x')
        return x + 1.

      def false_fun(x):
        x = variable(x, name='x')
        x = variable(x, name='y')
        return x + 2.

      return lax.cond(pred, true_fun, false_fun, x)

    with self.assertRaisesRegex(
        ValueError, 'Mismatching number of `sow`s between branches.'):
      reap_variables(f2)(True, 1.)

    with self.assertRaisesRegex(
        ValueError, 'Mismatching number of `sow`s between branches.'):
      plant_variables(f2)({}, True, 1.)

    def f3(pred, x):

      def true_fun(x):
        x = variable(x, name='x')
        return x + 1.

      def false_fun(x):
        x = variable(jnp.array([x, x]), name='x')
        return jnp.sum(x)

      return lax.cond(pred, true_fun, false_fun, x)

    with self.assertRaisesRegex(
        ValueError, 'Mismatched shape between branches: \'x\'.'):
      reap_variables(f3)(True, 1.)

    with self.assertRaisesRegex(
        ValueError, 'Mismatched shape between branches: \'x\'.'):
      plant_variables(f3)({}, True, 1.)

  def test_can_reap_values_from_either_branch_of_cond(self):

    def f(pred, x):

      def true_fun(x):
        x = variable(x, name='x')
        return x + 2.

      def false_fun(x):
        x = variable(x + 2., name='x')
        return x + 3.

      return lax.cond(pred, true_fun, false_fun, x)

    out, reaps = call_and_reap_variables(f)(True, 1.)
    self.assertEqual(out, 3.)
    self.assertDictEqual(reaps, dict(x=1.))

    out, reaps = call_and_reap_variables(f)(False, 1.)
    self.assertEqual(out, 6.)
    self.assertDictEqual(reaps, dict(x=3.))

  def test_can_plant_values_into_either_branch_of_cond(self):

    def f(pred, x):

      def true_fun(x):
        x = variable(x, name='x')
        return x + 2.

      def false_fun(x):
        x = variable(x + 2., name='x')
        return x + 3.

      return lax.cond(pred, true_fun, false_fun, x)

    out = plant_variables(f)(dict(x=4.), True, 1.)
    self.assertEqual(out, 6.)

    out = plant_variables(f)(dict(x=4.), False, 1.)
    self.assertEqual(out, 7.)

  def test_can_reap_values_from_any_branch_in_switch(self):

    def f(index, x):

      def branch1(x):
        x = variable(x, name='x')
        return x + 2.

      def branch2(x):
        x = variable(x + 2., name='x')
        return x + 3.

      def branch3(x):
        x = variable(x + 3., name='x')
        return x + 4.
      return lax.switch(index, (branch1, branch2, branch3), x)

    out, reaps = call_and_reap_variables(f)(0, 1.)
    self.assertEqual(out, 3.)
    self.assertDictEqual(reaps, dict(x=1.))

    out, reaps = call_and_reap_variables(f)(1, 1.)
    self.assertEqual(out, 6.)
    self.assertDictEqual(reaps, dict(x=3.))

    out, reaps = call_and_reap_variables(f)(2, 1.)
    self.assertEqual(out, 8.)
    self.assertDictEqual(reaps, dict(x=4.))

  def test_can_plant_values_into_any_branch_in_switch(self):

    def f(index, x):

      def branch1(x):
        x = variable(x, name='x')
        return x + 2.

      def branch2(x):
        x = variable(x + 2., name='x')
        return x + 3.

      def branch3(x):
        x = variable(x + 3., name='x')
        return x + 4.
      return lax.switch(index, (branch1, branch2, branch3), x)

    out = plant_variables(f)(dict(x=4.), 0, 1.)
    self.assertEqual(out, 6.)

    out = plant_variables(f)(dict(x=4.), 1, 1.)
    self.assertEqual(out, 7.)

    out = plant_variables(f)(dict(x=4.), 2, 1.)
    self.assertEqual(out, 8.)


if __name__ == '__main__':
  os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=2'
  absltest.main()
