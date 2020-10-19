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
"""Tests for tensorflow_probability.spinoffs.oryx.core.interpreters.harvest."""
import functools
import os

from absl.testing import absltest
import jax
from jax import lax
import jax.numpy as np
import numpy as onp

from oryx.core import trace_util
from oryx.core.interpreters import harvest
from oryx.internal import test_util


sow = harvest.sow
reap = harvest.reap
plant = harvest.plant
nest = harvest.nest

variable = functools.partial(sow, tag='variable')
harvest_variables = functools.partial(harvest.harvest, tag='variable')
plant_variables = functools.partial(plant, tag='variable')
reap_variables = functools.partial(reap, tag='variable')


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

  def test_should_append_in_append_mode(self):

    def f(x):
      x = variable(x, name='x', mode='append')
      y = variable(x + 1., name='x', mode='append')
      return y

    variables = reap_variables(f)(1.)
    self.assertTupleEqual(variables['x'].shape, (2,))
    onp.testing.assert_allclose(variables['x'], onp.array([1., 2.]))

  def test_should_clobber_in_clobber_mode(self):

    def f(x):
      x = variable(x, name='x', mode='clobber')
      y = variable(x + 1., name='x', mode='clobber')
      return y

    variables = reap_variables(f)(1.)
    self.assertEqual(variables['x'], 2)

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

    onp.testing.assert_allclose(reap_variables(f)(np.ones(2))['x'], np.ones(2))

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

    onp.testing.assert_allclose(plant_variables(f)({}, np.ones(2)), np.ones(2))
    onp.testing.assert_allclose(
        plant_variables(f)({
            'x': 2 * np.ones(2)
        }, np.ones(2)), 2 * np.ones(2))

  def test_plant_should_handle_closed_over_values(self):
    def foo(x):
      @jax.jit
      def bar(_):
        return variable(x, name='x')
      return bar(1.0)
    self.assertTupleEqual(harvest_variables(foo)({'x': 2.}, 1.), (2., {}))


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


class ControlFlowTest(test_util.TestCase):

  def test_strict_mode_in_scan_should_error(self):

    def body(carry, x):
      x = variable(x + carry, name='x', mode='strict')
      return x, x

    def f(init):
      return lax.scan(body, init, np.arange(5.))

    with self.assertRaisesRegex(ValueError,
                                'Cannot use strict mode in a scan.'):
      reap_variables(f)(1.)

  def test_harvest_append_mode_in_scan_should_accumulate(self):

    def body(carry, x):
      x = variable(x + carry, name='x', mode='append')
      return x, x

    def f(init):
      return lax.scan(body, init, np.arange(5.))

    (carry, out), variables = harvest_variables(f)({}, 1.)
    true_out = np.array([1., 2., 4., 7., 11.])
    onp.testing.assert_allclose(carry, 11.)
    onp.testing.assert_allclose(out, true_out)
    self.assertListEqual(['x'], list(variables.keys()))
    onp.testing.assert_allclose(variables['x'], true_out)

  def test_harvest_clobber_mode_in_scan_should_return_final_value(self):

    def body(carry, x):
      x = variable(x + carry, name='x', mode='clobber')
      return x, x

    def f(init):
      return lax.scan(body, init, np.arange(5.))

    (carry, out), variables = harvest_variables(f)({}, 1.)
    true_out = np.array([1., 2., 4., 7., 11.])
    onp.testing.assert_allclose(carry, 11.)
    onp.testing.assert_allclose(out, true_out)
    self.assertListEqual(['x'], list(variables.keys()))
    onp.testing.assert_allclose(variables['x'], true_out[-1])


if __name__ == '__main__':
  os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=2'
  absltest.main()
