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
"""Tests for tensorflow_probability.spinoffs.oryx.core.state.function."""

from absl.testing import absltest
import jax
from jax import core as jax_core
from jax import random
import jax.numpy as np
import numpy as onp

from oryx.core import primitive
from oryx.core.state import api
from oryx.core.state import function
from oryx.core.state import module
from oryx.internal import test_util

training_add_p = jax_core.Primitive('training_add')


def _training_add_impl(a, **_):
  return a + 1.


def _training_abstract_eval(a, **_):
  return a


def _training_add_kwargs_rule(a, *, kwargs):
  if kwargs.get('training', True):
    return a + 1.
  return a


training_add_p.def_impl(_training_add_impl)
training_add_p.def_abstract_eval(_training_abstract_eval)
function.kwargs_rules[training_add_p] = _training_add_kwargs_rule


class FunctionModuleTest(test_util.TestCase):

  def test_init_nonstateful_function(self):

    def f(x):
      return x

    m = api.init(f)(random.PRNGKey(0), 1.)
    self.assertIsInstance(m, function.FunctionModule)
    self.assertDictEqual(m.variables(), {})
    self.assertEqual(m.call(1.), 1.)
    self.assertEqual(m.update(1.).variables(), {})

  def test_init_stateful_function(self):

    def f(x, init_key=None):
      y = module.variable(np.ones(x.shape), name='y', key=init_key)
      return x + y

    m = api.init(f)(random.PRNGKey(0), 1.)
    self.assertIsInstance(m, function.FunctionModule)
    self.assertDictEqual(m.variables(), {'y': 1.})
    self.assertEqual(m.call(1.), 2.)
    self.assertEqual(m.update(1.).variables(), {'y': 1})

  def test_init_stateful_function_with_assign(self):

    def f(x, init_key=None):
      y = module.variable(np.zeros(x.shape), name='y', key=init_key)
      next_y = module.assign(y + 1., name='y')
      return x + next_y

    m = api.init(f)(random.PRNGKey(0), 1.)
    self.assertIsInstance(m, function.FunctionModule)
    self.assertDictEqual(m.variables(), {'y': 0.})
    self.assertEqual(m.call(1.), 2.)
    self.assertDictEqual(m.update(1.).variables(), {'y': 1.})

  def test_assign_with_no_matching_variable_should_error(self):

    def f(x, init_key=None):
      y = module.variable(np.zeros(x.shape), name='y', key=init_key)
      next_y = module.assign(y + 1., name='z')
      return x + next_y

    m = api.init(f)(random.PRNGKey(0), 1.)
    with self.assertRaisesRegex(
        ValueError, 'No variable declared for assign: z'):
      m(1.)

  def test_init_stateful_function_with_tied_in_assign(self):

    def f(x, init_key=None):
      y = module.variable(np.zeros(x.shape), name='y', key=init_key)
      next_y = module.assign(y + 1., name='y')
      return primitive.tie_in(next_y, x) + y

    m = api.init(f)(random.PRNGKey(0), 1.)
    self.assertIsInstance(m, function.FunctionModule)
    self.assertDictEqual(m.variables(), {'y': 0.})
    self.assertEqual(m.call(1.), 1.)
    self.assertDictEqual(m.update(1.).variables(), {'y': 1.})

  def test_init_of_composed_stateful_functions_should_have_flat_params(self):

    def f(x, init_key=None):
      y = module.variable(np.zeros(x.shape), name='y', key=init_key)
      next_y = module.assign(y + 1., name='y')
      return primitive.tie_in(next_y, x) + y

    def g(x, init_key=None):
      return f(x, init_key=init_key)

    m = api.init(g)(random.PRNGKey(0), 1.)
    self.assertIsInstance(m, function.FunctionModule)
    self.assertDictEqual(m.variables(), {'y': 0.})
    self.assertEqual(m.call(1.), 1.)
    self.assertDictEqual(m.update(1.).variables(), {'y': 1.})

  def test_init_of_nested_init_without_name_should_have_flat_params(self):

    def f(x, init_key=None):
      y = module.variable(np.zeros(x.shape), name='y', key=init_key)
      next_y = module.assign(y + 1., name='y')
      return primitive.tie_in(next_y, x) + y

    def g(x, init_key=None):
      return api.init(f)(init_key, x)(x)

    m = api.init(g)(random.PRNGKey(0), 1.)
    self.assertIsInstance(m, function.FunctionModule)
    self.assertDictEqual(m.variables(), {'y': 0.})
    self.assertEqual(m.call(1.), 1.)
    self.assertDictEqual(m.update(1.).variables(), {'y': 1.})

  def test_init_of_nested_init_with_name_should_have_nested_params(self):

    def f(x, init_key=None):
      y = module.variable(np.zeros(x.shape), name='y', key=init_key)
      next_y = module.assign(y + 1., name='y')
      return primitive.tie_in(next_y, x) + y

    def g(x, init_key=None):
      return api.init(f, name='f')(init_key, x)(x)

    m = api.init(g)(random.PRNGKey(0), 1.)
    self.assertIsInstance(m, function.FunctionModule)
    self.assertIsInstance(m.f, function.FunctionModule)
    self.assertDictEqual(m.f.variables(), {'y': 0.})
    self.assertEqual(m.call(1.), 1.)
    self.assertDictEqual(m.update(1.).f.variables(), {'y': 1.})

  def test_should_pass_kwarg_into_primitive(self):

    def f(x):
      return training_add_p.bind(x)

    m = api.init(f)(random.PRNGKey(0), 1.)
    self.assertEqual(m(1.), 2.)
    self.assertEqual(m(1., training=True), 2.)
    self.assertEqual(m(1., training=False), 1.)


class FunctionSpecTest(test_util.TestCase):

  def test_spec_works_for_identity_function(self):
    def f(x):
      return x
    out_spec = api.spec(f)(np.ones(5))
    self.assertTupleEqual(out_spec.shape, (5,))
    self.assertEqual(out_spec.dtype, np.float32)


class VmapModuleTest(absltest.TestCase):

  def test_vmap_of_init_should_return_ensemble(self):
    def f(x, init_key=None):
      w = module.variable(random.normal(init_key, x.shape), name='w')
      return np.dot(w, x)
    ensemble = jax.vmap(api.init(f))(
        random.split(random.PRNGKey(0)),
        np.ones([2, 5]))
    self.assertTupleEqual(ensemble.w.shape, (2, 5))
    onp.testing.assert_allclose(
        jax.vmap(api.call, in_axes=(0, None))(ensemble, np.ones(5)),
        jax.vmap(lambda key, x: f(x, init_key=key), in_axes=(0, None))(
            random.split(random.PRNGKey(0)), np.ones(5)),
        rtol=1e-5, atol=1e-5)
    onp.testing.assert_allclose(
        jax.vmap(api.call, in_axes=(0, 0))(
            ensemble, np.arange(10.).reshape((2, 5))),
        jax.vmap(lambda key, x: f(x, init_key=key), in_axes=(0, 0))(
            random.split(random.PRNGKey(0)), np.arange(10.).reshape((2, 5))),
        rtol=1e-5, atol=1e-5)

  def test_init_of_vmap_should_return_ensemble(self):
    def f(x, init_key=None):
      w = module.variable(random.normal(init_key, x.shape), name='w')
      return np.dot(w, x)
    def f_(init_key, x):
      return api.init(f)(init_key, x)
    ensemble = jax.vmap(f_, in_axes=(0, 0))(
        random.split(random.PRNGKey(0)),
        np.ones([2, 5]))
    self.assertTupleEqual(ensemble.w.shape, (2, 5))
    onp.testing.assert_allclose(
        jax.vmap(api.call, in_axes=(0, None))(ensemble, np.ones(5)),
        jax.vmap(lambda key, x: f(x, init_key=key), in_axes=(0, None))(
            random.split(random.PRNGKey(0)), np.ones(5)),
        rtol=1e-5, atol=1e-5)
    onp.testing.assert_allclose(
        jax.vmap(api.call, in_axes=(0, 0))(
            ensemble, np.arange(10.).reshape((2, 5))),
        jax.vmap(lambda key, x: f(x, init_key=key), in_axes=(0, 0))(
            random.split(random.PRNGKey(0)), np.arange(10.).reshape((2, 5))),
        rtol=1e-5, atol=1e-5)


if __name__ == '__main__':
  absltest.main()
