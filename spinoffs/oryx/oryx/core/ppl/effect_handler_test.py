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
# Lint as: python3
"""Tests for tensorflow_probability.spinoffs.oryx.core.ppl.effect_handler."""
from absl.testing import absltest
import jax
from jax import abstract_arrays
from jax import random
import jax.numpy as np

from oryx.core import primitive
from oryx.core.ppl import effect_handler
from oryx.internal import test_util

# Define a random normal primitive so we can use it for custom interpreter
# rules.
random_normal_p = primitive.FlatPrimitive('random_normal')


def random_normal(key, loc=0., scale=1.):
  return random_normal_p.bind(key, loc, scale)[0]


@random_normal_p.def_impl
def _random_normal_impl(key, loc, scale):
  return [random.normal(key) * scale + loc]


@random_normal_p.def_abstract_eval
def _random_normal_abstract(key, loc, scale):
  del key, loc, scale
  return [abstract_arrays.ShapedArray((), np.float32)]


class EffectHandlerTest(test_util.TestCase):

  def test_effect_handler_with_no_rules_should_be_identity(self):

    def f(key):
      return random_normal(key)

    transformation = effect_handler.make_effect_handler({})
    f_out, state = transformation(f)(None, random.PRNGKey(0))
    self.assertIs(state, None)
    self.assertEqual(f_out, f(random.PRNGKey(0)))

  def test_effect_handler_can_override_primitive_behavior(self):

    def random_normal_deterministic_rule(state, key, *_):
      del key
      return [0.], state

    rules = {random_normal_p: random_normal_deterministic_rule}

    def f(key):
      return random_normal(key)

    make_deterministic = effect_handler.make_effect_handler(rules)
    deterministic_f = make_deterministic(f)

    f_out, state = deterministic_f(None, random.PRNGKey(0))
    self.assertIs(state, None)
    self.assertEqual(f_out, 0.)

  def test_effect_handler_correctly_updates_state(self):

    def random_normal_counter_rule(count, key, *_):
      return [random_normal(key)], count + 1

    rules = {random_normal_p: random_normal_counter_rule}

    def f(key):
      k1, k2 = random.split(key)
      return random_normal(k1) + random_normal(k2)

    make_counter = effect_handler.make_effect_handler(rules)
    counter_f = make_counter(f)

    count = 0
    f_out, count = counter_f(count, random.PRNGKey(0))
    self.assertEqual(count, 2)
    self.assertEqual(f_out, f(random.PRNGKey(0)))

    f_out, count = counter_f(count, random.PRNGKey(1))
    self.assertEqual(count, 4)
    self.assertEqual(f_out, f(random.PRNGKey(1)))

  def test_correctly_overrides_behavior_inside_call_primitive(self):

    def random_normal_deterministic_rule(state, key, *_):
      del key
      return [0.], state

    rules = {random_normal_p: random_normal_deterministic_rule}

    @jax.jit
    def f(key):
      return random_normal(key)

    make_deterministic = effect_handler.make_effect_handler(rules)
    deterministic_f = make_deterministic(f)

    f_out, state = deterministic_f(None, random.PRNGKey(0))
    self.assertIs(state, None)
    self.assertEqual(f_out, 0.)

  def test_correctly_updates_state_inside_call_primitive(self):

    def random_normal_counter_rule(count, key, *_):
      return [random_normal(key)], count + 1

    rules = {random_normal_p: random_normal_counter_rule}

    @jax.jit
    def f(key):
      k1, k2 = random.split(key)
      return random_normal(k1) + random_normal(k2)

    make_counter = effect_handler.make_effect_handler(rules)
    counter_f = make_counter(f)

    f_out, count = counter_f(0, random.PRNGKey(0))
    self.assertEqual(count, 2)
    self.assertEqual(f_out, f(random.PRNGKey(0)))

  def test_effect_handler_correctly_maintains_python_structures(self):

    def random_normal_counter_rule(count, key, *_):
      return [random_normal(key)], count + 1

    rules = {random_normal_p: random_normal_counter_rule}

    @jax.jit
    def f(key):
      k1, k2 = random.split(key)
      return dict(x=random_normal(k1), y=random_normal(k2))

    make_counter = effect_handler.make_effect_handler(rules)
    counter_f = make_counter(f)

    f_out, count = counter_f(0, random.PRNGKey(0))
    self.assertEqual(count, 2)
    self.assertEqual(f_out, f(random.PRNGKey(0)))

  def test_basic_noncentering_parameterization_behaves_correctly(self):

    def random_normal_noncentering_rule(state, key, loc, scale):
      return [random_normal(key) * scale + loc], state

    rules = {random_normal_p: random_normal_noncentering_rule}

    def f(key):
      return random_normal(key, 2., 1.)

    noncenter = effect_handler.make_effect_handler(rules)
    noncentered_f = noncenter(f)
    # Programs should be semantically identical
    self.assertEqual(
        f(random.PRNGKey(0)),
        noncentered_f(None, random.PRNGKey(0))[0])

    # We should be sampling from an isotropic normal in the noncentered variant.
    noncenter_jaxpr = jax.make_jaxpr(noncentered_f)(None, random.PRNGKey(0))
    for eqn in noncenter_jaxpr.jaxpr.eqns:
      if eqn.primitive is random_normal_p:
        loc = eqn.invars[1].val
        scale = eqn.invars[2].val
        self.assertEqual(loc, 0.)
        self.assertEqual(scale, 1.)


if __name__ == '__main__':
  absltest.main()
