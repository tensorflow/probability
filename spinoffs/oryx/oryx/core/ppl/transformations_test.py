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
"""Tests for tensorflow_probability.spinoffs.oryx.core.ppl.transformations."""
from absl.testing import absltest
from jax import abstract_arrays
from jax import core as jax_core
from jax import random
import jax.numpy as np

from oryx.core.interpreters import log_prob as lp
from oryx.core.ppl import transformations
from oryx.internal import test_util

seed = random.PRNGKey
block = transformations.block
conditional = transformations.conditional
graph_replace = transformations.graph_replace
joint_log_prob = transformations.joint_log_prob
joint_sample = transformations.joint_sample
log_prob = transformations.log_prob
intervene = transformations.intervene
random_variable = transformations.random_variable

# Define a random normal primitive so we can register it with the `log_prob`
# transformation.
random_normal_p = jax_core.Primitive('random_normal')


def random_normal(key):
  return random_normal_p.bind(key)


def random_normal_impl(rng):
  return random.normal(rng)


def random_normal_abstract(_):
  return abstract_arrays.ShapedArray((), np.float32)


def random_normal_log_prob(x):
  return -0.5 * np.log(2 * np.pi) - 0.5 * x**2


def random_normal_log_prob_rule(incells, outcells):
  outcell, = outcells
  if not outcell.top():
    return incells, outcells, None
  x = outcell.val
  return incells, outcells, random_normal_log_prob(x)


random_normal_p.def_impl(random_normal_impl)
random_normal_p.def_abstract_eval(random_normal_abstract)
lp.log_prob_rules[random_normal_p] = random_normal_log_prob_rule
lp.log_prob_registry.add(random_normal_p)


class SampleTest(test_util.TestCase):

  def test_random_variable_should_tag_output_of_function(self):

    def model(key):
      return random_normal(key)

    self.assertDictEqual(
        joint_sample(random_variable(model, name='x'))(seed(0)),
        {'x': random.normal(seed(0))})

  def test_joint_sample_should_return_all_samples(self):

    def model(key):
      k1, k2 = random.split(key)
      z = random_variable(random_normal, name='z')(k1)
      return random_variable(lambda key: random_normal(key) + z, name='x')(k2)

    k1, k2 = random.split(seed(0))
    z = random.normal(k1)
    self.assertDictEqual(
        joint_sample(model)(seed(0)), {
            'z': z,
            'x': z + random.normal(k2)
        })

  def test_block_should_result_in_ignored_names_in_joint_sample(self):

    def model(key):
      k1, k2 = random.split(key)
      z = random_variable(random_normal, name='z')(k1)
      return random_variable(lambda key: random_normal(key) + z, name='x')(k2)

    k1, _ = random.split(seed(0))
    z = random.normal(k1)
    self.assertDictEqual(
        joint_sample(block(model, names=['x']))(seed(0)), {
            'z': z,
        })

  def test_block_composing_with_intervene(self):

    def program(key):
      k1, k2, k3 = random.split(key, 3)
      a = random_variable(random_normal, name='a')(k1)
      b = a + random_variable(random_normal, name='b')(k2)
      c = random_variable(b + random_normal(k3), name='c')
      return c

    program_without_a = block(program, names=['a'])
    c_given_b = intervene(program_without_a, b=10.)
    self.assertLess(abs(10 - c_given_b(random.PRNGKey(0))), 4.)

  def test_observing_latent_injects_value(self):

    def model(key):
      k1, k2 = random.split(key)
      z = random_variable(random_normal, name='z')(k1)
      return random_variable(lambda key: random_normal(key) + z, name='x')(k2)

    _, k2 = random.split(seed(0))
    self.assertEqual(intervene(model, x=1., z=1.)(seed(0)), 1.)
    self.assertEqual(intervene(model, x=1.)(seed(0)), 1.)
    self.assertEqual(intervene(model, z=1.)(seed(0)), 1. + random.normal(k2))

  def test_observing_nothing_returns_original_function(self):

    def model(key):
      k1, k2 = random.split(key)
      z = random_variable(random_normal, name='z')(k1)
      return random_variable(lambda key: random_normal(key) + z, name='x')(k2)

    k1, k2 = random.split(seed(0))
    self.assertDictEqual(
        joint_sample(intervene(model))(seed(0)), {
            'z': random.normal(k1),
            'x': random.normal(k1) + random.normal(k2)
        })

  def test_observing_joint_sample_removes_latent(self):

    def model(key):
      k1, k2 = random.split(key)
      z = random_variable(random_normal, name='z')(k1)
      return random_variable(lambda key: random_normal(key) + z, name='x')(k2)

    _, k2 = random.split(seed(0))
    self.assertDictEqual(
        joint_sample(intervene(model, z=1.))(seed(0)),
        {'x': 1. + random.normal(k2)})

  def test_conditioning_on_nothing_is_original_function(self):

    def model(key):
      k1, k2 = random.split(key)
      z = random_variable(random_normal, name='z')(k1)
      return random_variable(lambda key: random_normal(key) + z, name='x')(k2)

    k1, k2 = random.split(seed(0))
    self.assertDictEqual(
        joint_sample(conditional(model, []))(seed(0)), {
            'z': random.normal(k1),
            'x': random.normal(k1) + random.normal(k2)
        })

  def test_conditioning_injects_value(self):

    def model(key):
      k1, k2 = random.split(key)
      z = random_variable(random_normal, name='z')(k1)
      return random_variable(lambda key: random_normal(key) + z, name='x')(k2)

    _, k2 = random.split(seed(0))
    self.assertEqual(conditional(model, ['x', 'z'])(seed(0), 1., 1.), 1.)
    self.assertEqual(conditional(model, 'x')(seed(0), 1.), 1.)
    _, k2 = random.split(seed(0))
    self.assertEqual(
        conditional(model, 'z')(seed(0), 1.), 1. + random.normal(k2))

  def test_conditioning_removes_latent(self):

    def model(key):
      k1, k2 = random.split(key)
      z = random_variable(random_normal, name='z')(k1)
      return random_variable(lambda key: random_normal(key) + z, name='x')(k2)

    self.assertSetEqual(
        set(
            joint_sample(conditional(model, 'x'))(random.PRNGKey(0),
                                                  1.).keys()), {'z'})


class LogProbTest(test_util.TestCase):

  def test_log_prob_should_propagate_on_trivial_model(self):

    def model(key):
      return random_normal(key)

    self.assertEqual(log_prob(model)(0.1), random_normal_log_prob(0.1))

  def test_log_prob_should_fail_on_model_with_latents(self):

    def model(key):
      k1, k2 = random.split(key)
      z = random_variable(random_normal, name='z')(k1)
      return random_variable(lambda key: random_normal(key) + z, name='x')(k2)

    with self.assertRaisesRegex(ValueError,
                                'Cannot compute log_prob of function.'):
      log_prob(model)(0.1)

  def test_log_prob_should_work_with_joint_sample(self):

    def model(key):
      k1, k2 = random.split(key)
      z = random_variable(random_normal, name='z')(k1)
      return random_variable(lambda key: random_normal(key) + z, name='x')(k2)

    self.assertEqual(
        log_prob(joint_sample(model))({
            'z': 1.,
            'x': 1.
        }),
        random_normal_log_prob(1.) + random_normal_log_prob(0.))

    self.assertEqual(
        joint_log_prob(model)({
            'z': 1.,
            'x': 1.
        }),
        random_normal_log_prob(1.) + random_normal_log_prob(0.))

  def test_log_prob_should_work_with_nondependent_latents(self):

    @joint_sample
    def model(key):
      k1, k2 = random.split(key)
      z = random_variable(random_normal, name='z')(k1)
      return random_variable(lambda key: random_normal(key) + z, name='x')(k2)

    limited_model = lambda key: model(key)['z']

    self.assertEqual(log_prob(limited_model)(1.), random_normal_log_prob(1.))

  def test_intervened_log_prob(self):

    def model(key):
      k1, k2 = random.split(key)
      z = random_variable(random_normal, name='z')(k1)
      return random_variable(lambda key: random_normal(key) + z, name='x')(k2)

    intervened_model = intervene(model, z=1.)

    self.assertEqual(log_prob(intervened_model)(1.), random_normal_log_prob(0.))

  def test_conditional_log_prob(self):

    def model(key):
      k1, k2 = random.split(key)
      z = random_variable(random_normal, name='z')(k1)
      return random_variable(lambda key: random_normal(key) + z, name='x')(k2)

    conditioned_model = conditional(model, 'z')

    self.assertEqual(
        log_prob(conditioned_model)(1., 1.), random_normal_log_prob(0.))


class GraphReplaceTest(test_util.TestCase):

  def test_graph_replace_correctly_computes_outputs(self):

    def model(key):
      k1, k2, k3 = random.split(key, 3)
      z = random_variable(random_normal, name='z')(k1)
      x = random_variable(lambda key: random_normal(key) + z, name='x')(k2)
      y = random_variable(lambda key: random_normal(key) + x, name='y')(k3)
      return y

    k1, k2, k3 = random.split(random.PRNGKey(0), 3)

    x_to_y = graph_replace(model, 'x', 'y')
    self.assertEqual(x_to_y(random.PRNGKey(0), 1.), 1. + random.normal(k3))

    z_to_y = graph_replace(model, 'z', 'y')
    self.assertEqual(
        z_to_y(random.PRNGKey(0), 1.),
        1. + random.normal(k2) + random.normal(k3))

    z_to_xy = graph_replace(model, 'z', ['x', 'y'])
    self.assertListEqual(
        z_to_xy(random.PRNGKey(0), 1.),
        [1. + random.normal(k2), 1 + random.normal(k2) + random.normal(k3)])

    zx_to_y = graph_replace(model, ['z', 'x'], 'y')
    self.assertEqual(zx_to_y(random.PRNGKey(0), 1., 2.), 2. + random.normal(k3))

    to_y = graph_replace(model, [], 'y')
    self.assertEqual(
        to_y(random.PRNGKey(0)),
        random.normal(k1) + random.normal(k2) + random.normal(k3))

    y_to_z = graph_replace(model, 'y', 'z')
    self.assertEqual(y_to_z(random.PRNGKey(0), 1.), random.normal(k1))

  def test_graph_replace_has_correct_latents(self):

    def model(key):
      k1, k2, k3 = random.split(key, 3)
      z = random_variable(random_normal, name='z')(k1)
      x = random_variable(lambda key: random_normal(key) + z, name='x')(k2)
      y = random_variable(lambda key: random_normal(key) + x, name='y')(k3)
      return y

    k1, k2, _ = random.split(random.PRNGKey(0), 3)

    x_to_y = graph_replace(model, 'x', 'y')
    self.assertDictEqual(
        joint_sample(x_to_y)(random.PRNGKey(0), 1.), {'z': random.normal(k1)})

    z_to_y = graph_replace(model, 'z', 'y')
    self.assertDictEqual(
        joint_sample(z_to_y)(random.PRNGKey(0), 1.),
        {'x': 1 + random.normal(k2)})

    z_to_xy = graph_replace(model, 'z', ['x', 'y'])
    self.assertDictEqual(joint_sample(z_to_xy)(random.PRNGKey(0), 1.), {})

    zx_to_y = graph_replace(model, ['z', 'x'], 'y')
    self.assertDictEqual(joint_sample(zx_to_y)(random.PRNGKey(0), 1., 2.), {})

    to_y = graph_replace(model, [], 'y')
    self.assertDictEqual(
        joint_sample(to_y)(random.PRNGKey(0)), {
            'z': random.normal(k1),
            'x': random.normal(k1) + random.normal(k2)
        })


if __name__ == '__main__':
  absltest.main()
