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
"""Tests for tensorflow_probability.spinoffs.oryx.core.ppl.transformations."""
from absl.testing import absltest

import jax
from jax import abstract_arrays
from jax import core as jax_core
from jax import random
from jax.interpreters import batching
import jax.numpy as jnp
import numpy as np

from oryx.core.interpreters import log_prob as lp
from oryx.core.ppl import transformations
from oryx.internal import test_util
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions

seed = random.PRNGKey
block = transformations.block
conditional = transformations.conditional
graph_replace = transformations.graph_replace
intervene = transformations.intervene
joint_log_prob = transformations.joint_log_prob
joint_sample = transformations.joint_sample
log_prob = transformations.log_prob
plate = transformations.plate
random_variable = transformations.random_variable
trace = transformations.trace
trace_log_prob = transformations.trace_log_prob

# Define a random normal primitive so we can register it with the `log_prob`
# transformation.
random_normal_p = jax_core.Primitive('random_normal')


def random_normal(key, batch_ndims=0):
  return random_normal_p.bind(key, batch_ndims=batch_ndims)


def random_normal_impl(rng, *, batch_ndims):
  sample = random.normal
  for _ in range(batch_ndims):
    sample = jax.vmap(sample)
  return sample(rng)


def random_normal_abstract(key, **_):
  del key
  return abstract_arrays.ShapedArray((), jnp.float32)


def random_normal_log_prob_rule(incells, outcells, *, batch_ndims, **_):
  outcell, = outcells
  if not outcell.top():
    return incells, outcells, None
  x = outcell.val
  return incells, outcells, tfd.Independent(tfd.Normal(0., 1.),
                                            batch_ndims).log_prob(x)


def random_normal_batch_rule(args, _, *, batch_ndims):
  keys, = args
  out = random_normal_p.bind(keys, batch_ndims=batch_ndims + 1)
  return out, 0


batching.primitive_batchers[random_normal_p] = random_normal_batch_rule

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

  def test_trace_should_return_output_and_latents(self):

    def model(key):
      k1, k2 = random.split(key)
      z = random_variable(random_normal, name='z')(k1)
      return random_variable(lambda key: random_normal(key) + z)(k2)

    k1, k2 = random.split(seed(0))
    z = random.normal(k1)
    output, latents = trace(model)(seed(0))
    self.assertEqual(output, z + random.normal(k2))
    self.assertDictEqual(latents, dict(z=z))

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

    self.assertEqual(log_prob(model)(0.1), tfd.Normal(0., 1.).log_prob(0.1))

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
        tfd.Normal(0., 1.).log_prob(1.) + tfd.Normal(0., 1.).log_prob(0.))

    self.assertEqual(
        joint_log_prob(model)({
            'z': 1.,
            'x': 1.
        }),
        tfd.Normal(0., 1.).log_prob(1.) + tfd.Normal(0., 1.).log_prob(0.))

  def test_log_prob_should_work_with_trace(self):

    def model(key):
      k1, k2 = random.split(key)
      z = random_variable(random_normal, name='z')(k1)
      return random_variable(lambda key: random_normal(key) + z)(k2)

    log_prob_value = (
        tfd.Normal(0., 1.).log_prob(1.)
        + tfd.Normal(0., 1.).log_prob(0.))

    self.assertEqual(log_prob(trace(model))((1., dict(z=1.))), log_prob_value)

    self.assertEqual(trace_log_prob(model)(1., dict(z=1.)), log_prob_value)

  def test_log_prob_should_work_with_nondependent_latents(self):

    @joint_sample
    def model(key):
      k1, k2 = random.split(key)
      z = random_variable(random_normal, name='z')(k1)
      return random_variable(lambda key: random_normal(key) + z, name='x')(k2)

    limited_model = lambda key: model(key)['z']

    self.assertEqual(
        log_prob(limited_model)(1.),
        tfd.Normal(0., 1.).log_prob(1.))

  def test_intervened_log_prob(self):

    def model(key):
      k1, k2 = random.split(key)
      z = random_variable(random_normal, name='z')(k1)
      return random_variable(lambda key: random_normal(key) + z, name='x')(k2)

    intervened_model = intervene(model, z=1.)

    self.assertEqual(
        log_prob(intervened_model)(1.),
        tfd.Normal(0., 1.).log_prob(0.))

  def test_conditional_log_prob(self):

    def model(key):
      k1, k2 = random.split(key)
      z = random_variable(random_normal, name='z')(k1)
      return random_variable(lambda key: random_normal(key) + z, name='x')(k2)

    conditioned_model = conditional(model, 'z')

    self.assertEqual(
        log_prob(conditioned_model)(1., 1.),
        tfd.Normal(0., 1.).log_prob(0.))


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

  def test_plate_should_result_in_different_samples(self):

    @plate(name='foo')
    def model(key):
      return random_variable(random_normal)(key)

    out = jax.vmap(
        lambda _, key: model(key), in_axes=(0, None),
        axis_name='foo')(jnp.ones(3), random.PRNGKey(0))
    for i in range(3):
      for j in range(3):
        if i == j:
          continue
        self.assertNotAlmostEqual(out[i], out[j])

  def test_nested_plates_should_produce_multiple_axes(self):

    @plate(name='foo')
    @plate(name='bar')
    def model(key):
      return random_variable(random_normal)(key)

    out = jax.vmap(
        lambda _, key: jax.vmap(  # pylint: disable=g-long-lambda,unnecessary-lambda
            lambda _, key: model(key),  # pylint: disable=unnecessary-lambda
            in_axes=(0, None),
            axis_name='bar')(_, key),
        in_axes=(0, None),
        axis_name='foo')(jnp.ones((3, 2)), random.PRNGKey(0))
    for i in range(6):
      for j in range(6):
        if i == j:
          continue
        i1, i2 = i // 2, i - 2 * (i // 2)
        j1, j2 = j // 2, j - 2 * (j // 2)
        self.assertNotAlmostEqual(out[i1, i2], out[j1, j2])

  def test_plate_should_reduce_over_log_prob_named_axis(self):

    @plate(name='foo')
    def model(key):
      return random_variable(random_normal)(key)

    out = jax.vmap(
        log_prob(model), axis_name='foo', out_axes=None)(
            jnp.arange(3.))
    np.testing.assert_allclose(
        tfd.Normal(0., 1.).log_prob(jnp.arange(3.)).sum(), out)

  def test_nested_plates_should_reduce_over_all_axes(self):

    @plate(name='foo')
    @plate(name='bar')
    def model(key):
      return random_variable(random_normal)(key)

    out = jax.vmap(
        jax.vmap(log_prob(model), axis_name='bar', out_axes=None),
        axis_name='foo',
        out_axes=None)(
            jnp.arange(6.).reshape((3, 2)))
    np.testing.assert_allclose(
        tfd.Normal(0., 1.).log_prob(jnp.arange(6.)).sum(),
        out,
        rtol=1e-6,
        atol=1e-5)


if __name__ == '__main__':
  absltest.main()
