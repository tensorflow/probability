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
"""Tests for tensorflow_probability.spinoffs.oryx.experimental.nn.function."""

from absl.testing import absltest
import jax
from jax import random
from jax import tree_util
import jax.numpy as jnp
import numpy as np

from oryx.core import state
from oryx.experimental import nn
from oryx.internal import test_util


class AddOne(nn.Layer):

  @classmethod
  def initialize(cls, rng, in_spec):
    return nn.LayerParams()

  @classmethod
  def spec(cls, in_spec):
    return in_spec

  def _call(self, x):
    return x + 1


class ScalarMul(nn.Layer):

  @classmethod
  def initialize(cls, rng, in_spec, weight):
    weight = jnp.array(weight)
    return nn.LayerParams(params=weight)

  @classmethod
  def spec(cls, in_spec, weight):
    return in_spec

  def _call(self, x):
    return self.params * x


class Counter(nn.Layer):

  @classmethod
  def initialize(cls, rng, in_spec, initial_state):
    return nn.LayerParams(state=initial_state)

  @classmethod
  def spec(cls, in_spec, initial_state):
    return in_spec

  def _call(self, x):
    return self.state + x

  def _update(self, x):
    return self.replace(state=self.state + 1)


class IsTraining(nn.Layer):

  @classmethod
  def initialize(cls, rng, in_spec):
    return nn.LayerParams()

  @classmethod
  def spec(cls, in_spec):
    return in_spec

  def _call(self, x, training=True):
    if training:
      return jnp.ones_like(x)
    else:
      return jnp.zeros_like(x)


class Sampler(nn.Layer):

  @classmethod
  def initialize(cls, rng, in_spec):
    return nn.LayerParams()

  @classmethod
  def spec(cls, in_spec):
    return in_spec

  def _call(self, x, rng=None):
    assert rng is not None, 'Layer needs valid RNG'
    return random.normal(rng, x.shape)


class FunctionTest(test_util.TestCase):

  def setUp(self):
    super().setUp()
    self._seed = random.PRNGKey(0)

  def test_identity(self):
    def identity(x):
      return x
    in_spec = state.Shape(50)
    out_spec = state.spec(identity)(in_spec)
    self.assertEqual(out_spec, in_spec)

    net = state.init(identity)(self._seed, jnp.ones(50))
    np.testing.assert_array_equal(net(jnp.arange(5)), jnp.arange(5))

  def test_add(self):
    def add(x):
      return jnp.add(x, x)
    in_spec = state.Shape(50)
    out_spec = state.spec(add)(in_spec)
    self.assertEqual(out_spec, in_spec)

    in_spec = state.Shape((100, 20))
    out_spec = state.spec(add)(in_spec)
    self.assertEqual(out_spec, in_spec)

    in_spec = state.Shape(2)
    net = state.init(add)(self._seed, jnp.ones(2))
    np.testing.assert_array_equal(net(jnp.arange(5)), 2 * jnp.arange(5))

  def test_multiple_input_single_output(self):
    def add(x, y):
      return jnp.add(x, y)

    in_spec = (state.Shape((100, 20)), state.Shape(50))
    with self.assertRaises(ValueError):
      out_spec = state.spec(add)(*in_spec)

    in_spec = (state.Shape(50), state.Shape(50))
    out_spec = state.spec(add)(*in_spec)
    self.assertEqual(out_spec, in_spec[0])

    net = state.init(add)(self._seed, jnp.ones(50), jnp.ones(50))
    np.testing.assert_array_equal(net(jnp.arange(5), jnp.arange(5)),
                                  2 * jnp.arange(5))

  def test_single_input_multiple_output(self):
    def dup(x):
      return x, x

    in_spec = state.Shape(50)
    out_spec = state.spec(dup)(in_spec)
    self.assertEqual(out_spec, (in_spec, in_spec))

    net = state.init(dup)(self._seed, jnp.ones(50))
    for x1, x2 in zip(net(jnp.arange(5)),
                      (jnp.arange(5), jnp.arange(5))):
      np.testing.assert_array_equal(x1, x2)

  def test_multiple_input_multiple_output(self):
    def swap(x, y):
      return y, x

    in_spec = (state.Shape(50), state.Shape(20))
    out_spec = state.spec(swap)(*in_spec)
    self.assertEqual(out_spec, (in_spec[1], in_spec[0]))

    net = state.init(swap)(self._seed, jnp.ones(50), jnp.ones(20))
    for x1, x2 in zip(net(jnp.zeros(50), jnp.ones(20)),
                      (jnp.ones(20), jnp.zeros(50))):
      np.testing.assert_array_equal(x1, x2)

  def test_nested_add(self):
    def add(x):
      return (lambda x: jnp.add(x, x))(x)
    in_spec = state.Shape(50)
    out_spec = state.spec(add)(in_spec)
    self.assertEqual(out_spec, in_spec)

    in_spec = state.Shape((100, 20))
    out_spec = state.spec(add)(in_spec)
    self.assertEqual(out_spec, in_spec)

    in_spec = state.Shape(2)
    net = state.init(add)(self._seed, jnp.ones(2))
    np.testing.assert_array_equal(net(jnp.arange(5)), 2 * jnp.arange(5))

  def test_dense_function(self):
    def dense_no_rng(x):
      return nn.Dense(20)(x, name='dense')

    with self.assertRaises(ValueError):
      out_spec = state.spec(dense_no_rng)(state.Shape(50))

    with self.assertRaises(ValueError):
      net = state.init(dense_no_rng)(self._seed, state.Shape(2))

    def dense(x, init_key=None):
      return nn.Dense(20)(x, init_key=init_key, name='dense')

    out_spec = state.spec(dense)(state.Shape(2))
    self.assertEqual(out_spec, state.Shape(20))

    net = state.init(dense)(self._seed, jnp.ones(2))
    self.assertTupleEqual(net(jnp.ones(2)).shape, (20,))
    np.testing.assert_allclose(net(jnp.ones(2)),
                               dense(jnp.ones(2), init_key=self._seed),
                               rtol=1e-5)

  def test_dense_combinator(self):
    def dense(x, init_key=None):
      return (nn.Dense(50) >> nn.Dense(20))(x, init_key=init_key, name='dense')
    in_spec = state.Shape(50)
    out_spec = state.spec(dense)(in_spec)
    self.assertEqual(out_spec, state.Shape(20, dtype=in_spec.dtype))

    net = state.init(dense)(self._seed, jnp.ones(2))
    self.assertTupleEqual(net(jnp.ones(2)).shape, out_spec.shape)
    np.testing.assert_allclose(
        dense(jnp.ones(2), init_key=self._seed),
        net(jnp.ones(2)), rtol=1e-5)

  def test_add_one_combinator(self):
    def add_two(x, init_key=None):
      return (AddOne() >> AddOne())(x, name='add_one', init_key=init_key)

    in_spec = state.Shape(20)
    out_spec = state.spec(add_two)(in_spec)
    self.assertEqual(out_spec, in_spec)

    in_spec = state.Shape((5, 50))
    out_spec = state.spec(add_two)(in_spec)
    self.assertEqual(out_spec, in_spec)

    net = state.init(add_two)(self._seed, jnp.ones(2))
    np.testing.assert_allclose(
        net(jnp.ones(2)),
        3 * jnp.ones(2))
    np.testing.assert_array_equal(
        net(jnp.ones(2)),
        add_two(jnp.ones(2), init_key=self._seed))

  def test_add_one_imperative(self):
    def add_two(x, init_key=None):
      k1, k2 = random.split(init_key)
      x = AddOne()(x, name='add_one_1', init_key=k1)
      x = AddOne()(x, name='add_one_2', init_key=k2)
      return x
    in_spec = state.Shape(20)
    out_spec = state.spec(add_two)(in_spec)
    self.assertEqual(out_spec, in_spec)

    in_spec = state.Shape((5, 50))
    out_spec = state.spec(add_two)(in_spec)
    self.assertEqual(out_spec, in_spec)

    net = state.init(add_two)(self._seed, jnp.ones(2))
    np.testing.assert_array_equal(
        net(jnp.ones(2)),
        3 * jnp.ones(2))
    np.testing.assert_array_equal(
        net(jnp.ones(2)),
        add_two(jnp.ones(2), init_key=self._seed))

  def test_dense_imperative(self):
    def dense(x, init_key=None):
      key, subkey = random.split(init_key)
      x = nn.Dense(50)(x, init_key=key, name='dense1')
      x = nn.Dense(20)(x, init_key=subkey, name='dense2')
      return x
    in_spec = state.Shape(50)
    out_spec = state.spec(dense)(in_spec)
    self.assertEqual(out_spec, state.Shape(20, dtype=in_spec.dtype))

    net = state.init(dense)(self._seed, jnp.ones(2))
    self.assertTupleEqual(net(jnp.ones(2)).shape, out_spec.shape)
    np.testing.assert_allclose(
        dense(jnp.ones(2), init_key=self._seed),
        net(jnp.ones(2)), rtol=1e-5)

  def test_function_in_combinator(self):
    def add_one(x):
      return x + 1

    template = AddOne() >> add_one >> AddOne()

    net = state.init(template)(self._seed, jnp.ones(2))
    np.testing.assert_array_equal(net(jnp.zeros(2)), 3 * jnp.ones(2))

  def test_function_in_combinator_in_function(self):
    def add_one(x):
      return x + 1
    def template(x, init_key=None):
      return (AddOne() >> add_one >> AddOne())(x, init_key=init_key)

    net = state.init(template)(self._seed, jnp.ones(2))
    np.testing.assert_array_equal(net(jnp.zeros(2), init_key=self._seed),
                                  3 * jnp.ones(2))

  def test_grad_of_function_with_literal(self):
    def template(x, init_key=None):
      # 1.0 behaves like a literal when tracing
      return ScalarMul(1.0)(x, init_key=init_key, name='scalar_mul')
    net = state.init(template)(self._seed, jnp.ones(5))
    def loss(net, x):
      return net(x).sum()
    g = jax.grad(loss)(net, jnp.ones(5))
    def add(x, y):
      return x + y
    net = tree_util.tree_multimap(add, net, g)
    # w_new = w_old + 5
    np.testing.assert_array_equal(net(jnp.ones(5)), 6 * jnp.ones(5))

  def test_grad_of_function_constant(self):
    def template(x):
      return x + jnp.ones_like(x)
    net = state.init(template)(self._seed, jnp.ones(5))
    def loss(net, x):
      return net(x).sum()
    g = jax.grad(loss)(net, jnp.ones(5))
    def add(x, y):
      return x + y
    net = tree_util.tree_multimap(add, net, g)
    # w_new = w_old + 5
    np.testing.assert_array_equal(net(jnp.ones(5)), 2 * jnp.ones(5))

  def test_grad_of_function(self):
    def template(x, init_key=None):
      # jnp.ones(1) does not behave like a literal when tracing
      return ScalarMul(jnp.ones(1))(x, init_key=init_key, name='scalar_mul')
    net = state.init(template)(self._seed, jnp.ones(5))
    def loss(net, x):
      return net(x).sum()
    g = jax.grad(loss)(net, jnp.ones(5))
    def add(x, y):
      return x + y
    net = tree_util.tree_multimap(add, net, g)
    # w_new = w_old + 5
    np.testing.assert_array_equal(net(jnp.ones(5)), 6 * jnp.ones(5))

  def test_grad_of_stateful_function(self):
    def template(x, init_key=None):
      x = ScalarMul(jnp.ones(1))(x, init_key=init_key, name='scalar_mul')
      x = Counter(jnp.zeros(1))(x, init_key=init_key, name='counter')
      return x
    net = state.init(template)(self._seed, jnp.ones(5))
    def loss(net, x):
      return net(x).sum()
    g = jax.grad(loss)(net, jnp.ones(5))
    def add(x, y):
      return x + y
    net = tree_util.tree_multimap(add, net, g)
    # w_new = w_old + 5
    np.testing.assert_array_equal(net(jnp.ones(5)), 6 * jnp.ones(5))
    net = net.update(jnp.ones(5))
    np.testing.assert_array_equal(net(jnp.ones(5)), 7 * jnp.ones(5))

    g = jax.grad(loss)(net, jnp.ones(5))
    net = tree_util.tree_multimap(add, net, g)
    # w_new = w_old + 5
    np.testing.assert_array_equal(net(jnp.ones(5)), 12 * jnp.ones(5))

  def test_shared_layer(self):
    def template(x, init_key=None):
      layer = state.init(ScalarMul(2 * jnp.ones(1)), name='scalar_mul')(
          init_key, x)
      return layer(layer(x))
    net = state.init(template)(self._seed, jnp.ones(5))
    np.testing.assert_array_equal(net(jnp.ones(5)), 4 * jnp.ones(5))

  def test_grad_of_shared_layer(self):
    def template(x, init_key=None):
      layer = state.init(ScalarMul(2 * jnp.ones(1)), name='scalar_mul')(
          init_key, x)
      return layer(layer(x)).sum()
    net = state.init(template)(self._seed, jnp.ones(()))

    def loss(net, x):
      return net(x)
    g = jax.grad(loss)(net, jnp.ones(()))
    def add(x, y):
      return x + y
    net = tree_util.tree_multimap(add, net, g)
    np.testing.assert_array_equal(net(jnp.ones(())), 36.)

  def test_update(self):
    def template(x, init_key=None):
      return Counter(jnp.zeros(()))(x, init_key=init_key, name='counter')
    net = state.init(template)(self._seed, jnp.ones(()))
    self.assertEqual(net(jnp.ones(())), 1.)

    net2 = state.update(net, jnp.ones(()))
    self.assertEqual(net2(jnp.ones(())), 2.)

    net2 = net.update(jnp.ones(()))
    self.assertEqual(net2(jnp.ones(())), 2.)

  def test_update_in_combinator(self):
    def template(x, init_key=None):
      def increment(x, init_key=None):
        return Counter(jnp.zeros(()))(x, init_key=init_key, name='counter')
      return nn.Serial([increment, increment])(x, init_key=init_key,
                                               name='increment')
    net = state.init(template)(self._seed, jnp.ones(()))
    self.assertEqual(net(jnp.ones(())), 1.)
    net = state.update(net, jnp.ones(()))
    self.assertEqual(net(jnp.ones(())), 3.)

  def test_kwargs_training(self):
    def template(x, training=False, init_key=None):
      return IsTraining()(x, name='training', training=training,
                          init_key=init_key)
    net = state.init(template)(self._seed, jnp.ones(()))
    self.assertEqual(net(jnp.ones(()), training=True), 1.)
    self.assertEqual(net(jnp.ones(()), training=False), 0.)

    def template1(x, training=True, init_key=None):
      return IsTraining()(x, training=training, name='training',
                          init_key=init_key)
    net = state.init(template1)(self._seed, jnp.ones(()))
    self.assertEqual(net(jnp.ones(()), training=True), 1.)
    self.assertEqual(net(jnp.ones(()), training=False), 0.)

    def template2(x, init_key=None):
      return IsTraining()(x, name='training', init_key=init_key) + 1
    net = state.init(template2)(self._seed, jnp.ones(()))
    self.assertEqual(net(jnp.ones(()), training=True), 2.)
    self.assertEqual(net(jnp.ones(()), training=False), 1.)

  def test_kwargs_rng(self):
    def template(x, init_key=None):
      return Sampler()(x, name='sampler', init_key=init_key)
    with self.assertRaises(AssertionError):
      net = state.init(template)(self._seed, jnp.ones(()))
    def template1(x, rng, init_key=None):
      return Sampler()(x, rng=rng, init_key=init_key)
    net = state.init(template1)(self._seed, jnp.ones(()),
                                jnp.ones(2, dtype=jnp.uint32))
    x1 = net(jnp.ones(()), random.PRNGKey(0))
    x2 = net(jnp.ones(()), random.PRNGKey(1))
    self.assertNotEqual(x1, x2)

  def test_kwargs_training_rng(self):
    def template(x, rng, training=True, init_key=None):
      k1, k2 = random.split(init_key)
      x = Sampler()(x, rng=rng, name='sampler', init_key=k1)
      return (IsTraining()(x, training=training, name='training', init_key=k2)
              + x)

    net = state.init(template)(
        self._seed, jnp.ones(()), random.PRNGKey(0))
    x0n = net(jnp.ones(()), random.PRNGKey(0), training=False)
    x0t = net(jnp.ones(()), random.PRNGKey(0), training=True)
    x1n = net(jnp.ones(()), random.PRNGKey(1), training=False)
    x1t = net(jnp.ones(()), random.PRNGKey(1), training=True)
    # Different seeds generate different results
    # Same seed generates offset based on training flag
    self.assertNotEqual(x0n, x1n)
    self.assertNotEqual(x0t, x1t)
    np.testing.assert_allclose(x0n, x0t - 1, rtol=1e-6)
    np.testing.assert_allclose(x1n, x1t - 1, rtol=1e-6)

  def test_call_tuple(self):
    def template(x, init_key=None):
      return state.call((Counter(0.), AddOne()), x, init_key=init_key,
                        name='counter_add_one')
    layer = state.init(template)(self._seed, jnp.ones(()))
    self.assertTupleEqual(layer(jnp.zeros(())), (0, 1))

    layer = layer.update(jnp.zeros(()))
    self.assertTupleEqual(layer(jnp.zeros(())), (1, 1))

  def test_call_list(self):
    def template(x, init_key=None):
      return state.call([Counter(0.), AddOne()], x, init_key=init_key,
                        name='counter_add_one')
    layer = state.init(template)(self._seed, jnp.ones(()))
    self.assertEqual(layer(jnp.zeros(())), 1)

    layer = layer.update(jnp.zeros(()))
    self.assertEqual(layer(jnp.zeros(())), 2)

  def test_duplicate_names(self):
    def template(x, init_key=None):
      k1, k2 = random.split(init_key)
      layer1 = state.init(nn.Dense(20), name='dense')(k1, x)
      layer2 = state.init(nn.Dense(20), name='dense')(k2, x)
      return layer1(x) + layer2(x)
    with self.assertRaises(ValueError):
      state.init(template)(self._seed, jnp.ones(5))


if __name__ == '__main__':
  absltest.main()
