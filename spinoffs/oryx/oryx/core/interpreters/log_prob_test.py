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
"""Tests for tensorflow_probability.spinoffs.oryx.core.interpreters.log_prob."""
from absl.testing import absltest
import jax
from jax import abstract_arrays
from jax import core as jax_core
from jax import linear_util as lu
from jax import random
import jax.numpy as np

from oryx import bijectors as bb
from oryx import core
from oryx import distributions as bd
from oryx.core import state
from oryx.core.interpreters.log_prob import log_prob
from oryx.core.interpreters.log_prob import log_prob_registry
from oryx.core.interpreters.log_prob import log_prob_rules
from oryx.internal import test_util

random_normal_p = jax_core.Primitive('random_normal')


def random_normal(rng, name=None):
  return state.variable(random_normal_p.bind(rng, name=name), name=name)


def random_normal_impl(rng, name=None):
  del name
  return random.normal(rng)
random_normal_p.def_impl(random_normal_impl)


def random_normal_abstract(_, name=None):
  del name
  return abstract_arrays.ShapedArray((), np.float32)
random_normal_p.def_abstract_eval(random_normal_abstract)


def random_normal_log_prob_rule(incells, outcells, **_):
  outcell, = outcells
  if not outcell.top():
    return incells, outcells, None
  outval = outcell.val
  return incells, outcells, bd.Normal(0., 1.).log_prob(outval)
log_prob_rules[random_normal_p] = random_normal_log_prob_rule
log_prob_registry.add(random_normal_p)


def call(f):
  def wrapped(*args, **kwargs):
    fun = lu.wrap_init(f, kwargs)
    flat_args, in_tree = jax.tree_flatten(args)
    flat_fun, out_tree = jax.flatten_fun_nokwargs(fun, in_tree)
    ans = jax_core.call_p.bind(flat_fun, *flat_args)
    return jax.tree_unflatten(out_tree(), ans)
  return wrapped
jax_core.call_p.call_primitive = True


class LogProbTest(test_util.TestCase):

  def test_normal_log_prob(self):
    def f(rng):
      return random_normal(rng)
    f_lp = log_prob(f)
    self.assertEqual(f_lp(0.), bd.Normal(0., 1.).log_prob(0.))
    self.assertEqual(f_lp(1.), bd.Normal(0., 1.).log_prob(1.))

  def test_log_normal_log_prob(self):
    def f(rng):
      return np.exp(random_normal(rng))
    dist = bd.TransformedDistribution(
        bd.Normal(0., 1.), bb.Exp())
    f_lp = log_prob(f)
    self.assertEqual(f_lp(2.), dist.log_prob(2.))

  def test_multiple_sample(self):
    def f(rng):
      k1, k2 = random.split(rng)
      return random_normal(k1) + random_normal(k2)
    f_lp = log_prob(f)
    with self.assertRaises(ValueError):
      f_lp(0.1)

  def test_latent_variable(self):
    def f(rng):
      k1, k2 = random.split(rng)
      z = random_normal(k1)
      return random_normal(k2) + z
    f_lp = log_prob(f)
    with self.assertRaises(ValueError):
      f_lp(0.1)

  def test_conditional_log(self):
    def f(rng, x):
      return random_normal(rng) + x
    f_lp = log_prob(f)
    self.assertEqual(
        f_lp(0.1, 1.0),
        bd.Normal(0., 1.).log_prob(-0.9))

  def test_unzip(self):
    def f(rng):
      k1, k2 = random.split(rng)
      z = random_normal(k1, name='z')
      return random_normal(k2, name='x') + z
    init, _ = core.unzip(f, tag=state.VARIABLE)(random.PRNGKey(0))
    f_lp = log_prob(init)
    f_lp(init(random.PRNGKey(0)))

  def test_log_prob_in_call(self):
    def f(rng):
      z = call(lambda k: random_normal(k, name='z'))(rng)
      return z
    f_lp = log_prob(f)
    s = f(random.PRNGKey(0))
    self.assertEqual(
        f_lp(s),
        bd.Normal(0., 1.).log_prob(s))


if __name__ == '__main__':
  absltest.main()
