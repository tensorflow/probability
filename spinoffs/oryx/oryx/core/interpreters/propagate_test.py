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
"""Tests for tensorflow_probability.spinoffs.oryx.core.interpreters.propagate."""
from absl.testing import absltest
import jax
from jax import lax
import jax.numpy as np
import numpy as onp

from oryx.core import trace_util
from oryx.core.interpreters.propagate import Cell
from oryx.core.interpreters.propagate import propagate
from oryx.internal import test_util

inverse_rules = {}


class Inverse(Cell):

  def __init__(self, aval, val):
    super().__init__(aval)
    self.val = val

  def __lt__(self, other):
    return self.bottom() and other.top()

  def top(self):
    return self.val is not None

  def bottom(self):
    return self.val is None

  def join(self, other):
    if other.bottom():
      return self
    else:
      return other

  @classmethod
  def new(cls, val):
    aval = trace_util.get_shaped_aval(val)
    return Inverse(aval, val)

  @classmethod
  def unknown(cls, aval):
    return Inverse(aval, None)

  def flatten(self):
    return (self.val,), (self.aval,)

  @classmethod
  def unflatten(cls, data, xs):
    return Inverse(data[0], xs[0])


def exp_rule(invals, outvals):
  outval, = outvals
  inval, = invals
  if inval.bottom() and not outval.bottom():
    invals = [Inverse.new(np.log(outval.val))]
  elif outval.bottom() and not inval.bottom():
    outvals = [Inverse.new(np.exp(inval.val))]
  return invals, outvals, None


inverse_rules[lax.exp_p] = exp_rule


def add_rule(invals, outvals):
  outval, = outvals
  left, right = invals
  if not outval.bottom():
    if not left.bottom():
      invals = [left, Inverse.new(outval.val - left.val)]
    elif not right.bottom():
      invals = [Inverse.new(outval.val - right.val), right]
  elif outval.bottom() and not left.bottom() and not right.bottom():
    outvals = [Inverse.new(left.val + right.val)]
  return invals, outvals, None


inverse_rules[lax.add_p] = add_rule

ildj_rules = {}


class ILDJ(Cell):

  def __init__(self, aval, val, ildj):
    super().__init__(aval)
    self.val = val
    self.ildj = ildj

  def __lt__(self, other):
    return self.bottom() and other.top()

  def top(self):
    return self.val is not None

  def bottom(self):
    return self.val is None

  def join(self, other):
    if other.bottom():
      return self
    else:
      return other

  @classmethod
  def new(cls, val):
    aval = trace_util.get_shaped_aval(val)
    return ILDJ(aval, val, 0.)

  @classmethod
  def unknown(cls, aval):
    return ILDJ(aval, None, 0.)

  def flatten(self):
    return (self.val, self.ildj), (self.aval,)

  @classmethod
  def unflatten(cls, data, xs):
    return ILDJ(data[0], xs[0], xs[1])


def exp_ildj(invals, outvals):
  inval, = invals
  outval, = outvals
  if not inval.top() and outval.top():
    val, ildj = outval.val, outval.ildj
    invals = [ILDJ(inval.aval, np.log(val), ildj - np.log(val))]
  elif not outval.top() and inval.top():
    val, ildj = inval.val, inval.ildj
    outvals = [ILDJ(outval.aval, np.exp(val), ildj)]
  return invals, outvals, 1


ildj_rules[lax.exp_p] = exp_ildj


def add_ildj(invals, outvals):
  outval, = outvals
  left, right = invals
  if outval.top():
    val, ildj = outval.val, outval.ildj
    if left.top():
      invals = [left, ILDJ(right.aval, val - left.val, ildj)]
    elif right.top():
      invals = [ILDJ(left.aval, val - right.val, ildj), right]
  elif not outval.top() and left.top() and right.top():
    outvals = [ILDJ(outval.aval, left.val + right.val, 0.)]
  return invals, outvals, 1


ildj_rules[lax.add_p] = add_ildj


class PropagateTest(test_util.TestCase):

  def test_correct_inverse_for_identity_function(self):

    def f(x):
      return x

    jaxpr, _ = trace_util.stage(f)(1.)
    jaxpr, consts = jaxpr.jaxpr, jaxpr.literals
    env, _ = propagate(Inverse, inverse_rules, jaxpr,
                       list(map(Inverse.new, consts)),
                       [Inverse.unknown(var.aval) for var in jaxpr.invars],
                       list(map(Inverse.new, (1.,))))
    inval = env[jaxpr.invars[0]]
    self.assertEqual(inval.val, 1.)

  def test_should_propagate_to_invars_for_one_op_function(self):

    def f(x):
      return np.exp(x)

    jaxpr, _ = trace_util.stage(f)(1.)
    jaxpr, consts = jaxpr.jaxpr, jaxpr.literals
    env, _ = propagate(Inverse, inverse_rules, jaxpr,
                       list(map(Inverse.new, consts)),
                       [Inverse.unknown(var.aval) for var in jaxpr.invars],
                       list(map(Inverse.new, (1.,))))
    inval = env[jaxpr.invars[0]]
    self.assertEqual(inval.val, 0.)

  def test_should_propagate_to_invars_for_chain_function(self):

    def f(x):
      return 2. + np.exp(x)

    jaxpr, _ = trace_util.stage(f)(3.)
    jaxpr, consts = jaxpr.jaxpr, jaxpr.literals
    env, _ = propagate(Inverse, inverse_rules, jaxpr,
                       list(map(Inverse.new, consts)),
                       [Inverse.unknown(var.aval) for var in jaxpr.invars],
                       list(map(Inverse.new, (3.,))))
    inval = env[jaxpr.invars[0]]
    self.assertEqual(inval.val, 0.)

  def test_propagate_through_jit(self):

    def f(x):
      return jax.jit(np.exp)(x) + 2.

    jaxpr, _ = trace_util.stage(f)(3.)
    jaxpr, consts = jaxpr.jaxpr, jaxpr.literals
    env, _ = propagate(Inverse, inverse_rules, jaxpr,
                       list(map(Inverse.new, consts)),
                       [Inverse.unknown(var.aval) for var in jaxpr.invars],
                       list(map(Inverse.new, (3.,))))
    inval = env[jaxpr.invars[0]]
    self.assertEqual(inval.val, 0.)

  def test_propagation_should_not_reach_invars(self):

    def f(x):
      del x
      return 2.

    jaxpr, _ = trace_util.stage(f)(1.)
    jaxpr, consts = jaxpr.jaxpr, jaxpr.literals
    env, _ = propagate(Inverse, inverse_rules, jaxpr,
                       list(map(Inverse.new, consts)),
                       [Inverse.unknown(var.aval) for var in jaxpr.invars],
                       list(map(Inverse.new, (1.,))))
    self.assertTrue(env.read(jaxpr.invars[0]).bottom())

  def test_should_propagate_forward_and_backward(self):

    def f(x, y):
      return x + 1., np.exp(x + 1.) + y

    jaxpr, _ = trace_util.stage(f)(0., 2.)
    jaxpr, consts = jaxpr.jaxpr, jaxpr.literals
    env, _ = propagate(Inverse, inverse_rules, jaxpr,
                       list(map(Inverse.new, consts)),
                       [Inverse.unknown(var.aval) for var in jaxpr.invars],
                       list(map(Inverse.new, (0., 2.))))
    invals = [env[invar].val for invar in jaxpr.invars]
    onp.testing.assert_allclose(invals, (-1., 1.))

  def test_should_propagate_accumulated_values_in_one_op_function(self):

    def f(x):
      return np.exp(x)

    jaxpr, _ = trace_util.stage(f)(2.)
    jaxpr, consts = jaxpr.jaxpr, jaxpr.literals
    env, _ = propagate(ILDJ, ildj_rules, jaxpr, list(map(ILDJ.new, consts)),
                       [Inverse.unknown(var.aval) for var in jaxpr.invars],
                       list(map(ILDJ.new, (2.,))))
    inval = env[jaxpr.invars[0]]
    self.assertEqual(inval.ildj, -np.log(2.))

  def test_should_propagate_accumulated_values_in_chain_function(self):

    def f(x):
      return np.exp(x) + 2.

    jaxpr, _ = trace_util.stage(f)(4.)
    jaxpr, consts = jaxpr.jaxpr, jaxpr.literals
    env, _ = propagate(ILDJ, ildj_rules, jaxpr, list(map(ILDJ.new, consts)),
                       [Inverse.unknown(var.aval) for var in jaxpr.invars],
                       list(map(ILDJ.new, (4.,))))
    inval = env[jaxpr.invars[0]]
    self.assertEqual(inval.ildj, -np.log(2.))

  def test_propagate_should_accumulate_state(self):

    def f(x):
      return np.exp(x) + 2.

    jaxpr, _ = trace_util.stage(f)(4.)
    jaxpr, consts = jaxpr.jaxpr, jaxpr.literals
    _, state = propagate(
        ILDJ,
        ildj_rules,
        jaxpr,
        list(map(ILDJ.new, consts)),
        [Inverse.unknown(var.aval) for var in jaxpr.invars],
        list(map(ILDJ.new, (4.,))),
        reducer=lambda env, eqn, count, next: count + next,
        initial_state=0)
    self.assertEqual(state, 2)


if __name__ == '__main__':
  absltest.main()
