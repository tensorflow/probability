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
from jax.interpreters import xla
import jax.numpy as np
import numpy as onp

from oryx.core import trace_util
from oryx.core.interpreters.propagate import Cell
from oryx.core.interpreters.propagate import propagate
from oryx.core.interpreters.propagate import unknown


inverse_rules = {}


class Inverse(Cell):

  def __init__(self, val):
    self.val = val

  @classmethod
  def new(cls, val):
    return Inverse(val)


def exp_rule(invals, outvals):
  outval, = outvals
  inval, = invals
  done = False
  if inval.is_unknown() and not outval.is_unknown():
    invals = [Inverse(np.log(outval.val))]
    done = True
  elif outval.is_unknown() and not inval.is_unknown():
    outvals = [Inverse(np.exp(inval.val))]
    done = True
  return invals, outvals, done, None
inverse_rules[lax.exp_p] = exp_rule


def add_rule(invals, outvals):
  outval, = outvals
  left, right = invals
  done = False
  if not outval.is_unknown():
    if left is not unknown:
      invals = [left, Inverse(outval.val - left.val)]
      done = True
    elif not right.is_unknown():
      invals = [Inverse(outval.val - right.val), right]
      done = True
  elif outval.is_unknown() and not left.is_unknown() and not right.is_unknown():
    outvals = [Inverse(left.val + right.val)]
    done = True
  return invals, outvals, done, None
inverse_rules[lax.add_p] = add_rule


def xla_call_rule(invals, outvals, **params):
  del params
  f, invals = invals[0], invals[1:]
  subenv = f.call_wrapped(invals, outvals)
  new_invals = [subenv.read(invar) for invar in subenv.jaxpr.invars]
  new_outvals = [subenv.read(outvar) for outvar in subenv.jaxpr.outvars]
  done = all(not val.is_unknown() for val in new_invals + new_outvals)
  return new_invals, new_outvals, done, subenv
inverse_rules[xla.xla_call_p] = xla_call_rule


ildj_rules = {}


class ILDJ(Cell):

  def __init__(self, val, ildj):
    self.val = val
    self.ildj = ildj

  @classmethod
  def new(cls, val):
    return ILDJ(val, 0.)


def exp_ildj(invals, outvals):
  inval, = invals
  outval, = outvals
  done = False
  if inval.is_unknown() and not outval.is_unknown():
    val, ildj = outval.val, outval.ildj
    invals = [ILDJ(np.log(val), ildj - np.log(val))]
    done = True
  elif outval.is_unknown() and not inval.is_unknown():
    val, ildj = inval.val, inval.ildj
    outvals = [ILDJ(np.exp(val), ildj)]
    done = True
  return invals, outvals, done, None
ildj_rules[lax.exp_p] = exp_ildj


def add_ildj(invals, outvals):
  outval, = outvals
  left, right = invals
  done = False
  if not outval.is_unknown():
    val, ildj = outval.val, outval.ildj
    if not left.is_unknown():
      invals = [left, ILDJ(val - left.val, ildj)]
      done = True
    elif not right.is_unknown():
      invals = [ILDJ(val - right.val, ildj), right]
      done = True
  elif outval.is_unknown() and not left.is_unknown() and not right.is_unknown():
    outvals = [ILDJ(left.val + right.val, 0.)]
    done = True
  return invals, outvals, done, None
ildj_rules[lax.add_p] = add_ildj


class PropagateTest(absltest.TestCase):

  def test_correct_inverse_for_identity_function(self):
    def f(x):
      return x

    jaxpr, _ = trace_util.stage(f)(1.)
    jaxpr, consts = jaxpr.jaxpr, jaxpr.literals
    env = propagate(Inverse, inverse_rules, jaxpr,
                    list(map(Inverse.new, consts)),
                    [unknown] * len(jaxpr.invars),
                    list(map(Inverse.new, (1.,))))
    inval = env[jaxpr.invars[0]]
    self.assertEqual(inval.val, 1.)

  def test_should_propagate_to_invars_for_one_op_function(self):
    def f(x):
      return np.exp(x)

    jaxpr, _ = trace_util.stage(f)(1.)
    jaxpr, consts = jaxpr.jaxpr, jaxpr.literals
    env = propagate(Inverse, inverse_rules, jaxpr,
                    list(map(Inverse.new, consts)),
                    [unknown] * len(jaxpr.invars),
                    list(map(Inverse.new, (1.,))))
    inval = env[jaxpr.invars[0]]
    self.assertEqual(inval.val, 0.)

  def test_should_propagate_to_invars_for_chain_function(self):
    def f(x):
      return 2. + np.exp(x)

    jaxpr, _ = trace_util.stage(f)(3.)
    jaxpr, consts = jaxpr.jaxpr, jaxpr.literals
    env = propagate(Inverse, inverse_rules, jaxpr,
                    list(map(Inverse.new, consts)),
                    [unknown] * len(jaxpr.invars),
                    list(map(Inverse.new, (3.,))))
    inval = env[jaxpr.invars[0]]
    self.assertEqual(inval.val, 0.)

  def test_propagate_through_jit(self):
    def f(x):
      return jax.jit(np.exp)(x) + 2.

    jaxpr, _ = trace_util.stage(f)(3.)
    jaxpr, consts = jaxpr.jaxpr, jaxpr.literals
    env = propagate(Inverse, inverse_rules, jaxpr,
                    list(map(Inverse.new, consts)),
                    [unknown] * len(jaxpr.invars),
                    list(map(Inverse.new, (3.,))))
    inval = env[jaxpr.invars[0]]
    self.assertEqual(inval.val, 0.)
    self.assertLen(env.subenvs, 1)

  def test_propagation_should_not_reach_invars(self):
    def f(x):
      del x
      return 2.

    jaxpr, _ = trace_util.stage(f)(1.)
    jaxpr, consts = jaxpr.jaxpr, jaxpr.literals
    env = propagate(Inverse, inverse_rules, jaxpr,
                    list(map(Inverse.new, consts)),

                    [unknown] * len(jaxpr.invars),
                    list(map(Inverse.new, (1.,))))
    self.assertNotIn(jaxpr.invars[0], env)

  def test_should_propagate_forward_and_backward(self):
    def f(x, y):
      return x + 1., np.exp(x + 1.) + y

    jaxpr, _ = trace_util.stage(f)(0., 2.)
    jaxpr, consts = jaxpr.jaxpr, jaxpr.literals
    env = propagate(Inverse, inverse_rules, jaxpr,
                    list(map(Inverse.new, consts)),
                    [unknown] * len(jaxpr.invars),
                    list(map(Inverse.new, (0., 2.))))
    invals = [env[invar].val for invar in jaxpr.invars]
    onp.testing.assert_allclose(invals, (-1., 1.))

  def test_should_propagate_accumulated_values_in_one_op_function(self):
    def f(x):
      return np.exp(x)

    jaxpr, _ = trace_util.stage(f)(2.)
    jaxpr, consts = jaxpr.jaxpr, jaxpr.literals
    env = propagate(ILDJ, ildj_rules, jaxpr,
                    list(map(ILDJ.new, consts)),
                    [unknown] * len(jaxpr.invars),
                    list(map(ILDJ.new, (2.,))))
    inval = env[jaxpr.invars[0]]
    self.assertEqual(inval.ildj, -np.log(2.))

  def test_should_propagate_accumulated_values_in_chain_function(self):
    def f(x):
      return np.exp(x) + 2.

    jaxpr, _ = trace_util.stage(f)(4.)
    jaxpr, consts = jaxpr.jaxpr, jaxpr.literals
    env = propagate(ILDJ, ildj_rules, jaxpr,
                    list(map(ILDJ.new, consts)),
                    [unknown] * len(jaxpr.invars),
                    list(map(ILDJ.new, (4.,))))
    inval = env[jaxpr.invars[0]]
    self.assertEqual(inval.ildj, -np.log(2.))


if __name__ == '__main__':
  absltest.main()
