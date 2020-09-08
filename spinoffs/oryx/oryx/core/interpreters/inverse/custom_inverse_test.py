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
"""Tests for tensorflow_probability.spinoffs.oryx.core.interpreters.inverse.custom_inverse."""
import functools

from absl.testing import absltest

import jax.numpy as jnp
from oryx.core.interpreters.inverse import core
from oryx.core.interpreters.inverse import custom_inverse
from oryx.core.interpreters.inverse import rules
from oryx.internal import test_util


# `rules` is only needed to populate the inverse registry.
del rules


class CustomInverseTest(test_util.TestCase):

  def test_unary_inverse(self):

    @custom_inverse.custom_inverse
    def add_one(x):
      return x + 1.

    self.assertEqual(core.inverse(add_one)(1.), 0.)

  def test_noninvertible_error_should_cause_unary_inverse_to_fail(self):

    @custom_inverse.custom_inverse
    def add_one(x):
      return x + 1.

    def add_one_inv(_):
      raise custom_inverse.NonInvertibleError()

    add_one.def_inverse_unary(add_one_inv)

    with self.assertRaises(ValueError):
      core.inverse(add_one)(1.)

  def test_unary_ildj(self):

    @custom_inverse.custom_inverse
    def add_one(x):
      return x + 1.

    def add_one_ildj(y):
      del y
      return 4.

    add_one.def_inverse_unary(f_ildj=add_one_ildj)
    self.assertEqual(core.inverse(add_one)(2.), 1.)
    self.assertEqual(core.ildj(add_one)(2.), 4.)

  def test_unary_inverse_and_ildj(self):

    @custom_inverse.custom_inverse
    def add_one(x):
      return x + 1.

    def add_one_inv(y):
      return jnp.exp(y)

    def add_one_ildj(y):
      del y
      return 4.

    add_one.def_inverse_unary(add_one_inv, add_one_ildj)
    self.assertEqual(core.inverse(add_one)(2.), jnp.exp(2.))
    self.assertEqual(core.ildj(add_one)(2.), 4.)

  def test_binary_inverse_and_ildj(self):

    @custom_inverse.custom_inverse
    def add(x, y):
      return x + y

    def add_ildj(invals, z, z_ildj):
      x, y = invals
      if z is None:
        raise custom_inverse.NonInvertibleError()
      if x is not None and y is None:
        return (x, z - x), (jnp.zeros_like(z_ildj), z_ildj)
      elif x is None and y is not None:
        return (z - y, y), (z_ildj, jnp.zeros_like(z_ildj))
      return (None, None), (0., 0.)

    add_one_left = functools.partial(add, 1.)
    add_one_right = lambda x: add(x, 1.)
    self.assertEqual(core.inverse(add_one_left)(2.), 1.)
    add.def_inverse_and_ildj(add_ildj)
    self.assertEqual(core.inverse(add_one_left)(2.), 1.)
    self.assertEqual(core.inverse(add_one_right)(2.), 1.)

    def add_ildj2(invals, z, z_ildj):
      x, y = invals
      if z is None:
        raise custom_inverse.NonInvertibleError()
      if x is not None and y is None:
        return (x, z**2), (jnp.zeros_like(z_ildj), 3. + z_ildj)
      elif x is None and y is not None:
        return (z / 4, y), (2. + z_ildj, jnp.zeros_like(z_ildj))
      return (None, None), (0., 0.)

    add.def_inverse_and_ildj(add_ildj2)
    self.assertEqual(core.inverse(add_one_left)(2.), 4.)
    self.assertEqual(core.ildj(add_one_left)(2.), 3.)

    self.assertEqual(core.inverse(add_one_right)(2.), 0.5)
    self.assertEqual(core.ildj(add_one_right)(2.), 2.)

  def test_noninvertible_error_should_cause_binary_inverse_to_fail(self):

    @custom_inverse.custom_inverse
    def add(x, y):
      return x + y

    def add_ildj(invals, z, z_ildj):
      x, y = invals
      if z is None:
        raise custom_inverse.NonInvertibleError()
      if x is not None and y is None:
        return (x, z - x), (jnp.zeros_like(z_ildj), z_ildj)
      elif x is None and y is not None:
        # Cannot invert if we don't know x
        raise custom_inverse.NonInvertibleError()
      return (None, None), (0., 0.)

    add.def_inverse_and_ildj(add_ildj)

    core.inverse(lambda x: add(1., x))(2.)
    with self.assertRaises(ValueError):
      core.inverse(lambda x: add(x, 1.))(2.)

  def test_inverse_with_tuple_inputs(self):

    @custom_inverse.custom_inverse
    def dense(params, x):
      w, b = params
      return w * x + b

    def dense_ildj(invals, out, out_ildj):
      (w, b), _ = invals
      if w is None or b is None:
        raise custom_inverse.NonInvertibleError()
      in_ildj = core.ildj(lambda x: w * x + b)(out)
      return ((w, b), (out - b) / w), ((jnp.zeros_like(w), jnp.zeros_like(b)),
                                       out_ildj + in_ildj)

    dense_apply = functools.partial(dense, (2., 1.))
    self.assertEqual(core.inverse(dense_apply)(5.), 2.)
    dense.def_inverse_and_ildj(dense_ildj)
    self.assertEqual(core.inverse(dense_apply)(5.), 2.)

    def dense_ildj2(invals, out, out_ildj):
      (w, b), _ = invals
      if w is None and b is None:
        raise custom_inverse.NonInvertibleError()
      in_ildj = jnp.ones_like(out)
      return ((w, b), w * out), ((jnp.zeros_like(w), jnp.zeros_like(b)),
                                 out_ildj + in_ildj)

    dense.def_inverse_and_ildj(dense_ildj2)
    self.assertEqual(core.inverse(dense_apply)(5.), 10.)
    self.assertEqual(core.ildj(dense_apply)(5.), 1.)


if __name__ == '__main__':
  absltest.main()
