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
"""Tests for tensorflow_probability.spinoffs.oryx.core.interpreters.inverse."""
import os

from absl.testing import absltest
import jax
import jax.numpy as np
import numpy as onp

from oryx.core.interpreters import harvest
from oryx.core.interpreters.inverse import core
from oryx.core.interpreters.inverse import rules
del rules  # needed for registration only


class InverseTest(absltest.TestCase):

  def test_trivial_inverse(self):
    def f(x):
      return x
    f_inv = core.inverse(f)
    onp.testing.assert_allclose(f_inv(1.0), 1.0)

    def f2(x, y):
      return x, y
    f2_inv = core.inverse(f2)
    onp.testing.assert_allclose(f2_inv(1.0, 2.0), (1.0, 2.0))

  def test_mul_inverse(self):
    def f(x):
      return x * 2.
    f_inv = core.inverse(f)
    onp.testing.assert_allclose(f_inv(1.0), 0.5)

    def f2(x):
      return 2. * x
    f2_inv = core.inverse(f2)
    onp.testing.assert_allclose(f2_inv(1.0), 0.5)

  def test_div_inverse(self):
    def f(x):
      return x / 2.
    f_inv = core.inverse(f)
    onp.testing.assert_allclose(f_inv(1.0), 2.)

    def f2(x):
      return 2. / x
    f2_inv = core.inverse(f2)
    onp.testing.assert_allclose(f2_inv(1.0), 2.)

  def test_trivial_noninvertible(self):
    def f(x):
      del x
      return 1.
    with self.assertRaises(ValueError):
      core.inverse(f)(1.)

  def test_noninvertible(self):
    def f(x, y):
      return x + y, x + y
    with self.assertRaises(ValueError):
      core.inverse(f)(1., 2.)

  def test_simple_inverse(self):
    def f(x):
      return np.exp(x)
    f_inv = core.inverse(f, 0.1)
    onp.testing.assert_allclose(f_inv(1.0), 0.)

    def f2(x):
      return np.exp(x)
    f2_inv = core.inverse(f2, np.zeros(2))
    onp.testing.assert_allclose(f2_inv(np.ones(2)), np.zeros(2))

  def test_conditional_inverse(self):
    def f(x, y):
      return x + 1., np.exp(x + 1.) + y
    f_inv = core.inverse(f, 0., 2.)
    onp.testing.assert_allclose(f_inv(0., 2.), (-1., 1.))

  def test_simple_ildj(self):
    def f(x):
      return np.exp(x)
    f_inv = core.ildj(f, 0.1)
    onp.testing.assert_allclose(f_inv(2.0), -np.log(2.))

    def f2(x):
      return np.exp(x)
    f2_inv = core.ildj(f2, np.zeros(2))
    onp.testing.assert_allclose(f2_inv(2 * np.ones(2)), -2 * np.log(2.))

  def test_advanced_inverse_two(self):
    def f(x, y):
      return np.exp(x), x ** 2 + y
    f_inv = core.inverse(f, 0.1, 0.2)
    onp.testing.assert_allclose(f_inv(2.0, 2.0),
                                (np.log(2.), 2 - np.log(2.) ** 2))

  def test_advanced_inverse_three(self):
    def f(x, y, z):
      return np.exp(x), x ** 2 + y, np.exp(z + y)
    f_inv = core.inverse(f, 0., 0., 0.)
    onp.testing.assert_allclose(f_inv(2.0, 2.0, 2.0),
                                (np.log(2.), 2 - np.log(2.) ** 2,
                                 np.log(2.0) - (2 - np.log(2.) ** 2)))

  def test_mul_inverse_ildj(self):
    def f(x):
      return x * 2
    f_inv = core.inverse_and_ildj(f, 1.)
    x, ildj_ = f_inv(2.)
    onp.testing.assert_allclose(x, 1.)
    onp.testing.assert_allclose(ildj_, -np.log(np.abs(jax.jacrev(f)(1.))),
                                atol=1e-6, rtol=1e-6)

    def f2(x):
      return 2 * x
    f2_inv = core.inverse_and_ildj(f2, 1.)
    x, ildj_ = f2_inv(2.)
    onp.testing.assert_allclose(x, 1.)
    onp.testing.assert_allclose(ildj_, -np.log(np.abs(jax.jacrev(f)(1.))),
                                atol=1e-6, rtol=1e-6)

  def test_lower_triangular_jacobian(self):
    def f(x, y):
      return x + 2., np.exp(x) + y
    def f_vec(x):
      return np.array([x[0] + 2., np.exp(x[0]) + x[1]])
    f_inv = core.inverse_and_ildj(f, 0., 0.)
    x, ildj_ = f_inv(3., np.exp(1.) + 1.)
    onp.testing.assert_allclose(x, (1., 1.))
    onp.testing.assert_allclose(ildj_, -np.log(
        np.abs(np.linalg.slogdet(jax.jacrev(f_vec)(np.ones(2)))[0])),
                                atol=1e-6, rtol=1e-6)

  def test_div_inverse_ildj(self):
    def f(x):
      return x / 2
    f_inv = core.inverse_and_ildj(f, 2.)
    x, ildj_ = f_inv(2.)
    onp.testing.assert_allclose(x, 4.)
    onp.testing.assert_allclose(ildj_, -np.log(np.abs(jax.jacrev(f)(4.))),
                                atol=1e-6, rtol=1e-6)

    def f2(x):
      return 3. / x
    f2_inv = core.inverse_and_ildj(f2, 2.)
    x, ildj_ = f2_inv(2.)
    onp.testing.assert_allclose(x, 1.5)
    onp.testing.assert_allclose(ildj_, -np.log(np.abs(jax.jacrev(f2)(1.5))),
                                atol=1e-6, rtol=1e-6)

  def test_inverse_of_jit(self):
    def f(x):
      x = jax.jit(lambda x: x)(x)
      return x / 2.
    f_inv = core.inverse_and_ildj(f, 2.)
    x, ildj_ = f_inv(2.)
    onp.testing.assert_allclose(x, 4.)
    onp.testing.assert_allclose(ildj_, -np.log(np.abs(jax.jacrev(f)(4.))),
                                atol=1e-6, rtol=1e-6)

    def f2(x):
      return jax.jit(lambda x: 3. / x)(x)
    f2_inv = core.inverse_and_ildj(f2, 2.)
    x, ildj_ = f2_inv(2.)
    onp.testing.assert_allclose(x, 1.5)
    onp.testing.assert_allclose(ildj_, -np.log(np.abs(jax.jacrev(f2)(1.5))),
                                atol=1e-6, rtol=1e-6)

  def test_inverse_of_pmap(self):
    def f(x):
      return jax.pmap(lambda x: np.exp(x) + 2.)(x)
    f_inv = core.inverse_and_ildj(f, np.ones(2) * 4)
    x, ildj_ = f_inv(np.ones(2) * 4)
    onp.testing.assert_allclose(x, np.log(2.) * np.ones(2))
    onp.testing.assert_allclose(ildj_,
                                -np.log(np.abs(np.sum(jax.jacrev(f)(
                                    np.log(2.) * np.ones(2))))),
                                atol=1e-6, rtol=1e-6)

  def test_pmap_forward(self):
    def f(x, y):
      z = jax.pmap(np.exp)(x)
      return x + 2., z + y
    def f_vec(x):
      return np.array([x[0] + 2., np.exp(x[0]) + x[1]])
    f_inv = core.inverse_and_ildj(f, np.ones(2), np.ones(2))
    x, ildj_ = f_inv(2 * np.ones(2), np.ones(2))
    onp.testing.assert_allclose(x, (np.zeros(2), np.zeros(2)))
    onp.testing.assert_allclose(ildj_, -np.log(
        np.abs(np.linalg.slogdet(jax.jacrev(f_vec)(np.ones(2)))[0])),
                                atol=1e-6, rtol=1e-6)

  def test_inverse_of_sow_is_identity(self):
    def f(x):
      return harvest.sow(x, name='x', tag='foo')
    x, ildj_ = core.inverse_and_ildj(f, 1.)(1.)
    self.assertEqual(x, 1.)
    self.assertEqual(ildj_, 0.)

  def test_inverse_of_nest(self):
    def f(x):
      x = harvest.nest(lambda x: x, scope='foo')(x)
      return x / 2.
    f_inv = core.inverse_and_ildj(f, 2.)
    x, ildj_ = f_inv(2.)
    onp.testing.assert_allclose(x, 4.)
    onp.testing.assert_allclose(ildj_, -np.log(np.abs(jax.jacrev(f)(4.))),
                                atol=1e-6, rtol=1e-6)

  def test_inverse_of_split(self):
    def f(x):
      return np.split(x, 2)
    f_inv = core.inverse_and_ildj(f, np.ones(4))
    x, ildj_ = f_inv([np.ones(2), np.ones(2)])
    onp.testing.assert_allclose(x, np.ones(4))
    onp.testing.assert_allclose(ildj_, 0., atol=1e-6, rtol=1e-6)

  def test_inverse_of_concatenate(self):
    def f(x, y):
      return np.concatenate([x, y])
    f_inv = core.inverse_and_ildj(f, np.ones(2), np.ones(2))
    (x, y), ildj_ = f_inv(np.ones(4))
    onp.testing.assert_allclose(x, np.ones(2))
    onp.testing.assert_allclose(y, np.ones(2))
    onp.testing.assert_allclose(ildj_, 0., atol=1e-6, rtol=1e-6)

if __name__ == '__main__':
  os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=2'
  absltest.main()
