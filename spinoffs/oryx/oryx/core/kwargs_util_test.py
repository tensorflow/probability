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
"""Tests for tensorflow_probability.spinoffs.oryx.core.kwargs_util."""
from absl.testing import absltest

import jax
from oryx.core import kwargs_util
from oryx.core.interpreters.inverse import custom_inverse


class KwargsUtilTest(absltest.TestCase):

  def test_filter_kwargs(self):
    kwargs = dict(
        rng=1,
        training=True
    )
    def foo1(x, y):
      return x + y
    self.assertDictEqual(
        kwargs_util.filter_kwargs(foo1, kwargs),
        {})
    def foo2(x, y, rng=None):
      del rng
      return x + y
    self.assertDictEqual(
        kwargs_util.filter_kwargs(foo2, kwargs),
        {'rng': 1})
    def foo3(x, y, rng=None, training=False):
      del rng, training
      return x + y
    self.assertDictEqual(
        kwargs_util.filter_kwargs(foo3, kwargs),
        {'rng': 1, 'training': True})

  def test_filter_kwargs_accepts_all(self):
    kwargs = dict(
        rng=1,
        training=True
    )
    def foo1(x, y, **kwargs):
      del kwargs
      return x + y
    self.assertDictEqual(
        kwargs_util.filter_kwargs(foo1, kwargs),
        {'rng': 1, 'training': True})
    def foo2(x, y, training=False, **kwargs):
      del training, kwargs
      return x + y
    self.assertDictEqual(
        kwargs_util.filter_kwargs(foo2, kwargs),
        {'rng': 1, 'training': True})

  def test_filter_incomplete_kwargs(self):
    kwargs = dict(
        rng=1,
    )
    def foo1(x, y):
      return x + y
    self.assertDictEqual(
        kwargs_util.filter_kwargs(foo1, kwargs),
        {})
    def foo2(x, y, rng=None):
      del rng
      return x + y
    self.assertDictEqual(
        kwargs_util.filter_kwargs(foo2, kwargs),
        {'rng': 1})
    def foo3(x, y, rng=None, training=False):
      del rng, training
      return x + y
    self.assertDictEqual(
        kwargs_util.filter_kwargs(foo3, kwargs),
        {'rng': 1})

  def test_check_kwargs(self):
    def foo1(x, y):
      return x + y
    self.assertFalse(
        kwargs_util.check_in_kwargs(foo1, 'rng'))
    self.assertFalse(
        kwargs_util.check_in_kwargs(foo1, 'training'))
    def foo2(x, y, rng=None):
      del rng
      return x + y
    self.assertTrue(
        kwargs_util.check_in_kwargs(foo2, 'rng'))
    self.assertFalse(
        kwargs_util.check_in_kwargs(foo2, 'training'))
    def foo3(x, y, rng=None, training=False):
      del rng, training
      return x + y
    self.assertTrue(
        kwargs_util.check_in_kwargs(foo3, 'rng'))
    self.assertTrue(
        kwargs_util.check_in_kwargs(foo3, 'training'))
    def foo4(x, y, *, rng, training):
      del rng, training
      return x + y
    self.assertTrue(
        kwargs_util.check_in_kwargs(foo4, 'rng'))
    self.assertTrue(
        kwargs_util.check_in_kwargs(foo4, 'training'))

  def test_check_custom_jvp(self):
    self.assertFalse(kwargs_util.check_in_kwargs(jax.nn.relu, 'foo'))

  def test_check_custom_inverse(self):
    @custom_inverse.custom_inverse
    def f(x, bar=2):
      return x + bar
    self.assertFalse(kwargs_util.check_in_kwargs(f, 'foo'))
    self.assertTrue(kwargs_util.check_in_kwargs(f, 'bar'))

if __name__ == '__main__':
  absltest.main()
