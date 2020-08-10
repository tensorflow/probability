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
"""Tests for tensorflow_probability.experimental.lazybones.utils."""

import math
from absl.testing import absltest
import tensorflow_probability as tfp

lb = tfp.experimental.lazybones


class DeferredUtilsTest(absltest.TestCase):

  def test_iter_edges(self):
    m = lb.DeferredInput(math)
    a = m.exp(1.)
    b = m.log(2.)
    c = a + b
    self.assertSetEqual(
        set([
            ('__call__', '__add__'),
            ('__call__', '__add__'),
            ('exp', '__call__'),
            ('math', 'exp'),
            ('log', '__call__'),
            ('math', 'log'),
        ]),
        set((p.name, v.name) for p, v in lb.utils.iter_edges(c)))

  def test_get_leaves(self):
    m = lb.DeferredInput(math)
    a = m.exp(1.)
    b = m.log(2.)
    self.assertSetEqual({a, b}, lb.utils.get_leaves(m))
    self.assertSetEqual({a, b}, lb.utils.get_leaves([a, b, a]))

  def test_get_roots(self):
    m = lb.DeferredInput(math)
    a = m.exp(1.)
    b = m.log(2.)
    c = a + b
    self.assertSetEqual({m}, lb.utils.get_roots(m))
    self.assertSetEqual({m}, lb.utils.get_roots(c))
    self.assertSetEqual({m}, lb.utils.get_roots([a, b]))

  def test_is_any_ancestor(self):
    o = lb.DeferredInput(math)
    n = lb.DeferredInput(math)
    m = lb.DeferredInput(math)
    a = m.exp(1.)
    b = m.log(2.)
    c = a + b  # pylint: disable=unused-variable
    self.assertFalse(lb.utils.is_any_ancestor(n, m))
    self.assertTrue(lb.utils.is_any_ancestor(a, m))
    self.assertTrue(lb.utils.is_any_ancestor([a, n], m))
    self.assertFalse(lb.utils.is_any_ancestor([o, n], m))
    self.assertTrue(lb.utils.is_any_ancestor([a, n], [m, o]))


if __name__ == '__main__':
  absltest.main()
